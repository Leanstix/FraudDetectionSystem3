from __future__ import annotations

from pathlib import Path

from src.agents.baseline_builder_agent import BaselineBuilderAgent
from src.agents.communication_risk_agent import CommunicationRiskAgent
from src.agents.entity_resolution_agent import EntityResolutionAgent
from src.agents.fusion_decision_agent import FusionDecisionAgent
from src.agents.geospatial_agent import GeoSpatialAgent
from src.agents.ingestion_agent import DataIngestionAgent
from src.agents.novelty_drift_agent import NoveltyDriftAgent
from src.agents.submission_writer import SubmissionWriter
from src.agents.temporal_sequence_agent import TemporalSequenceAgent
from src.agents.transaction_behavior_agent import TransactionBehaviorAgent
from src.config import Settings
from src.data.baseline_store import BaselineStore
from src.data.dataset_inspector import inspect_pair
from src.data.entity_resolution import EntityResolver
from src.data.feature_store import FeatureStore
from src.data.loaders import DatasetLoader, PairedDatasetLoader
from src.data.normalize import Normalizer
from src.llm.client import LLMClient
from src.llm.communication_analyzer import CommunicationAnalyzer
from src.tracing import TracingManager
from src.types import DatasetBundle, PipelineArtifacts
from src.utils.io import write_csv


class FraudPipeline:
    def __init__(
        self,
        settings: Settings,
        reference_path: str,
        target_path: str,
        output_path: str,
        dataset_name: str,
        no_llm: bool = False,
        verbose: bool = False,
    ):
        self.settings = settings
        self.reference_path = reference_path
        self.target_path = target_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.no_llm = no_llm
        self.verbose = verbose

        if no_llm:
            self.settings.llm_enabled = False

        self.normalizer = Normalizer()
        self.paired_loader = PairedDatasetLoader(reference_path, target_path)

        self.baseline_store = BaselineStore()
        self.baseline_agent = BaselineBuilderAgent(self.baseline_store)

        self.resolver = EntityResolver()
        self.entity_agent = EntityResolutionAgent(self.resolver)
        self.feature_store = FeatureStore(self.baseline_store)

        self.tracing = TracingManager(settings)
        self.llm_client = LLMClient(settings, self.tracing)
        self.communication_analyzer = CommunicationAnalyzer(self.llm_client)

        self.transaction_agent = TransactionBehaviorAgent()
        self.temporal_agent = TemporalSequenceAgent()
        self.geo_agent = GeoSpatialAgent()
        self.communication_agent = CommunicationRiskAgent(self.communication_analyzer)
        self.novelty_agent = NoveltyDriftAgent()

        self.fusion_agent = FusionDecisionAgent(settings)
        self.submission_writer = SubmissionWriter()

    def _ingest(self, input_path: str) -> tuple[dict, DatasetLoader]:
        loader = DatasetLoader(input_path)
        agent = DataIngestionAgent(loader, self.normalizer)
        return agent.run(), loader

    def inspect(self) -> dict:
        return inspect_pair(self.reference_path, self.target_path)

    def run(self) -> PipelineArtifacts:
        session_id = None
        if self.tracing.is_enabled():
            session_id = self.tracing.generate_session_id()

        # 1. load + normalize reference dataset
        reference_data, reference_loader = self._ingest(self.reference_path)
        reference_paths = reference_loader.load_all()

        # 2. load + normalize target dataset
        target_data, target_loader = self._ingest(self.target_path)
        target_paths = target_loader.load_all()

        # 3. build baseline from reference/train
        baseline_artifacts = self.baseline_agent.run(reference_data)

        # 4. entity resolution on target
        entity_out = self.entity_agent.run(
            transactions_df=target_data["transactions"],
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
        )
        enriched_transactions = entity_out["enriched_transactions_df"]

        # 5. feature building on target with baseline deltas
        features_df = self.feature_store.build_all(
            transactions_df=enriched_transactions,
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
        )

        # 6. run all specialist agents
        transaction_scores = self.transaction_agent.run(features_df)
        temporal_scores = self.temporal_agent.run(features_df)
        geo_scores = self.geo_agent.run(features_df)
        comm_scores = self.communication_agent.run(features_df, target_data["sms"], target_data["mails"], self.dataset_name)
        novelty_scores = self.novelty_agent.run(features_df)

        outputs = [transaction_scores, temporal_scores, geo_scores, comm_scores, novelty_scores]

        # 7. fuse scores
        merged = self.fusion_agent.merge_scores(features_df, outputs)
        scored = self.fusion_agent.compute_final_score(merged)
        threshold = self.fusion_agent.choose_threshold(scored)
        flagged_full = self.fusion_agent.flag_transactions(scored, threshold)

        # 8. validate + write submission
        submission_input = flagged_full[["transaction_id", "final_risk_score", "flagged", "top_risk_reasons"]].copy()
        submission = self.submission_writer.run(submission_input, enriched_transactions, self.output_path)

        # 9. write diagnostics file
        output = Path(self.output_path)
        diag_name = output.stem.replace("_submission", "") + "_diagnostics.csv"
        diagnostics_path = str(output.with_name(diag_name))
        diagnostics_df = flagged_full.sort_values(["final_risk_score", "transaction_id"], ascending=[False, True]).copy()
        write_csv(diagnostics_path, diagnostics_df)

        # 10. flush tracing
        self.tracing.flush()

        reference_bundle = DatasetBundle(
            paths=reference_paths,
            transactions_df=reference_data["transactions"],
            users_df=reference_data["users"],
            locations_df=reference_data["locations"],
            sms_df=reference_data["sms"],
            mails_df=reference_data["mails"],
        )
        target_bundle = DatasetBundle(
            paths=target_paths,
            transactions_df=enriched_transactions,
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
        )

        artifacts = PipelineArtifacts(
            dataset_name=self.dataset_name,
            reference_bundle=reference_bundle,
            target_bundle=target_bundle,
            transactions_df=enriched_transactions,
            features_df=features_df,
            final_scores_df=flagged_full,
            baseline_artifacts=baseline_artifacts,
            submission=submission,
            diagnostics_path=diagnostics_path,
            tracing_enabled=self.tracing.is_enabled(),
            session_id=session_id,
            metadata={
                "entity_resolution": entity_out.get("linking_diagnostics", {}),
                "threshold": threshold,
            },
        )
        return artifacts

    def print_summary(self, artifacts: PipelineArtifacts) -> None:
        total = artifacts.submission.total_transactions
        flagged = artifacts.submission.flagged_count
        pct = (100.0 * flagged / total) if total else 0.0

        print(f"dataset name: {artifacts.dataset_name}")
        print(f"reference path: {self.reference_path}")
        print(f"target path: {self.target_path}")
        print(f"total transaction count: {total}")
        print(f"total flagged count: {flagged}")
        print(f"flagged percentage: {pct:.2f}%")
        print(f"tracing enabled: {artifacts.tracing_enabled}")
        if artifacts.tracing_enabled and artifacts.session_id:
            print(f"generated session ID: {artifacts.session_id}")
        print(f"output submission path: {artifacts.submission.output_path}")
        print(f"diagnostics path: {artifacts.diagnostics_path}")
