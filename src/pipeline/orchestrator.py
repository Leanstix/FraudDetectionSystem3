from __future__ import annotations

from pathlib import Path

from src.agents.audio_context_agent import AudioContextAgent
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
from src.data.dataset_inspector import inspect_pair
from src.data.entity_resolution import EntityResolver
from src.data.feature_store import FeatureStore
from src.data.loaders import DatasetLoader
from src.data.normalize import Normalizer
from src.llm.audio_reasoner import AudioReasoner
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
        input_path: str,
        output_path: str,
        dataset_name: str,
        no_llm: bool = False,
        verbose: bool = False,
    ):
        self.settings = settings
        self.reference_path = reference_path
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.no_llm = no_llm
        self.verbose = verbose

        if no_llm:
            self.settings.llm_enabled = False

        self.normalizer = Normalizer()
        self.feature_store = FeatureStore()

        self.resolver = EntityResolver()
        self.entity_agent = EntityResolutionAgent(self.resolver)

        self.tracing = TracingManager(settings)
        self.llm_client = LLMClient(settings, self.tracing)
        self.communication_analyzer = CommunicationAnalyzer(self.llm_client)
        self.audio_reasoner = AudioReasoner(self.llm_client, self.settings)

        self.transaction_agent = TransactionBehaviorAgent()
        self.temporal_agent = TemporalSequenceAgent()
        self.geo_agent = GeoSpatialAgent()
        self.communication_agent = CommunicationRiskAgent(self.communication_analyzer)
        self.audio_agent = AudioContextAgent(self.audio_reasoner)
        self.novelty_agent = NoveltyDriftAgent()

        self.fusion_agent = FusionDecisionAgent(settings)
        self.submission_writer = SubmissionWriter()

    def _ingest(self, input_path: str) -> tuple[dict, DatasetLoader]:
        loader = DatasetLoader(input_path)
        agent = DataIngestionAgent(loader, self.normalizer)
        return agent.run(), loader

    def inspect_pair(self) -> dict:
        return inspect_pair(self.reference_path, self.input_path)

    def run_pair(self) -> PipelineArtifacts:
        session_id = None
        if self.tracing.is_enabled():
            session_id = self.tracing.generate_session_id()

        # 1. load + normalize TRAIN
        reference_data, reference_loader = self._ingest(self.reference_path)
        reference_paths = reference_loader.load_all()

        # 2. load + normalize VALIDATION
        target_data, target_loader = self._ingest(self.input_path)
        target_paths = target_loader.load_all()

        # 3. entity resolution on target
        entity_out = self.entity_agent.run(
            transactions_df=target_data["transactions"],
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
            audio_df=target_data["audio"],
        )
        enriched_transactions = entity_out["enriched_transactions_df"]

        # 4. build features using TRAIN baseline and VALIDATION target
        features_df = self.feature_store.build_all(
            reference_transactions_df=reference_data["transactions"],
            target_transactions_df=enriched_transactions,
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
            audio_df=target_data["audio"],
        )

        # 5. run specialist agents
        transaction_scores = self.transaction_agent.run(features_df)
        temporal_scores = self.temporal_agent.run(features_df)
        geo_scores = self.geo_agent.run(features_df)
        comm_scores = self.communication_agent.run(features_df, target_data["sms"], target_data["mails"], self.dataset_name)
        audio_scores = self.audio_agent.run(features_df, target_data["audio"], self.dataset_name)
        novelty_scores = self.novelty_agent.run(features_df)

        outputs = [
            transaction_scores,
            temporal_scores,
            geo_scores,
            comm_scores,
            audio_scores,
            novelty_scores,
        ]

        # 6. fuse scores
        merged = self.fusion_agent.merge_scores(features_df, outputs)
        scored = self.fusion_agent.compute_final_score(merged)
        threshold = self.fusion_agent.choose_threshold(scored)
        flagged_full = self.fusion_agent.flag_transactions(scored, threshold)

        # 7. validate + write submission
        submission_input = flagged_full[["transaction_id", "final_risk_score", "flagged", "top_risk_reasons"]].copy()
        submission = self.submission_writer.run(submission_input, enriched_transactions, self.output_path)

        # 8. write diagnostics
        output = Path(self.output_path)
        diag_name = output.stem.replace("_submission", "") + "_diagnostics.csv"
        diagnostics_path = str(output.with_name(diag_name))
        diagnostics_df = flagged_full.sort_values(["final_risk_score", "transaction_id"], ascending=[False, True]).copy()
        write_csv(diagnostics_path, diagnostics_df)

        # 9. flush tracing
        self.tracing.flush()

        reference_bundle = DatasetBundle(
            paths=reference_paths,
            transactions_df=reference_data["transactions"],
            users_df=reference_data["users"],
            locations_df=reference_data["locations"],
            sms_df=reference_data["sms"],
            mails_df=reference_data["mails"],
            audio_df=reference_data["audio"],
        )
        target_bundle = DatasetBundle(
            paths=target_paths,
            transactions_df=enriched_transactions,
            users_df=target_data["users"],
            locations_df=target_data["locations"],
            sms_df=target_data["sms"],
            mails_df=target_data["mails"],
            audio_df=target_data["audio"],
        )

        artifacts = PipelineArtifacts(
            dataset_name=self.dataset_name,
            reference_bundle=reference_bundle,
            target_bundle=target_bundle,
            reference_transaction_count=int(len(reference_data["transactions"])),
            target_transaction_count=int(len(enriched_transactions)),
            transactions_df=enriched_transactions,
            features_df=features_df,
            final_scores_df=flagged_full,
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

    def run(self) -> PipelineArtifacts:
        return self.run_pair()

    def inspect(self) -> dict:
        return self.inspect_pair()

    def print_summary(self, artifacts: PipelineArtifacts) -> None:
        total = artifacts.submission.total_transactions
        flagged = artifacts.submission.flagged_count
        pct = (100.0 * flagged / total) if total else 0.0

        print(f"dataset name: {artifacts.dataset_name}")
        print(f"reference transaction count: {artifacts.reference_transaction_count}")
        print(f"target transaction count: {artifacts.target_transaction_count}")
        print(f"total flagged count: {flagged}")
        print(f"flagged percentage: {pct:.2f}%")
        print(f"tracing enabled: {artifacts.tracing_enabled}")
        if artifacts.tracing_enabled and artifacts.session_id:
            print(f"session ID: {artifacts.session_id}")
        print(f"output path: {artifacts.submission.output_path}")
        print(f"diagnostics path: {artifacts.diagnostics_path}")
