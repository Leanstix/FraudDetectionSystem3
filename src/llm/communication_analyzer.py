from __future__ import annotations

import json
import re

import numpy as np

from src.constants import PHISHING_DOMAIN_HINTS, SUSPICIOUS_KEYWORDS
from src.llm.client import LLMClient
from src.llm.prompts import communication_risk_prompt


def _normalize_scores(payload: dict) -> dict:
    base = {
        "risk_score": 0.0,
        "urgency_score": 0.0,
        "payment_bait_score": 0.0,
        "credential_theft_score": 0.0,
        "explanation": "",
    }
    for key in ["risk_score", "urgency_score", "payment_bait_score", "credential_theft_score"]:
        try:
            base[key] = float(np.clip(float(payload.get(key, 0.0)), 0.0, 1.0))
        except Exception:
            base[key] = 0.0
    base["explanation"] = str(payload.get("explanation", "")).strip()[:300]
    return base


class CommunicationAnalyzer:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze_sms_thread(self, sms_text: str, dataset_name: str) -> dict:
        heuristic = self.heuristic_score_sms(sms_text)
        return self._maybe_enrich_with_llm(sms_text, dataset_name, "sms_thread", heuristic)

    def analyze_mail_thread(self, mail_text: str, dataset_name: str) -> dict:
        heuristic = self.heuristic_score_mail(mail_text)
        return self._maybe_enrich_with_llm(mail_text, dataset_name, "mail_thread", heuristic)

    def heuristic_score_sms(self, sms_text: str) -> dict:
        return self._heuristic_score(sms_text, source="sms")

    def heuristic_score_mail(self, mail_text: str) -> dict:
        return self._heuristic_score(mail_text, source="mail")

    def _heuristic_score(self, text: str, source: str) -> dict:
        t = (text or "").lower()
        keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in t)
        domain_hits = sum(1 for hint in PHISHING_DOMAIN_HINTS if hint in t)
        urgency_hits = sum(1 for kw in ["urgent", "immediate", "suspend", "locked", "within 24h", "action required"] if kw in t)
        credential_hits = sum(1 for kw in ["password", "verify identity", "login", "credential", "confirm account"] if kw in t)
        payment_hits = sum(1 for kw in ["payment", "card", "paypal", "invoice", "renewal"] if kw in t)
        has_url = 1 if ("http://" in t or "https://" in t) else 0

        risk = np.clip(0.08 * keyword_hits + 0.18 * domain_hits + 0.14 * has_url + 0.12 * urgency_hits + 0.12 * credential_hits, 0.0, 1.0)
        urgency = np.clip(0.2 * urgency_hits + 0.15 * has_url, 0.0, 1.0)
        payment = np.clip(0.15 * payment_hits + 0.2 * domain_hits, 0.0, 1.0)
        cred = np.clip(0.18 * credential_hits + 0.2 * domain_hits + 0.1 * has_url, 0.0, 1.0)

        return {
            "risk_score": float(risk),
            "urgency_score": float(urgency),
            "payment_bait_score": float(payment),
            "credential_theft_score": float(cred),
            "explanation": f"heuristic_{source}: kw={keyword_hits}, domain={domain_hits}, url={has_url}",
        }

    def _maybe_enrich_with_llm(self, text: str, dataset_name: str, task_name: str, heuristic: dict) -> dict:
        heuristic = _normalize_scores(heuristic)
        risk = heuristic["risk_score"]

        # Use LLM only for ambiguous or moderately suspicious cases.
        ambiguous = 0.35 <= risk <= 0.8
        if not ambiguous or not self.llm_client.is_enabled():
            return heuristic

        prompt = communication_risk_prompt(text)
        raw = self.llm_client.invoke(prompt, dataset_name=dataset_name, task_name=task_name)
        if not raw:
            return heuristic

        parsed = self._parse_json_like(raw)
        if not parsed:
            return heuristic

        llm_scores = _normalize_scores(parsed)
        blended = {
            "risk_score": float(np.clip(0.55 * heuristic["risk_score"] + 0.45 * llm_scores["risk_score"], 0.0, 1.0)),
            "urgency_score": float(np.clip(0.55 * heuristic["urgency_score"] + 0.45 * llm_scores["urgency_score"], 0.0, 1.0)),
            "payment_bait_score": float(np.clip(0.55 * heuristic["payment_bait_score"] + 0.45 * llm_scores["payment_bait_score"], 0.0, 1.0)),
            "credential_theft_score": float(np.clip(0.55 * heuristic["credential_theft_score"] + 0.45 * llm_scores["credential_theft_score"], 0.0, 1.0)),
            "explanation": (heuristic["explanation"] + " | llm:" + llm_scores["explanation"]).strip()[:300],
        }
        return blended

    @staticmethod
    def _parse_json_like(raw: str) -> dict | None:
        text = raw.strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        blob = match.group(0)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            # Lightweight repair for single quotes.
            repaired = blob.replace("'", '"')
            try:
                return json.loads(repaired)
            except Exception:
                return None
