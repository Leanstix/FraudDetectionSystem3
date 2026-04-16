from __future__ import annotations


def communication_risk_prompt(thread_text: str) -> str:
    return (
        "You are a fraud-risk communication analyst. "
        "Given this communication thread, return compact JSON only with keys: "
        "risk_score, urgency_score, payment_bait_score, credential_theft_score, explanation. "
        "All scores must be floats from 0 to 1. "
        "No markdown, no prose before/after JSON.\n\n"
        f"THREAD:\n{thread_text[:12000]}"
    )


def audio_summary_prompt(audio_context: str) -> str:
    return (
        "You are analyzing an audio event in a fraud investigation. "
        "Return compact JSON only with keys: risk_score, confidence, explanation. "
        "Scores must be floats 0..1 and explanation must be <= 220 chars. "
        "No markdown.\n\n"
        f"AUDIO_CONTEXT:\n{audio_context[:12000]}"
    )
