from __future__ import annotations

REQUIRED_FILES = {
    "transactions": "transactions.csv",
    "users": "users.json",
    "locations": "locations.json",
    "sms": "sms.json",
    "mails": "mails.json",
}

REQUIRED_DIRS = {
    "audio": "audio",
}

TRANSACTION_REQUIRED_COLUMNS = [
    "transaction_id",
    "sender_id",
    "recipient_id",
    "transaction_type",
    "amount",
    "location",
    "payment_method",
    "sender_iban",
    "recipient_iban",
    "balance_after",
    "description",
    "timestamp",
]

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "suspended",
    "locked",
    "customs",
    "confirm payment",
    "click",
    "password",
    "account",
    "security",
    "action required",
    "benefit",
    "social security",
]

PHISHING_DOMAIN_HINTS = [
    "paypa1",
    "amaz0n",
    "netfl1x",
    "secure-pay",
    "verify",
    "billing",
]

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RANDOM_SEED = 42

DEFAULT_AUDIO_HALF_LIFE_HOURS = 72.0
DEFAULT_AUDIO_MATCH_WINDOW_HOURS = 168.0

BASELINE_DELTA_COLUMNS = [
    "ref_sender_amount_robust_z",
    "ref_recipient_amount_robust_z",
    "pair_seen_in_reference",
    "payment_method_seen_by_sender_ref",
    "transaction_type_seen_by_sender_ref",
    "reference_hour_rarity",
    "reference_weekday_rarity",
    "unseen_transaction_type_indicator",
    "unseen_payment_method_indicator",
    "unseen_location_pattern_indicator",
]
