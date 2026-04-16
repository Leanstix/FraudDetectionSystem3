from __future__ import annotations

from pathlib import Path

from src.data.baseline_store import BaselineStore
from src.data.loaders import DatasetLoader
from src.data.normalize import Normalizer


def _load_bundle(path: str) -> dict:
    loader = DatasetLoader(path)
    normalizer = Normalizer()
    return {
        "transactions": normalizer.normalize_transactions(loader.load_transactions()),
        "users": normalizer.normalize_users(loader.load_users()),
        "locations": normalizer.normalize_locations(loader.load_locations()),
        "sms": normalizer.normalize_sms(loader.load_sms()),
        "mails": normalizer.normalize_mails(loader.load_mails()),
    }


def test_baseline_fits_reference_dataset():
    reference_zip = Path("The Truman Show - train.zip")
    baseline = BaselineStore()

    ref = _load_bundle(str(reference_zip))
    baseline.fit(ref["transactions"], ref["users"], ref["locations"], ref["sms"], ref["mails"])

    assert not baseline.build_sender_profiles().empty
    assert not baseline.build_recipient_profiles().empty
    assert not baseline.build_pair_profiles().empty


def test_target_transform_produces_delta_features():
    reference_zip = Path("The Truman Show - train.zip")
    target_zip = Path("The Truman Show - validation.zip")

    baseline = BaselineStore()
    ref = _load_bundle(str(reference_zip))
    tgt = _load_bundle(str(target_zip))

    baseline.fit(ref["transactions"], ref["users"], ref["locations"], ref["sms"], ref["mails"])
    transformed = baseline.transform_target(tgt["transactions"])

    expected = {
        "transaction_id",
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
    }
    assert expected.issubset(set(transformed.columns))
    assert len(transformed) == len(tgt["transactions"])
