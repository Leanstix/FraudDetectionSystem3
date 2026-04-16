from __future__ import annotations

from pathlib import Path

from src.data.loaders import DatasetLoader, PairedDatasetLoader
from src.data.schemas import REQUIRED_FILE_NAMES, validate_json_list, validate_transactions_schema


def _reference_zip() -> str:
    path = Path("The Truman Show - train.zip")
    if not path.exists():
        raise FileNotFoundError("The Truman Show - train.zip not found")
    return str(path)


def _target_zip() -> str:
    path = Path("The Truman Show - validation.zip")
    if not path.exists():
        raise FileNotFoundError("The Truman Show - validation.zip not found")
    return str(path)


def test_reference_and_target_dataset_zips_load():
    paired = PairedDatasetLoader(_reference_zip(), _target_zip())
    reference_paths, target_paths = paired.load_both()

    assert Path(reference_paths.dataset_root).exists()
    assert Path(target_paths.dataset_root).exists()


def test_required_files_found():
    ref_loader = DatasetLoader(_reference_zip())
    tgt_loader = DatasetLoader(_target_zip())

    ref_root = Path(ref_loader.load_all().dataset_root)
    tgt_root = Path(tgt_loader.load_all().dataset_root)

    ref_files = {p.name for p in ref_root.iterdir() if p.is_file()}
    tgt_files = {p.name for p in tgt_root.iterdir() if p.is_file()}

    assert REQUIRED_FILE_NAMES.issubset(ref_files)
    assert REQUIRED_FILE_NAMES.issubset(tgt_files)


def test_schemas_validate():
    loader = DatasetLoader(_target_zip())
    tx = loader.load_transactions()
    users = loader.load_users()
    locations = loader.load_locations()
    sms = loader.load_sms()
    mails = loader.load_mails()

    validate_transactions_schema(tx)
    validate_json_list("users.json", users, {"first_name", "last_name", "iban", "residence"})
    validate_json_list("locations.json", locations, {"biotag", "timestamp", "lat", "lng", "city"})
    validate_json_list("sms.json", sms, {"sms"})
    validate_json_list("mails.json", mails, {"mail"})
