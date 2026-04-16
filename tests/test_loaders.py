from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.data.loaders import DatasetLoader, PairedDatasetLoader
from src.data.schemas import REQUIRED_FILE_NAMES, validate_audio_index, validate_json_list, validate_transactions_schema


def _write_dataset(root: Path, dataset_name: str, n_tx: int) -> Path:
    ds = root / dataset_name
    ds.mkdir(parents=True, exist_ok=True)

    tx_rows = [
        "transaction_id,sender_id,recipient_id,transaction_type,amount,location,payment_method,sender_iban,recipient_iban,balance_after,description,timestamp"
    ]
    for i in range(n_tx):
        tx_rows.append(
            f"tx_{i},SMTH-JONX-{i},DOEY-JANX-{i},transfer,{100+i},Metropolis,card,IT60X111{i},IT60Y222{i},{9000-i},invoice {i},2026-04-01T0{i%9}:00:00Z"
        )
    (ds / "transactions.csv").write_text("\n".join(tx_rows) + "\n", encoding="utf-8")

    users = [
        {
            "first_name": "John",
            "last_name": "Smith",
            "birth_year": 1987,
            "salary": 50000,
            "job": "Engineer",
            "iban": "IT60X1110",
            "residence": {"city": "Metropolis", "lat": 45.1, "lng": 9.1},
            "description": "regular user",
        },
        {
            "first_name": "Jane",
            "last_name": "Doe",
            "birth_year": 1989,
            "salary": 45000,
            "job": "Analyst",
            "iban": "IT60Y2220",
            "residence": {"city": "Metropolis", "lat": 45.2, "lng": 9.2},
            "description": "recipient user",
        },
    ]
    (ds / "users.json").write_text(json.dumps(users), encoding="utf-8")

    locations = [
        {"biotag": "SMTH-JONX-99", "timestamp": "2026-04-01T00:00:00Z", "lat": 45.11, "lng": 9.11, "city": "Metropolis"}
    ]
    (ds / "locations.json").write_text(json.dumps(locations), encoding="utf-8")

    sms = [
        {
            "sms": "From: John Smith\nTo: Jane Doe\nDate: 2026-04-01T00:10:00Z\nMessage: urgent verify your transfer"
        }
    ]
    (ds / "sms.json").write_text(json.dumps(sms), encoding="utf-8")

    mails = [
        {
            "mail": "From: notify@secure-pay.example\nTo: john@example.com\nSubject: Verify account\nDate: 2026-04-01T00:11:00Z\n\n<html><body>Click link now</body></html>"
        }
    ]
    (ds / "mails.json").write_text(json.dumps(mails), encoding="utf-8")

    audio = ds / "audio"
    audio.mkdir(exist_ok=True)
    (audio / "John Smith_2026-04-01T001200.mp3").write_bytes(b"ID3\x00\x00\x00\x00")

    zip_path = root / f"{dataset_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in ds.rglob("*"):
            zf.write(path, arcname=f"{dataset_name}/{path.relative_to(ds)}")
    return zip_path


def test_reference_and_target_dataset_zips_load(tmp_path: Path):
    reference_zip = _write_dataset(tmp_path, "Deus Ex - train", 8)
    target_zip = _write_dataset(tmp_path, "Deus Ex - validation", 9)

    paired = PairedDatasetLoader(str(reference_zip), str(target_zip))
    reference_paths, target_paths = paired.load_both()

    assert Path(reference_paths.dataset_root).exists()
    assert Path(target_paths.dataset_root).exists()
    assert reference_paths.audio_file_count > 0
    assert target_paths.audio_file_count > 0


def test_required_files_found_and_audio_index(tmp_path: Path):
    reference_zip = _write_dataset(tmp_path, "Deus Ex - train", 5)
    target_zip = _write_dataset(tmp_path, "Deus Ex - validation", 6)

    ref_loader = DatasetLoader(str(reference_zip))
    tgt_loader = DatasetLoader(str(target_zip))

    ref_root = Path(ref_loader.load_all().dataset_root)
    tgt_root = Path(tgt_loader.load_all().dataset_root)

    ref_files = {p.name for p in ref_root.iterdir() if p.is_file()}
    tgt_files = {p.name for p in tgt_root.iterdir() if p.is_file()}

    assert REQUIRED_FILE_NAMES.issubset(ref_files)
    assert REQUIRED_FILE_NAMES.issubset(tgt_files)

    ref_audio = ref_loader.load_audio_files()
    tgt_audio = tgt_loader.load_audio_files()
    validate_audio_index(ref_audio)
    validate_audio_index(tgt_audio)


def test_schemas_validate(tmp_path: Path):
    target_zip = _write_dataset(tmp_path, "Deus Ex - validation", 7)
    loader = DatasetLoader(str(target_zip))

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
