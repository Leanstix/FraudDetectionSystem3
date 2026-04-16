from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.config import Settings
from src.pipeline.orchestrator import FraudPipeline


def _write_dataset(root: Path, dataset_name: str, n_tx: int) -> Path:
    ds = root / dataset_name
    ds.mkdir(parents=True, exist_ok=True)

    tx_rows = [
        "transaction_id,sender_id,recipient_id,transaction_type,amount,location,payment_method,sender_iban,recipient_iban,balance_after,description,timestamp"
    ]
    for i in range(n_tx):
        tx_rows.append(
            f"tx_{i},SMTH-JONX-{i},DOEY-JANX-{i},transfer,{120+i},Metropolis,card,IT60X111{i},IT60Y222{i},{8800-i},urgent invoice {i},2026-04-01T0{i%9}:1{i%5}:00Z"
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


def test_pipeline_end_to_end_creates_outputs_and_is_deterministic(tmp_path: Path):
    reference_zip = _write_dataset(tmp_path, "Deus Ex - train", 15)
    validation_zip = _write_dataset(tmp_path, "Deus Ex - validation", 16)

    output_path_1 = tmp_path / "deus_ex_validation_submission_1.txt"
    output_path_2 = tmp_path / "deus_ex_validation_submission_2.txt"

    settings = Settings.from_env_and_file()

    pipeline_1 = FraudPipeline(
        settings=settings,
        reference_path=str(reference_zip),
        input_path=str(validation_zip),
        output_path=str(output_path_1),
        dataset_name="deus_ex",
        no_llm=True,
        verbose=False,
    )
    artifacts_1 = pipeline_1.run_pair()

    assert output_path_1.exists()
    assert Path(artifacts_1.diagnostics_path).exists()
    assert artifacts_1.submission.flagged_count > 0
    assert artifacts_1.submission.flagged_count < artifacts_1.submission.total_transactions

    pipeline_2 = FraudPipeline(
        settings=settings,
        reference_path=str(reference_zip),
        input_path=str(validation_zip),
        output_path=str(output_path_2),
        dataset_name="deus_ex",
        no_llm=True,
        verbose=False,
    )
    artifacts_2 = pipeline_2.run_pair()

    assert output_path_2.exists()
    assert Path(artifacts_2.diagnostics_path).exists()

    text1 = output_path_1.read_text(encoding="ascii")
    text2 = output_path_2.read_text(encoding="ascii")
    assert text1 == text2
