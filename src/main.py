from __future__ import annotations

import argparse
import sys

from src.cli import run_inspect_pair, run_predict_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reply Mirror Fraud Detection Agent Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect-pair", help="Inspect reference and target datasets")
    inspect_cmd.add_argument("--reference", required=True, help="Path to reference/train dataset zip or folder")
    inspect_cmd.add_argument("--input", required=True, help="Path to target/validation dataset zip or folder")
    inspect_cmd.add_argument("--config", default=None, help="Optional YAML config path")

    predict_cmd = subparsers.add_parser("predict-pair", help="Generate fraud submission using reference+target")
    predict_cmd.add_argument("--reference", required=True, help="Path to reference/train dataset zip or folder")
    predict_cmd.add_argument("--input", required=True, help="Path to target/validation dataset zip or folder")
    predict_cmd.add_argument("--output", required=True, help="Output submission TXT path")
    predict_cmd.add_argument("--config", default=None, help="Optional YAML config path")
    predict_cmd.add_argument("--verbose", action="store_true", help="Verbose logging")
    predict_cmd.add_argument("--no-llm", action="store_true", help="Disable LLM communication analysis")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-pair":
        run_inspect_pair(reference_path=args.reference, input_path=args.input, config_path=args.config)
        return 0

    if args.command == "predict-pair":
        run_predict_pair(
            reference_path=args.reference,
            input_path=args.input,
            output_path=args.output,
            no_llm=bool(args.no_llm),
            verbose=bool(args.verbose),
            config_path=args.config,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
