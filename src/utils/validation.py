from __future__ import annotations


def assert_ascii_lines(lines: list[str]) -> None:
    if not lines:
        raise ValueError("Submission cannot be empty")
    for line in lines:
        if not line:
            raise ValueError("Submission contains an empty line")
        line.encode("ascii")


def assert_not_all(flagged_count: int, total_count: int) -> None:
    if flagged_count <= 0:
        raise ValueError("Submission cannot contain zero transactions")
    if flagged_count >= total_count:
        raise ValueError("Submission cannot contain all transactions")
