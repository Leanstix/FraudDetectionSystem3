from __future__ import annotations

import re

from bs4 import BeautifulSoup


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(html_or_text: str) -> str:
    if not html_or_text:
        return ""
    if "<html" in html_or_text.lower() or "<body" in html_or_text.lower():
        soup = BeautifulSoup(html_or_text, "lxml")
        text = soup.get_text(" ", strip=True)
    else:
        text = html_or_text
    return normalize_whitespace(text)


def parse_mail_headers(raw_mail: str) -> dict[str, str]:
    header_blob = raw_mail.split("\n\n", 1)[0]
    parsed: dict[str, str] = {}
    for line in header_blob.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        parsed[k.strip().lower()] = v.strip()
    return parsed


def find_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s\)\]\>\"']+", text or "", flags=re.IGNORECASE)
