"""Unit tests for the parsers in server/importer.py (pure functions, no network)."""

import json

from importer import (
    _chunk_list,
    _parse_generic_csv,
    _parse_generic_json,
    _parse_linkedin_csv,
    _parse_plain_text,
    _parse_twitter_archive,
)


# ── Twitter archive ──────────────────────────────────────────────────────────

def test_twitter_archive_strips_js_assignment_and_retweets():
    payload = [
        {"tweet": {"full_text": "Shipped the new search endpoint today"}},
        {"tweet": {"full_text": "RT @someone: reposted content"}},
        {"tweet": {"text": "Short form text field works too"}},
    ]
    data = "window.YTD.tweet.part0 = " + json.dumps(payload) + ";"
    texts = _parse_twitter_archive(data)
    assert texts == [
        "Shipped the new search endpoint today",
        "Short form text field works too",
    ]


def test_twitter_archive_accepts_plain_json_array():
    data = json.dumps([{"tweet": {"full_text": "Plain array entry"}}])
    assert _parse_twitter_archive(data) == ["Plain array entry"]


# ── LinkedIn CSV ─────────────────────────────────────────────────────────────

def test_linkedin_csv_builds_sentences():
    data = (
        "First Name,Last Name,Company,Position,Connected On\n"
        "Dana,Hassan,Acme Logistics,Operations Manager,01 Mar 2025\n"
    )
    entries = _parse_linkedin_csv(data)
    assert len(entries) == 1
    assert "Connected with Dana Hassan" in entries[0]
    assert "Operations Manager at Acme Logistics" in entries[0]
    assert "01 Mar 2025" in entries[0]


def test_linkedin_csv_company_without_position():
    data = "First Name,Last Name,Company,Position\nSam,Lee,Northwind,\n"
    entries = _parse_linkedin_csv(data)
    assert entries == ["Connected with Sam Lee who works at Northwind"]


# ── Generic JSON / CSV / plain text ──────────────────────────────────────────

def test_generic_json_collects_long_strings_recursively():
    data = json.dumps({
        "note": "This string is long enough to keep",
        "short": "tiny",
        "nested": {"items": ["Another sufficiently long string here"]},
    })
    texts = _parse_generic_json(data)
    assert "This string is long enough to keep" in texts
    assert "Another sufficiently long string here" in texts
    assert "tiny" not in texts


def test_generic_csv_joins_rows():
    data = "name,city\nNoor,Cairo\n"
    assert _parse_generic_csv(data) == ["name: Noor, city: Cairo"]


def test_plain_text_splits_paragraphs():
    data = "First paragraph about the project.\n\nSecond paragraph with more detail."
    assert len(_parse_plain_text(data)) == 2


def test_plain_text_falls_back_to_lines_over_20_chars():
    data = "short line\nThis line is comfortably longer than twenty characters\nok"
    parts = _parse_plain_text(data)
    assert parts == ["This line is comfortably longer than twenty characters"]


# ── Batching ─────────────────────────────────────────────────────────────────

def test_chunk_list_sizes():
    chunks = list(_chunk_list(list(range(10)), 4))
    assert [len(c) for c in chunks] == [4, 4, 2]
