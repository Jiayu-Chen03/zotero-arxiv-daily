"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever_uses_rss_metadata(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]

    def fail_if_called(_paper):
        raise AssertionError("full text must not be fetched before reranking")

    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", fail_if_called)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", fail_if_called)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", fail_if_called)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].abstract == "We propose a neural architecture search method for efficient transformers."
    assert papers[0].url == "https://arxiv.org/abs/2508.14001v1"
    assert papers[0].full_text is None


def test_arxiv_retriever_enriches_only_when_requested(config, monkeypatch):
    calls = []
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: calls.append(paper.paper_id) or "full text")

    paper = SimpleNamespace(
        source="arxiv",
        title="Selected paper",
        authors=["Test Author"],
        abstract="Abstract",
        url="https://arxiv.org/abs/2608.00001v1",
        pdf_url="https://arxiv.org/pdf/2608.00001v1",
        full_text=None,
    )
    ArxivRetriever(config).enrich_paper(paper)

    assert calls == ["2608.00001v1"]
    assert paper.full_text == "full text"


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
