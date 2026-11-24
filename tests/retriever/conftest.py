"""Retriever-specific fixtures."""

from pathlib import Path

import feedparser
import pytest
import requests


@pytest.fixture()
def mock_feedparser(monkeypatch):
    """Serve the arXiv RSS fixture without making network requests."""
    body = Path(__file__).with_name("arxiv_rss_example.xml").read_bytes()
    original_get = requests.get

    def get(url, **kwargs):
        if "rss.arxiv.org" in url:
            response = requests.Response()
            response.status_code = 200
            response._content = body
            return response
        return original_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", get)
    return feedparser.parse(body)


@pytest.fixture()
def mock_biorxiv_api(monkeypatch):
    """Patch requests.get to return the canned bioRxiv API response."""
    import requests
    from types import SimpleNamespace

    from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE

    original_get = requests.get

    def _patched(url, **kwargs):
        if "api.biorxiv.org" in url:
            resp = SimpleNamespace()
            resp.status_code = 200
            resp.json = lambda: SAMPLE_BIORXIV_API_RESPONSE
            resp.raise_for_status = lambda: None
            return resp
        return original_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", _patched)
