from .base import BaseRetriever, register_retriever
from ..digest_state import DigestState, arxiv_id
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
import calendar
from datetime import datetime, timedelta, timezone
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import re
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests
import feedparser

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
FEED_TIMEOUT = (10, 30)
API_URL = "https://export.arxiv.org/api/query"
API_PAGE_SIZE = 200
BACKFILL_WINDOW = timedelta(hours=72)
MISSED_RUN_GAP = timedelta(hours=36)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180

HTML_TAG_PATTERN = re.compile(
    r"</?(?:p|a|span|div|b|i|strong|em|br|hr|h[1-6]|ul|ol|li|sub|sup|table|tr|td|th)\b(?:\s+[^>]*)?/?>",
    flags=re.IGNORECASE,
)
ARXIV_HEADER_PATTERN = re.compile(
    r"^(?:arxiv:\s*\S+(?:\s+\[[^\]]*\])?)?\s*(?:announce\s+type:\s*[\w-]+)?\s*(?:abstract:\s*)?",
    flags=re.IGNORECASE,
)


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _clean_abstract(summary_raw: str) -> str:
    cleaned = HTML_TAG_PATTERN.sub(" ", summary_raw).strip()
    cleaned = ARXIV_HEADER_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"^abstract:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return " ".join(cleaned.split())


def _extract_authors(entry: Any) -> list[ArxivResult.Author]:
    author_str = getattr(entry, "author", "") or ""
    if not author_str and hasattr(entry, "authors") and entry.authors:
        author_str = ", ".join(
            a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a))
            for a in entry.authors
        )
    author_names = [a.strip() for a in author_str.split(",") if a.strip()]
    if not author_names:
        author_names = ["Unknown"]
    return [ArxivResult.Author(name=name) for name in author_names]


def _parse_entry_time(struct_time: Any) -> datetime:
    if struct_time:
        try:
            return datetime.fromtimestamp(calendar.timegm(struct_time), tz=timezone.utc)
        except Exception:
            pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _entry_to_arxiv_result(entry: Any) -> ArxivResult:
    raw_id = getattr(entry, "id", "") or ""
    paper_id = raw_id.removeprefix("oai:arXiv.org:")
    if "://" in paper_id:
        paper_id = paper_id.partition("/abs/")[2]
    title_raw = getattr(entry, "title", "") or ""
    title = " ".join(title_raw.split())

    authors = _extract_authors(entry)
    summary = _clean_abstract(getattr(entry, "summary", "") or "")

    link = getattr(entry, "link", "") or f"https://arxiv.org/abs/{paper_id}"
    link = link.replace("http://arxiv.org/", "https://arxiv.org/", 1)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    source_url = f"https://arxiv.org/src/{paper_id}"

    tags = getattr(entry, "tags", []) or []
    categories = [
        t.get("term") for t in tags
        if isinstance(t, dict) and t.get("term")
    ]
    # Prefer Atom's authoritative arxiv:primary_category field, with fallback to first category
    primary_category = ""
    arxiv_primary = getattr(entry, "arxiv_primary_category", None)
    if isinstance(arxiv_primary, dict):
        primary_category = arxiv_primary.get("term", "") or ""
    elif hasattr(arxiv_primary, "term"):
        primary_category = getattr(arxiv_primary, "term", "") or ""
    elif isinstance(arxiv_primary, str):
        primary_category = arxiv_primary

    if not primary_category and categories:
        primary_category = categories[0]

    published = _parse_entry_time(getattr(entry, "published_parsed", None))
    updated = _parse_entry_time(getattr(entry, "updated_parsed", None))
    comment = getattr(entry, "arxiv_comment", "") or ""
    journal_ref = getattr(entry, "arxiv_journal_reference", "") or getattr(entry, "arxiv_journal_ref", "") or ""
    doi = getattr(entry, "arxiv_doi", "") or ""

    links = [
        ArxivResult.Link(href=link, rel="alternate"),
        ArxivResult.Link(href=pdf_url, title="pdf", rel="related"),
        ArxivResult.Link(href=source_url, title="source", rel="related"),
    ]

    return ArxivResult(
        entry_id=link,
        updated=updated,
        published=published,
        title=title,
        authors=authors,
        summary=summary,
        comment=comment,
        journal_ref=journal_ref,
        doi=doi,
        primary_category=primary_category,
        categories=categories,
        links=links,
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if not self.retriever_config.category:
            raise ValueError("category must be specified for arxiv.")
        self.categories = set(self.retriever_config.category)
        self.include_cross_list = self.retriever_config.get("include_cross_list", False)
        self.defer_full_text = False
        self._raw_by_url: dict[str, ArxivResult] = {}
        self._retrieved_at = None
        self.state = None
        state_path = os.getenv("ARXIV_DIGEST_STATE")
        if state_path and not config.executor.debug:
            self.state = DigestState(Path(state_path), config.zotero.api_key, config.email.receiver)

    def _fetch_rss(self):
        query = "+".join(sorted(self.categories))
        url = f"https://rss.arxiv.org/atom/{query}"
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=FEED_TIMEOUT)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                title = feed.feed.get("title", "")
                if "Feed error for query" in title:
                    raise ValueError(f"Invalid arXiv categories: {query}")
                if not title or (feed.bozo and not feed.entries):
                    raise RuntimeError("Invalid arXiv RSS response")
                return feed
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == 4:
                    raise RuntimeError("Could not retrieve arXiv RSS after 5 attempts") from exc
                logger.warning(f"arXiv RSS attempt {attempt + 1} failed: {exc}")
                sleep(5)

    def _fetch_api(self, now: datetime) -> list[ArxivResult]:
        since = now - BACKFILL_WINDOW
        categories = " OR ".join(f"cat:{category}" for category in sorted(self.categories))
        date_range = f"[{since:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
        params = {
            "search_query": f"({categories}) AND submittedDate:{date_range}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": API_PAGE_SIZE,
            "start": 0,
        }

        papers = []
        with requests.Session() as session:
            while True:
                for attempt in range(3):
                    try:
                        response = session.get(API_URL, params=params, timeout=FEED_TIMEOUT)
                        response.raise_for_status()
                        feed = feedparser.parse(response.content)
                        total = int(feed.feed.get("opensearch_totalresults", -1))
                        valid_ids = all("/abs/" in entry.get("id", "") for entry in feed.entries)
                        if feed.bozo or not valid_ids or total < 0 or (not feed.entries and params["start"] < total):
                            raise RuntimeError("Invalid or incomplete arXiv API response")
                        break
                    except (requests.RequestException, RuntimeError, ValueError) as exc:
                        if attempt == 2:
                            raise RuntimeError("Could not retrieve an arXiv API page after 3 attempts") from exc
                        logger.warning(f"arXiv API attempt {attempt + 1} failed: {exc}")
                        sleep(3)

                for entry in feed.entries:
                    paper = _entry_to_arxiv_result(entry)
                    if since <= paper.published <= now and (
                        self.include_cross_list or paper.primary_category in self.categories
                    ):
                        papers.append(paper)
                params["start"] += len(feed.entries)
                if params["start"] >= total:
                    break
                sleep(3)

        logger.info(f"arXiv API 72-hour backfill (submission time): {len(papers)} eligible papers")
        return papers

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        now = datetime.now(timezone.utc)
        self._retrieved_at = None
        papers = {}
        feed_is_stale = False
        try:
            feed = self._fetch_rss()
        except RuntimeError as exc:
            logger.warning(f"{exc}; falling back to the arXiv API")
        else:
            updated = feed.feed.get("updated_parsed")
            feed_is_stale = bool(updated and now - _parse_entry_time(updated) >= MISSED_RUN_GAP)
            allowed = {"new", "cross"} if self.include_cross_list else {"new"}
            for entry in feed.entries:
                if entry.get("arxiv_announce_type", "new") in allowed:
                    paper = _entry_to_arxiv_result(entry)
                    papers.setdefault(arxiv_id(paper.entry_id), paper)
            logger.info(
                f"arXiv RSS announcement feed updated {feed.feed.get('updated', 'unknown')}: "
                f"{len(feed.entries)} entries, {len(papers)} eligible unique papers"
            )

        gap = self.state is not None and (
            self.state.last_retrieved_at is None
            or now - self.state.last_retrieved_at >= MISSED_RUN_GAP
        )
        if not papers or gap or (self.state is not None and feed_is_stale):
            try:
                for paper in self._fetch_api(now):
                    papers.setdefault(arxiv_id(paper.entry_id), paper)
            except RuntimeError as exc:
                if not papers:
                    raise RuntimeError("arXiv RSS yielded no eligible papers and API fallback failed") from exc
                logger.warning(f"{exc}; using RSS and keeping the recovery window open")
            else:
                self._retrieved_at = now
        else:
            self._retrieved_at = now

        remaining = [
            paper for paper_id, paper in papers.items()
            if self.state is None or not self.state.contains(paper_id)
        ]
        logger.info(f"arXiv delivery history excluded {len(papers) - len(remaining)} already sent papers")
        return remaining[:10] if self.config.executor.debug else remaining

    def record_delivery(self, delivered: list[Paper]) -> None:
        if self.state is not None:
            self.state.record(
                [arxiv_id(paper.url) for paper in delivered if paper.source == self.name],
                self._retrieved_at,
            )

    @staticmethod
    def _extract_full_text(raw_paper: ArxivResult) -> str | None:
        return (
            extract_text_from_tar(raw_paper)
            or extract_text_from_html(raw_paper)
            or extract_text_from_pdf(raw_paper)
        )

    def enrich_paper(self, paper: Paper) -> None:
        raw_paper = self._raw_by_url.get(paper.url)
        if raw_paper is not None:
            paper.full_text = self._extract_full_text(raw_paper)
            sleep(1)

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        if self.defer_full_text:
            self._raw_by_url[raw_paper.entry_id] = raw_paper
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[author.name for author in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None if self.defer_full_text else self._extract_full_text(raw_paper),
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
