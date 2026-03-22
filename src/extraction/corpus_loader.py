"""
Corpus Loader — PMC XML parsing and iteration for FRLM.

Provides:
    - PMCCorpusLoader: Download PMC OA subset papers and parse XML to structured dicts
    - download_pmc_oa_subset: Query PMC via E-utilities API and download matching papers
    - parse_pmc_xml: Parse a single PMC XML file into title, abstract, body sections, etc.
    - iterate_corpus: Generator over parsed papers in a directory

Handles XML parsing errors gracefully, logs and skips malformed files.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from config.config import get_secret
from xml.etree.ElementTree import Element

logger = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================


@dataclass
class ParsedPaper:
    """Structured representation of a parsed PMC paper.

    Attributes
    ----------
    pmcid : str
        PMC identifier (e.g., "PMC1234567").
    pmid : str
        PubMed identifier (e.g., "12345678"). May be empty.
    doi : str
        Digital Object Identifier. May be empty.
    title : str
        Paper title.
    abstract : str
        Abstract text (all paragraphs joined).
    body_sections : Dict[str, str]
        Section name → section text mapping.
    full_text : str
        All body text concatenated.
    references : List[Dict[str, str]]
        List of reference dicts with keys: id, title, authors, journal, year.
    publication_date : Optional[date]
        Date of publication, if parseable.
    journal : str
        Journal name.
    authors : List[str]
        List of author names.
    keywords : List[str]
        Paper keywords.
    source_file : str
        Path to the source XML file.
    metadata : Dict[str, Any]
        Additional metadata.
    """

    pmcid: str = ""
    pmid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    body_sections: Dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    references: List[Dict[str, str]] = field(default_factory=list)
    publication_date: Optional[date] = None
    journal: str = ""
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_text(self) -> str:
        """Return all extractable text: title + abstract + full text."""
        parts = [self.title, self.abstract, self.full_text]
        return "\n\n".join(p for p in parts if p)

    @property
    def word_count(self) -> int:
        """Approximate word count of all text."""
        return len(self.all_text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "pmcid": self.pmcid,
            "pmid": self.pmid,
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "body_sections": self.body_sections,
            "full_text": self.full_text,
            "references": self.references,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "journal": self.journal,
            "authors": self.authors,
            "keywords": self.keywords,
            "source_file": self.source_file,
            "metadata": self.metadata,
        }


# ===========================================================================
# NCBI E-utilities API
# ===========================================================================

NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_OA_FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"

# Rate limiting: NCBI allows 3 requests/second without API key, 10 with API key
_NCBI_REQUEST_DELAY = 0.35  # Default delay between requests


def _ncbi_request(
    url: str,
    params: Dict[str, str],
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Make a rate-limited request to NCBI E-utilities.

    Parameters
    ----------
    url : str
        E-utilities endpoint URL.
    params : dict
        Query parameters.
    api_key : str, optional
        NCBI API key for higher rate limits.
    max_retries : int
        Maximum retry attempts on failure.
    retry_delay : float
        Initial delay between retries (exponential backoff).

    Returns
    -------
    str
        Response text.

    Raises
    ------
    urllib.error.URLError
        If all retries fail.
    """
    if api_key:
        params["api_key"] = api_key

    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"

    for attempt in range(max_retries):
        try:
            time.sleep(_NCBI_REQUEST_DELAY)
            with urllib.request.urlopen(full_url, timeout=60) as response:
                return response.read().decode("utf-8")
        except urllib.error.URLError as e:
            delay = retry_delay * (2**attempt)
            logger.warning(
                "NCBI request failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            time.sleep(delay)

    raise urllib.error.URLError(f"Failed to fetch {url} after {max_retries} attempts")


def _parse_esearch_response(xml_text: str) -> Tuple[int, List[str]]:
    """Parse ESearch XML response to extract count and ID list.

    Returns
    -------
    Tuple[int, List[str]]
        (total_count, list_of_ids)
    """
    root = ET.fromstring(xml_text)

    count_elem = root.find(".//Count")
    total_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

    id_list = []
    for id_elem in root.findall(".//IdList/Id"):
        if id_elem.text:
            id_list.append(id_elem.text)

    return total_count, id_list


# ===========================================================================
# XML Parsing
# ===========================================================================


def _get_text_recursive(element: Optional[Element]) -> str:
    """Recursively extract all text from an XML element and its children."""
    if element is None:
        return ""

    texts = []
    if element.text:
        texts.append(element.text)

    for child in element:
        texts.append(_get_text_recursive(child))
        if child.tail:
            texts.append(child.tail)

    return " ".join(texts).strip()


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove artifacts."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove common XML artifacts
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


def _parse_date(date_elem: Optional[Element]) -> Optional[date]:
    """Parse a JATS date element into a Python date."""
    if date_elem is None:
        return None

    year_elem = date_elem.find("year")
    month_elem = date_elem.find("month")
    day_elem = date_elem.find("day")

    try:
        year = int(year_elem.text) if year_elem is not None and year_elem.text else None
        if year is None:
            return None

        month = int(month_elem.text) if month_elem is not None and month_elem.text else 1
        day = int(day_elem.text) if day_elem is not None and day_elem.text else 1

        # Clamp month and day to valid ranges
        month = max(1, min(12, month))
        day = max(1, min(31, day))

        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def _extract_article_ids(article_meta: Element) -> Dict[str, str]:
    """Extract article identifiers (PMID, PMCID, DOI) from article-meta."""
    ids = {"pmid": "", "pmcid": "", "doi": ""}

    for article_id in article_meta.findall(".//article-id"):
        id_type = article_id.get("pub-id-type", "")
        id_value = article_id.text or ""

        if id_type == "pmid":
            ids["pmid"] = id_value
        elif id_type == "pmc":
            ids["pmcid"] = f"PMC{id_value}" if not id_value.startswith("PMC") else id_value
        elif id_type == "doi":
            ids["doi"] = id_value

    return ids


def _extract_abstract(article_meta: Element) -> str:
    """Extract abstract text from article-meta."""
    abstract_parts = []

    for abstract in article_meta.findall(".//abstract"):
        # Handle structured abstracts
        for sec in abstract.findall(".//sec"):
            title = sec.find("title")
            if title is not None and title.text:
                abstract_parts.append(f"{title.text}:")
            for p in sec.findall(".//p"):
                abstract_parts.append(_get_text_recursive(p))

        # Handle simple abstracts (just paragraphs)
        for p in abstract.findall("p"):
            abstract_parts.append(_get_text_recursive(p))

        # Handle abstract without explicit structure
        if not abstract_parts:
            abstract_parts.append(_get_text_recursive(abstract))

    return _clean_text(" ".join(abstract_parts))


def _extract_body_sections(body: Optional[Element]) -> Tuple[Dict[str, str], str]:
    """Extract body sections and full text from the body element.

    Returns
    -------
    Tuple[Dict[str, str], str]
        (section_name_to_text_dict, full_concatenated_text)
    """
    if body is None:
        return {}, ""

    sections: Dict[str, str] = {}
    all_text_parts: List[str] = []

    # Try to extract structured sections
    for sec in body.findall(".//sec"):
        title_elem = sec.find("title")
        section_name = (
            _clean_text(title_elem.text)
            if title_elem is not None and title_elem.text
            else "unnamed"
        )
        section_name = section_name.lower()

        # Extract all paragraphs in this section (excluding nested sections)
        paragraphs = []
        for p in sec.findall("p"):
            paragraphs.append(_get_text_recursive(p))

        section_text = _clean_text(" ".join(paragraphs))

        if section_text:
            # Handle duplicate section names
            if section_name in sections:
                sections[section_name] += " " + section_text
            else:
                sections[section_name] = section_text

            all_text_parts.append(section_text)

    # If no sections found, try to extract all paragraphs directly
    if not sections:
        for p in body.findall(".//p"):
            text = _get_text_recursive(p)
            if text:
                all_text_parts.append(_clean_text(text))

    full_text = _clean_text(" ".join(all_text_parts))

    return sections, full_text


def _extract_references(back: Optional[Element]) -> List[Dict[str, str]]:
    """Extract references from the back matter."""
    if back is None:
        return []

    references = []

    for ref in back.findall(".//ref"):
        ref_data: Dict[str, str] = {
            "id": ref.get("id", ""),
            "title": "",
            "authors": "",
            "journal": "",
            "year": "",
        }

        # Try different citation formats (element-citation, mixed-citation, etc.)
        citation = (
            ref.find(".//element-citation")
            or ref.find(".//mixed-citation")
            or ref.find(".//citation")
        )

        if citation is not None:
            # Article title
            article_title = citation.find(".//article-title")
            if article_title is not None:
                ref_data["title"] = _get_text_recursive(article_title)

            # Authors
            author_names = []
            for name in citation.findall(".//name"):
                surname = name.find("surname")
                given = name.find("given-names")
                if surname is not None and surname.text:
                    author_str = surname.text
                    if given is not None and given.text:
                        author_str += f" {given.text}"
                    author_names.append(author_str)
            ref_data["authors"] = ", ".join(author_names)

            # Journal/Source
            source = citation.find(".//source")
            if source is not None:
                ref_data["journal"] = _get_text_recursive(source)

            # Year
            year = citation.find(".//year")
            if year is not None and year.text:
                ref_data["year"] = year.text

        references.append(ref_data)

    return references


def _extract_authors(article_meta: Element) -> List[str]:
    """Extract author names from article-meta."""
    authors = []

    for contrib in article_meta.findall(".//contrib[@contrib-type='author']"):
        name = contrib.find("name")
        if name is not None:
            surname = name.find("surname")
            given = name.find("given-names")
            if surname is not None and surname.text:
                author_str = surname.text
                if given is not None and given.text:
                    author_str = f"{given.text} {author_str}"
                authors.append(author_str)

    return authors


def _extract_keywords(article_meta: Element) -> List[str]:
    """Extract keywords from article-meta."""
    keywords = []

    for kwd_group in article_meta.findall(".//kwd-group"):
        for kwd in kwd_group.findall("kwd"):
            if kwd.text:
                keywords.append(_clean_text(kwd.text))

    return keywords


def parse_pmc_xml(filepath: Path) -> ParsedPaper:
    """Parse a PMC XML file into a structured ParsedPaper object.

    Parameters
    ----------
    filepath : Path
        Path to the PMC XML file (supports .xml, .xml.gz, .nxml).

    Returns
    -------
    ParsedPaper
        Structured paper data.

    Raises
    ------
    ET.ParseError
        If XML parsing fails.
    FileNotFoundError
        If the file doesn't exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XML file not found: {filepath}")

    paper = ParsedPaper(source_file=str(filepath))

    # Handle gzipped files
    try:
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(filepath)
    except ET.ParseError as e:
        logger.error("XML parse error in %s: %s", filepath, e)
        raise

    root = tree.getroot()

    # Find the article element (handle both JATS and older PMC formats)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        logger.warning("No article element found in %s", filepath)
        return paper

    # Front matter (metadata)
    front = article.find("front")
    if front is not None:
        journal_meta = front.find("journal-meta")
        article_meta = front.find("article-meta")

        # Journal name
        if journal_meta is not None:
            journal_title = (
                journal_meta.find(".//journal-title")
                or journal_meta.find(".//abbrev-journal-title")
            )
            if journal_title is not None and journal_title.text:
                paper.journal = _clean_text(journal_title.text)

        if article_meta is not None:
            # Article IDs
            ids = _extract_article_ids(article_meta)
            paper.pmid = ids["pmid"]
            paper.pmcid = ids["pmcid"]
            paper.doi = ids["doi"]

            # Title
            title_group = article_meta.find("title-group")
            if title_group is not None:
                article_title = title_group.find("article-title")
                if article_title is not None:
                    paper.title = _clean_text(_get_text_recursive(article_title))

            # Abstract
            paper.abstract = _extract_abstract(article_meta)

            # Publication date
            pub_date = (
                article_meta.find(".//pub-date[@pub-type='epub']")
                or article_meta.find(".//pub-date[@pub-type='ppub']")
                or article_meta.find(".//pub-date[@date-type='pub']")
                or article_meta.find(".//pub-date")
            )
            paper.publication_date = _parse_date(pub_date)

            # Authors
            paper.authors = _extract_authors(article_meta)

            # Keywords
            paper.keywords = _extract_keywords(article_meta)

    # Body (main text)
    body = article.find("body")
    paper.body_sections, paper.full_text = _extract_body_sections(body)

    # Back matter (references)
    back = article.find("back")
    paper.references = _extract_references(back)

    # Extract PMCID from filename if not found in metadata
    if not paper.pmcid:
        match = re.search(r"PMC\d+", filepath.name, re.IGNORECASE)
        if match:
            paper.pmcid = match.group(0).upper()

    return paper


# ===========================================================================
# Corpus Loader Class
# ===========================================================================


class PMCCorpusLoader:
    """Load and parse PMC Open Access papers.

    Parameters
    ----------
    corpus_dir : Path or str
        Directory containing or to contain PMC XML files.
    cache_dir : Path or str
        Directory for caching downloaded metadata.
    api_key : str, optional
        NCBI API key for higher rate limits.
    max_retries : int
        Maximum retries for network requests.
    retry_delay : float
        Initial retry delay (seconds).
    """

    def __init__(
        self,
        corpus_dir: Path | str,
        cache_dir: Optional[Path | str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.corpus_dir / ".cache"
        self.api_key = api_key or get_secret("ncbi.api_key")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Statistics
        self._stats: Dict[str, int] = {
            "downloaded": 0,
            "parsed": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Ensure directories exist
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PMCCorpusLoader initialized: corpus_dir=%s, cache_dir=%s",
            self.corpus_dir,
            self.cache_dir,
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return download/parse statistics."""
        return self._stats.copy()

    def download_pmc_oa_subset(
        self,
        query: str,
        max_papers: int = 1000,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        filter_open_access: bool = True,
    ) -> List[str]:
        """Download PMC papers matching a query via E-utilities.

        Parameters
        ----------
        query : str
            PubMed/PMC search query (e.g., "cancer AND drug therapy").
        max_papers : int
            Maximum number of papers to download.
        min_year : int, optional
            Minimum publication year filter.
        max_year : int, optional
            Maximum publication year filter.
        filter_open_access : bool
            If True, filter to Open Access subset only.

        Returns
        -------
        List[str]
            List of downloaded PMCID identifiers.
        """
        logger.info(
            "Searching PMC: query='%s', max_papers=%d, years=[%s-%s]",
            query,
            max_papers,
            min_year or "any",
            max_year or "any",
        )

        # Build the search query
        search_terms = [query]
        if filter_open_access:
            search_terms.append("open access[filter]")
        if min_year:
            search_terms.append(f"{min_year}[PDAT]:{max_year or '3000'}[PDAT]")

        full_query = " AND ".join(search_terms)

        # Step 1: ESearch to get PMCIDs
        pmcids = self._esearch_pmcids(full_query, max_papers)
        logger.info("Found %d PMCIDs matching query", len(pmcids))

        if not pmcids:
            return []

        # Step 2: Download each paper
        downloaded = []
        for idx, pmcid in enumerate(pmcids, start=1):
            success = self._download_paper_by_pmcid(pmcid)
            if success:
                downloaded.append(pmcid)
                self._stats["downloaded"] += 1
            else:
                self._stats["failed"] += 1

            if idx % 50 == 0:
                logger.info("Download progress: %d/%d", idx, len(pmcids))

        logger.info(
            "Download complete: %d successful, %d failed",
            len(downloaded),
            len(pmcids) - len(downloaded),
        )
        return downloaded

    def _esearch_pmcids(self, query: str, max_results: int) -> List[str]:
        """Run ESearch and return list of PMCIDs."""
        pmcids = []
        batch_size = min(10000, max_results)  # NCBI limit
        retstart = 0

        while len(pmcids) < max_results:
            params = {
                "db": "pmc",
                "term": query,
                "retmax": str(min(batch_size, max_results - len(pmcids))),
                "retstart": str(retstart),
                "retmode": "xml",
                "usehistory": "n",
            }

            try:
                response = _ncbi_request(
                    NCBI_ESEARCH_URL,
                    params,
                    api_key=self.api_key,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                )
                total_count, batch_ids = _parse_esearch_response(response)

                if not batch_ids:
                    break

                pmcids.extend([f"PMC{pid}" for pid in batch_ids])
                retstart += len(batch_ids)

                if retstart >= total_count:
                    break

            except Exception as e:
                logger.error("ESearch failed: %s", e)
                break

        return pmcids[:max_results]

    def _download_paper_by_pmcid(self, pmcid: str) -> bool:
        """Download a single paper by PMCID.

        Tries multiple sources:
        1. EFetch for full XML
        2. OA bulk download
        """
        # Normalize PMCID
        pmcid_num = pmcid.replace("PMC", "")
        output_path = self.corpus_dir / f"{pmcid}.xml"

        if output_path.exists():
            logger.debug("Already exists: %s", pmcid)
            self._stats["skipped"] += 1
            return True

        tmp_path = output_path.with_suffix(".xml.tmp")

        # Try EFetch first
        try:
            params = {
                "db": "pmc",
                "id": pmcid_num,
                "rettype": "xml",
                "retmode": "xml",
            }
            response = _ncbi_request(
                NCBI_EFETCH_URL,
                params,
                api_key=self.api_key,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
            )

            # Validate that we got actual XML content
            if "<article" in response or "<pmc-articleset" in response:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(response)
                tmp_path.replace(output_path)
                logger.debug("Downloaded via EFetch: %s", pmcid)
                return True
            else:
                logger.warning("EFetch returned non-XML content for %s", pmcid)

        except Exception as e:
            logger.warning("EFetch failed for %s: %s", pmcid, e)

        # Clean up temp file if download failed
        if tmp_path.exists():
            tmp_path.unlink()

        return False

    def parse_pmc_xml(self, filepath: Path) -> Optional[ParsedPaper]:
        """Parse a single PMC XML file with error handling.

        Parameters
        ----------
        filepath : Path
            Path to the XML file.

        Returns
        -------
        ParsedPaper or None
            Parsed paper data, or None if parsing failed.
        """
        try:
            paper = parse_pmc_xml(filepath)
            self._stats["parsed"] += 1
            return paper
        except ET.ParseError as e:
            logger.warning("XML parse error in %s: %s", filepath, e)
            self._stats["failed"] += 1
            return None
        except Exception as e:
            logger.warning("Unexpected error parsing %s: %s", filepath, e)
            self._stats["failed"] += 1
            return None

    def iterate_corpus(
        self,
        directory: Optional[Path] = None,
        pattern: str = "*.xml",
        max_papers: Optional[int] = None,
        skip_errors: bool = True,
    ) -> Generator[ParsedPaper, None, None]:
        """Iterate over parsed papers in a directory.

        Parameters
        ----------
        directory : Path, optional
            Directory to scan. Defaults to self.corpus_dir.
        pattern : str
            Glob pattern for files to parse.
        max_papers : int, optional
            Maximum number of papers to yield.
        skip_errors : bool
            If True, skip files that fail to parse.

        Yields
        ------
        ParsedPaper
            Parsed paper data.
        """
        directory = directory or self.corpus_dir
        xml_files = sorted(directory.glob(pattern))

        logger.info(
            "Iterating corpus: directory=%s, pattern=%s, found=%d files",
            directory,
            pattern,
            len(xml_files),
        )

        count = 0
        for filepath in xml_files:
            if max_papers is not None and count >= max_papers:
                break

            paper = self.parse_pmc_xml(filepath)

            if paper is not None:
                yield paper
                count += 1
            elif not skip_errors:
                raise RuntimeError(f"Failed to parse {filepath}")

            if count % 100 == 0:
                logger.info("Parsed %d papers", count)

        logger.info(
            "Corpus iteration complete: parsed=%d, failed=%d",
            self._stats["parsed"],
            self._stats["failed"],
        )

    def get_text_chunks(
        self,
        paper: ParsedPaper,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        sections: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Split a paper into overlapping text chunks.

        Parameters
        ----------
        paper : ParsedPaper
            Parsed paper data.
        chunk_size : int
            Target size of each chunk in words.
        chunk_overlap : int
            Word overlap between consecutive chunks.
        sections : List[str], optional
            Sections to include. If None, uses all text.

        Returns
        -------
        List[Dict[str, Any]]
            List of chunk dicts with keys: text, pmcid, section, chunk_idx.
        """
        chunks = []

        # Gather text from specified sections or all text
        if sections:
            section_texts = []
            for sec_name in sections:
                sec_name_lower = sec_name.lower()
                # Check body_sections
                for key, text in paper.body_sections.items():
                    if sec_name_lower in key.lower():
                        section_texts.append((sec_name, text))
                        break
                # Special handling for abstract
                if sec_name_lower == "abstract" and paper.abstract:
                    section_texts.append(("abstract", paper.abstract))
            texts_to_chunk = section_texts
        else:
            texts_to_chunk = [("full", paper.all_text)]

        for section_name, text in texts_to_chunk:
            words = text.split()
            if not words:
                continue

            start = 0
            chunk_idx = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])

                chunks.append({
                    "text": chunk_text,
                    "pmcid": paper.pmcid,
                    "section": section_name,
                    "chunk_idx": chunk_idx,
                    "start_word": start,
                    "end_word": end,
                })

                chunk_idx += 1
                start = end - chunk_overlap if end < len(words) else end

        return chunks


# ===========================================================================
# Convenience functions
# ===========================================================================


def load_corpus_from_config(config: Any) -> PMCCorpusLoader:
    """Create a PMCCorpusLoader from an FRLMConfig.

    Parameters
    ----------
    config : FRLMConfig
        FRLM configuration object.

    Returns
    -------
    PMCCorpusLoader
        Configured corpus loader.
    """
    from config.config import FRLMConfig

    if not isinstance(config, FRLMConfig):
        raise TypeError(f"Expected FRLMConfig, got {type(config)}")

    corpus_dir = config.paths.resolve("corpus_dir")
    cache_dir = config.paths.resolve("cache_dir")

    return PMCCorpusLoader(
        corpus_dir=corpus_dir,
        cache_dir=cache_dir,
        api_key=get_secret("ncbi.api_key"),
        max_retries=3,
        retry_delay=2.0,
    )


# Alias for backwards compatibility with __init__.py
CorpusLoader = PMCCorpusLoader

