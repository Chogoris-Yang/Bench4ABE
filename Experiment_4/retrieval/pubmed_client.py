"""
Experiment_4/retrieval/pubmed_client.py — PubMed E-utilities client.
Fetches paper abstracts in real-time (no pre-built DB needed).
"""

import time
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

from config_rag import PUBMED_MAX_RESULTS, PUBMED_TOP_K, PUBMED_EMAIL


class PubMedClient:
    """Lightweight PubMed client using NCBI E-utilities (REST API)."""

    BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, email: str = PUBMED_EMAIL):
        self.email = email
        self._cache: Dict[str, List[Dict]] = {}  # query → results cache

    def search(self, query: str, max_results: int = PUBMED_MAX_RESULTS) -> List[Dict]:
        """
        Search PubMed for papers relevant to a research question.
        Returns list of {pmid, title, abstract, year, journal}.
        """
        # Check cache
        cache_key = f"{query}::{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Step 1: esearch — get PMIDs
        pmids = self._esearch(query, max_results)
        if not pmids:
            print(f"    [PubMed] No results for query")
            return []

        # Step 2: efetch — get abstracts
        papers = self._efetch(pmids)

        # Cache and return
        self._cache[cache_key] = papers
        return papers

    def search_for_task(self, task_description: str) -> List[Dict]:
        """
        Build a PubMed query from a research task description.
        Extracts key biological terms and methods.
        """
        # Extract meaningful keywords
        query = self._build_query(task_description)
        print(f"    [PubMed] Query: {query[:200]}")
        return self.search(query)

    def _build_query(self, task: str) -> str:
        """Extract a PubMed-optimized query from a task description."""
        # Key biological entities and methods to search for
        bio_terms = [
            # Sequence / genomics
            "genomic", "sequence", "enhancer", "promoter", "chromatin",
            "Hi-C", "scRNA-seq", "scATAC-seq", "single-cell", "transcriptom",
            # Protein / structure
            "protein", "structure", "post-translational", "phosphorylation",
            "glycosylation", "ubiquitin", "AlphaFold",
            # Drug / interaction
            "drug-target", "binding", "pocket", "ligand", "pharmacophore",
            # Metagenomics
            "metagenom", "strain", "binning", "ANI", "microbiome",
            # Splicing / regulation
            "splicing", "RBP", "RNA-binding", "regulatory",
            # Variant / genetics
            "variant", "non-coding", "GWAS", "pathogenic", "eQTL",
            # Spatial / imputation
            "spatial transcriptom", "MERFISH", "Visium", "imputation",
            # Network
            "gene regulatory network", "GRN",
            # ML methods
            "deep learning", "transformer", "state space model",
            "graph neural network", "contrastive learning", "diffusion model",
        ]

        # Find matching terms
        task_lower = task.lower()
        matched = [t for t in bio_terms if t.lower() in task_lower]

        if matched:
            query = " AND ".join(f'"{t}"' if " " in t else t for t in matched[:4])
        else:
            # Fallback: use first 5 content words
            words = [w for w in re.findall(r'[a-zA-Z]{4,}', task) if w.lower() not in
                     ('that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their',
                      'which', 'what', 'when', 'about', 'your', 'will', 'also', 'some', 'more',
                      'than', 'into', 'such', 'other', 'these')]
            words = list(dict.fromkeys(words))[:5]  # unique, first 5
            query = " AND ".join(words)

        # Add methods filter
        query += ' AND ("method"[Title/Abstract] OR "computational"[Title/Abstract] OR "deep learning"[Title/Abstract] OR "bioinformatics"[Title/Abstract])'

        return query[:400]  # E-utilities limit

    def _esearch(self, query: str, max_results: int) -> List[str]:
        """E-utilities esearch: query → PMID list."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "email": self.email,
        }
        url = self.BASE_SEARCH + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pmids = data.get("esearchresult", {}).get("idlist", [])
            print(f"    [PubMed] Found {len(pmids)} PMIDs")
            return pmids
        except Exception as e:
            print(f"    [PubMed] esearch error: {e}")
            return []

    def _efetch(self, pmids: List[str]) -> List[Dict]:
        """E-utilities efetch: PMIDs → {pmid, title, abstract, year, journal}."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:PUBMED_TOP_K]),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.email,
        }
        url = self.BASE_FETCH + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml_text = resp.read().decode("utf-8")
            return self._parse_efetch_xml(xml_text)
        except Exception as e:
            print(f"    [PubMed] efetch error: {e}")
            # Fallback: try JSON summary
            return self._efetch_summary(pmids)

    def _efetch_summary(self, pmids: List[str]) -> List[Dict]:
        """Fallback: use esummary (JSON) for basic info."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:PUBMED_TOP_K]),
            "retmode": "json",
            "email": self.email,
        }
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("result", {})
            papers = []
            for pmid in pmids:
                info = results.get(pmid, {})
                if info and isinstance(info, dict):
                    papers.append({
                        "pmid": pmid,
                        "title": info.get("title", ""),
                        "abstract": "",  # esummary doesn't include abstract
                        "year": info.get("pubdate", "")[:4],
                        "journal": info.get("source", ""),
                    })
            return papers[:PUBMED_TOP_K]
        except Exception as e:
            print(f"    [PubMed] esummary error: {e}")
            return []

    def _parse_efetch_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed efetch XML response."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid = article.findtext(".//PMID", "")
                    title = article.findtext(".//ArticleTitle", "")
                    abstract_parts = article.findall(".//AbstractText")
                    abstract = " ".join(
                        (a.text or "") + "".join((e.tail or "") for e in a)
                        for a in abstract_parts
                    )
                    if not abstract:
                        abstract = article.findtext(".//Abstract/AbstractText", "")

                    year = article.findtext(".//PubDate/Year", "")
                    journal = article.findtext(".//Journal/Title", "")

                    papers.append({
                        "pmid": pmid,
                        "title": title.strip() if title else "",
                        "abstract": abstract.strip() if abstract else "",
                        "year": year.strip() if year else "",
                        "journal": journal.strip() if journal else "",
                    })
                except Exception:
                    continue

            print(f"    [PubMed] Parsed {len(papers)} abstracts")
            return papers[:PUBMED_TOP_K]
        except ET.ParseError as e:
            print(f"    [PubMed] XML parse error: {e}")
            return []
