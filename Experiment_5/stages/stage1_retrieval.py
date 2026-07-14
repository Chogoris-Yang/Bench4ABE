"""
Experiment_5/stages/stage1_retrieval.py — Stage 1: Information Retrieval.
Supports 3 modes: PubMed only, PDF folder only, Hybrid (PubMed + PDF).
Retrieved context is scored for relevance and completeness.
"""

import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

from pipeline_components import (
    call_llm, generator_client, judge_client, GENERATOR_MODEL, JUDGE_MODEL_1,
    load_pdfs_from_directory, parse_generic_score,
)
from config_exp5 import (
    RETRIEVAL_MODE, RETRIEVAL_TOP_K, PUBMED_EMAIL, PDF_PAPERS_DIR,
)


# =========================================================
# PubMed Client
# =========================================================

class PubMedClient:
    BASE_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        pmids = self._esearch(query, max_results)
        if not pmids:
            return []
        return self._efetch(pmids)

    def _esearch(self, query: str, max_results: int) -> List[str]:
        params = {"db": "pubmed", "term": query[:400], "retmax": max_results,
                  "retmode": "json", "sort": "relevance", "email": PUBMED_EMAIL}
        url = self.BASE_SEARCH + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"    [PubMed] Search error: {e}")
            return []

    def _efetch(self, pmids: List[str]) -> List[Dict]:
        params = {"db": "pubmed", "id": ",".join(pmids[:RETRIEVAL_TOP_K]),
                  "retmode": "xml", "rettype": "abstract", "email": PUBMED_EMAIL}
        url = self.BASE_FETCH + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=20) as resp:
                xml_text = resp.read().decode("utf-8")
            return self._parse_xml(xml_text)
        except Exception as e:
            print(f"    [PubMed] Fetch error: {e}")
            return []

    def _parse_xml(self, xml_text: str) -> List[Dict]:
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    papers.append({
                        "pmid": article.findtext(".//PMID", ""),
                        "title": (article.findtext(".//ArticleTitle", "") or "").strip(),
                        "abstract": " ".join(
                            (a.text or "") + "".join((e.tail or "") for e in a)
                            for a in article.findall(".//AbstractText")
                        ).strip(),
                        "year": (article.findtext(".//PubDate/Year", "") or "").strip(),
                        "journal": (article.findtext(".//Journal/Title", "") or "").strip(),
                    })
                except Exception:
                    continue
        except ET.ParseError:
            pass
        return papers[:RETRIEVAL_TOP_K]


# =========================================================
# BM25 Sparse Retriever
# =========================================================

class BM25Retriever:
    """Simple BM25 for matching queries to documents."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def retrieve(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        if not docs:
            return []
        q_tokens = self._tokenize(query)
        doc_tokens = [self._tokenize(d.get("text", d.get("abstract", ""))) for d in docs]
        dlens = [len(dt) for dt in doc_tokens]
        avg_dl = sum(dlens) / len(dlens) if dlens else 1
        df = {}
        for dt in doc_tokens:
            for t in set(dt):
                df[t] = df.get(t, 0) + 1
        N = len(docs)
        scores = []
        for i, dt in enumerate(doc_tokens):
            tf = {}
            for t in dt:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in q_tokens:
                if t not in df:
                    continue
                idf = __import__('math').log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                num = tf.get(t, 0) * (self.k1 + 1)
                denom = tf.get(t, 0) + self.k1 * (1 - self.b + self.b * dlens[i] / avg_dl)
                score += idf * num / denom
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{**docs[idx], "bm25_score": round(s, 4)} for idx, s in scores[:top_k] if s > 0]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-z0-9]{2,}', text.lower())


# =========================================================
# Query Builder
# =========================================================

def build_pubmed_query(question: str) -> str:
    """Extract a PubMed-friendly query from a research question."""
    bio_terms = [
        "genomic", "sequence", "enhancer", "promoter", "chromatin",
        "Hi-C", "scRNA-seq", "scATAC-seq", "single-cell", "transcriptom",
        "protein", "structure", "post-translational", "phosphorylation",
        "glycosylation", "ubiquitin", "AlphaFold",
        "drug-target", "binding", "pocket", "ligand", "pharmacophore",
        "metagenom", "strain", "binning", "microbiome",
        "splicing", "RBP", "RNA-binding", "regulatory",
        "variant", "non-coding", "GWAS", "pathogenic", "eQTL",
        "spatial transcriptom", "MERFISH", "Visium", "imputation",
        "gene regulatory network", "GRN",
        "deep learning", "transformer", "state space model",
        "graph neural network", "contrastive learning",
    ]
    q_lower = question.lower()
    matched = [t for t in bio_terms if t.lower() in q_lower]
    if matched:
        query = " AND ".join(f'"{t}"' if " " in t else t for t in matched[:4])
    else:
        words = list(dict.fromkeys(re.findall(r'[a-zA-Z]{4,}', question)))[:5]
        query = " AND ".join(words)
    return query[:400] + ' AND ("method"[Title/Abstract] OR "computational"[Title/Abstract])'


# =========================================================
# Retrieval Stage
# =========================================================

SCORING_PROMPT = """Evaluate this retrieved information for a research task.

Research Question: {question}

Retrieved Content:
{retrieved_text}

Evaluate:
1. Relevance (0-100): How relevant is the retrieved info to the question?
2. Completeness (0-100): Does it cover the key aspects needed to design an experiment?

Output:
Relevance: xx
Completeness: xx"""


def run_stage1(question: str, mode: str = None) -> Dict:
    """
    Stage 1: Information Retrieval.

    Args:
        question: The research question
        mode: "pubmed" | "pdf" | "hybrid" (default from config)

    Returns:
        {retrieved_docs, context_text, scores, mode}
    """
    if mode is None:
        mode = RETRIEVAL_MODE

    print(f"\n{'='*60}")
    print(f"[Stage 1] Information Retrieval — mode={mode}")
    print(f"{'='*60}")

    bm25 = BM25Retriever()
    pubmed = PubMedClient()
    all_docs = []

    # ── PubMed retrieval ──
    if mode in ("pubmed", "hybrid"):
        query = build_pubmed_query(question)
        print(f"  [PubMed] Query: {query[:200]}")
        pubmed_papers = pubmed.search(query)
        for p in pubmed_papers:
            p["text"] = f"Title: {p.get('title','')}\nAbstract: {p.get('abstract','')}"
            p["source"] = "pubmed"
        all_docs.extend(pubmed_papers)
        print(f"  [PubMed] Found {len(pubmed_papers)} papers")

    # ── PDF retrieval ──
    if mode in ("pdf", "hybrid"):
        pdf_papers = load_pdfs_from_directory(PDF_PAPERS_DIR)
        for p in pdf_papers:
            p["text"] = f"File: {p.get('filename','')}\n{p.get('text','')}"
        all_docs.extend(pdf_papers)
        print(f"  [PDF] Loaded {len(pdf_papers)} papers")

    # ── Retrieve top-K ──
    if all_docs:
        retrieved = bm25.retrieve(question, all_docs, top_k=RETRIEVAL_TOP_K)
    else:
        retrieved = []

    # Build context
    parts = []
    for i, doc in enumerate(retrieved):
        src = doc.get("source", "?")
        if src == "pubmed":
            parts.append(f"[Paper {i+1}] PMID:{doc.get('pmid','?')} {doc.get('title','')}\n{doc.get('abstract','')}")
        else:
            parts.append(f"[Paper {i+1}] {doc.get('filename','?')}\n{doc.get('text','')[:2000]}")

    context = "\n\n---\n\n".join(parts) if parts else "(No relevant documents found)"

    # ── Score retrieval quality ──
    print(f"\n  [Scoring] Retrieval quality...")
    score_text = call_llm(
        judge_client, JUDGE_MODEL_1,
        "You evaluate retrieval quality for research tasks.",
        SCORING_PROMPT.format(question=question, retrieved_text=context[:5000]),
        temperature=0.0, max_tokens=256,
    )
    scores = parse_generic_score(score_text, "Relevance", "Completeness")
    scores["retrieval_quality"] = round((scores.get("relevance", 0) + scores.get("completeness", 0)) / 2, 1)
    print(f"    Relevance={scores.get('relevance',0)} Completeness={scores.get('completeness',0)} → {scores['retrieval_quality']}")

    return {
        "mode": mode,
        "retrieved_docs": retrieved,
        "context": context,
        "scores": scores,
        "doc_count": len(retrieved),
    }
