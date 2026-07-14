"""
Experiment_4/retrieval/retriever.py — Unified retrieval interface.
Implements Dense (embedding), Sparse (BM25), and Hybrid retrievers.
"""

import os
import json
import math
import re
import time
from collections import Counter
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from retrieval.pubmed_client import PubMedClient
from config_rag import (
    RETRIEVAL_TOP_K, SPARSE_K1, SPARSE_B, HYBRID_DENSE_CANDIDATES,
    RAG_CONTEXT_MAX_TOKENS,
)

# =========================================================
# Local TF-IDF Vectorizer (no external API needed)
# =========================================================

class TFIDFVectorizer:
    """
    Pure Python TF-IDF vectorizer — no API, no numpy, no sklearn.
    Builds vocabulary from documents, vectorizes queries/docs into sparse vectors.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}       # token → index
        self.idf: Dict[str, float] = {}        # token → IDF
        self.vocab_size: int = 0

    def fit(self, texts: List[str]):
        """Build vocabulary and compute IDF from a corpus."""
        doc_count = len(texts)
        df = Counter()

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                df[token] += 1

        # Keep top max_features by document frequency
        top_tokens = [t for t, _ in df.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top_tokens)}
        self.vocab_size = len(self.vocab)

        # Compute IDF
        for token, idx in self.vocab.items():
            self.idf[token] = math.log((doc_count + 1) / (df.get(token, 0) + 1)) + 1.0

    def transform(self, text: str) -> Dict[int, float]:
        """Convert text to sparse TF-IDF vector {index: value}."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        vec = {}
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tf_val = count / len(tokens) if tokens else 0
                vec[idx] = tf_val * self.idf.get(token, 1.0)
        return vec

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase + split on non-alphanumeric + filter short tokens."""
        return [t for t in re.findall(r'[a-z0-9]{2,}', text.lower())
                if t not in _STOP_WORDS]


# Minimal English stop words
_STOP_WORDS = set(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might must can could and or not no but if then else when "
    "where why how who whom which what this that these those it its we you they "
    "he she them their our my your his her me us to of in for on with at by from "
    "about into through during before after above below between up down out off "
    "over under again further then once here there all both each every any most "
    "other some such only own same so than too very just because as until while "
    "also".split()
)


def _sparse_dot(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Dot product of two sparse vectors."""
    if len(a) > len(b):
        a, b = b, a  # iterate over smaller
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _sparse_norm(a: Dict[int, float]) -> float:
    """L2 norm of sparse vector."""
    return math.sqrt(sum(v * v for v in a.values()))


def cosine_similarity_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not a or not b:
        return 0.0
    dot = _sparse_dot(a, b)
    norm_a = _sparse_norm(a)
    norm_b = _sparse_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =========================================================
# Document Store
# =========================================================

class DocumentStore:
    """In-memory document store with TF-IDF vectors (local, no API)."""

    def __init__(self):
        self.docs: List[Dict] = []
        self._texts: List[str] = []
        self._vectorizer: Optional[TFIDFVectorizer] = None
        self._vectors: List[Dict[int, float]] = []

    def add_pubmed_papers(self, papers: List[Dict]):
        """Add PubMed paper results to the store."""
        for paper in papers:
            text = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"
            if not text.strip() or text.strip() == "Title: \nAbstract: ":
                continue
            self.docs.append({
                "source": "pubmed",
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "text": text,
                "year": paper.get("year", ""),
                "journal": paper.get("journal", ""),
            })
            self._texts.append(text)

    def add_prior_protocols(self, protocols: List[Dict]):
        """Add prior experimental protocols."""
        for proto in protocols:
            text = proto.get("text", proto.get("plan", str(proto)))
            if not text.strip():
                continue
            self.docs.append({
                "source": "prior_protocol",
                "task": proto.get("task", proto.get("question", "")),
                "text": text,
            })
            self._texts.append(text)

    def add_texts(self, texts: List[str], source: str = "generic"):
        """Add raw text documents."""
        for text in texts:
            if not text.strip():
                continue
            self.docs.append({"source": source, "text": text})
            self._texts.append(text)

    def get_texts(self) -> List[str]:
        return self._texts

    def get_docs(self) -> List[Dict]:
        return self.docs

    def size(self) -> int:
        return len(self.docs)

    def build_vectors(self) -> List[Dict[int, float]]:
        """Build TF-IDF vectors for all documents (local, no API)."""
        if self.size() == 0:
            return []
        print(f"  [Store] Building TF-IDF vectors for {self.size()} documents...")
        self._vectorizer = TFIDFVectorizer()
        self._vectorizer.fit(self._texts)
        self._vectors = [self._vectorizer.transform(t) for t in self._texts]
        print(f"  [Store] TF-IDF vocab size: {self._vectorizer.vocab_size}")
        return self._vectors

    def get_vectors(self) -> List[Dict[int, float]]:
        return self._vectors

    def get_vectorizer(self) -> Optional[TFIDFVectorizer]:
        return self._vectorizer


# =========================================================
# Abstract Retriever
# =========================================================

class BaseRetriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    def retrieve(self, query: str, store: DocumentStore, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


# =========================================================
# Dense Retriever (embedding + cosine similarity)
# =========================================================

class DenseRetriever(BaseRetriever):
    """Dense retrieval using local TF-IDF vectors + cosine similarity (no API needed)."""

    def name(self) -> str:
        return "dense"

    def retrieve(self, query: str, store: DocumentStore, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        if store.size() == 0:
            return []

        # Get or build TF-IDF vectors
        vectors = store.get_vectors()
        vec = store.get_vectorizer()
        if not vectors or not vec:
            print("  [Dense] Building TF-IDF vectors...")
            vectors = store.build_vectors()
            vec = store.get_vectorizer()

        if not vectors or not vec:
            return []

        # Vectorize query
        print(f"  [Dense] Vectorizing query ({len(query)} chars)...")
        query_vec = vec.transform(query)

        if not query_vec:
            return []

        # Cosine similarity
        scores = []
        for i, doc_vec in enumerate(vectors):
            sim = cosine_similarity_sparse(query_vec, doc_vec)
            if sim > 0:
                scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in scores[:top_k]:
            doc = store.get_docs()[idx]
            results.append({**doc, "retrieval_score": round(sim, 4)})

        if results:
            print(f"  [Dense] Retrieved {len(results)} docs (top sim={scores[0][1]:.4f}: {results[0].get('title','')[:60]})")
        else:
            print(f"  [Dense] No matching documents found")
        return results


# =========================================================
# Sparse Retriever (BM25)
# =========================================================

class SparseRetriever(BaseRetriever):
    """BM25 sparse retrieval — no embeddings needed."""

    def __init__(self, k1: float = SPARSE_K1, b: float = SPARSE_B):
        self.k1 = k1
        self.b = b

    def name(self) -> str:
        return "sparse"

    def retrieve(self, query: str, store: DocumentStore, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        if store.size() == 0:
            return []

        docs = store.get_docs()
        texts = store.get_texts()

        # Tokenize
        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(t) for t in texts]

        # Compute document frequencies
        df = {}
        doc_len = []
        for tokens in doc_tokens_list:
            doc_len.append(len(tokens))
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        avg_dl = sum(doc_len) / len(doc_len) if doc_len else 1
        N = len(docs)

        # Score each document
        scores = []
        for i, doc_tokens in enumerate(doc_tokens_list):
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                if token not in df:
                    continue
                idf = math.log((N - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
                token_tf = tf.get(token, 0)
                numerator = token_tf * (self.k1 + 1)
                denominator = token_tf + self.k1 * (1 - self.b + self.b * doc_len[i] / avg_dl)
                score += idf * numerator / denominator
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                doc = docs[idx]
                results.append({**doc, "retrieval_score": round(score, 4)})

        print(f"  [Sparse] Retrieved {len(results)} docs (top BM25: {scores[0][1]:.4f})" if results else "  [Sparse] No results")
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]{2,}', text.lower())


# =========================================================
# Hybrid Retriever (Dense → Sparse Rerank)
# =========================================================

class HybridRetriever(BaseRetriever):
    """Dense retrieval → BM25 reranking."""

    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()

    def name(self) -> str:
        return "hybrid"

    def retrieve(self, query: str, store: DocumentStore, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        if store.size() == 0:
            return []

        # Phase 1: Dense retrieval (wider)
        candidate_k = min(HYBRID_DENSE_CANDIDATES, store.size())
        print(f"  [Hybrid] Phase 1: Dense retrieval ({candidate_k} candidates)")
        candidates = self.dense.retrieve(query, store, top_k=candidate_k)

        if not candidates:
            return []

        # Phase 2: Build temporary store from candidates for BM25 reranking
        tmp_store = DocumentStore()
        for c in candidates:
            tmp_store.add_texts([c.get("text", "")], source=c.get("source", "candidate"))

        # Phase 3: BM25 rerank
        print(f"  [Hybrid] Phase 2: BM25 rerank (top {top_k})")
        reranked = self.sparse.retrieve(query, tmp_store, top_k=top_k)

        # Map back to original docs
        results = []
        for rr in reranked:
            for c in candidates:
                if c.get("text") == rr.get("text"):
                    results.append({**c, "rerank_score": rr.get("retrieval_score", 0)})
                    break

        print(f"  [Hybrid] Final: {len(results)} docs")
        return results[:top_k]


# =========================================================
# Knowledge Source Builder
# =========================================================

class KnowledgeSource:
    """
    Builds a DocumentStore from multiple knowledge sources:
    1. PubMed papers (live search)
    2. Prior protocols (from bio_test.jsonl or similar)
    """

    def __init__(self):
        self.pubmed = PubMedClient()

    def build_store(self, question: str, prior_protocols: List[Dict] = None) -> DocumentStore:
        """
        Build a DocumentStore for a given research question.

        Args:
            question: The research task description
            prior_protocols: Optional list of {task, text} prior protocols
        """
        store = DocumentStore()

        # Source 1: PubMed papers
        print(f"  [Knowledge] Fetching PubMed papers...")
        papers = self.pubmed.search_for_task(question)
        if papers:
            store.add_pubmed_papers(papers)
            print(f"  [Knowledge] Added {len(papers)} PubMed papers")
        else:
            print(f"  [Knowledge] No PubMed results — will use prior protocols only")

        # Source 2: Prior protocols
        if prior_protocols:
            store.add_prior_protocols(prior_protocols)
            print(f"  [Knowledge] Added {len(prior_protocols)} prior protocols")

        print(f"  [Knowledge] Total store size: {store.size()} documents")
        return store

    def build_store_from_sources(
        self,
        question: str,
        sources: List[str],
        prior_protocols: List[Dict] = None,
    ) -> DocumentStore:
        """
        Build a store from specified sources: ["papers", "prior", "all"].

        - "papers": Only PubMed papers
        - "prior": Only prior protocols
        - "all": Both papers and protocols
        """
        store = DocumentStore()

        if "papers" in sources or "all" in sources:
            papers = self.pubmed.search_for_task(question)
            if papers:
                store.add_pubmed_papers(papers)

        if ("prior" in sources or "all" in sources) and prior_protocols:
            store.add_prior_protocols(prior_protocols)

        return store


# =========================================================
# Convenience Factory
# =========================================================

def get_retriever(method: str) -> BaseRetriever:
    """Factory: get retriever by name."""
    if method == "dense":
        return DenseRetriever()
    elif method == "sparse":
        return SparseRetriever()
    elif method == "hybrid":
        return HybridRetriever()
    else:
        raise ValueError(f"Unknown retriever: {method}. Options: dense, sparse, hybrid")
