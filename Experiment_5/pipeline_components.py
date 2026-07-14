"""
Experiment_5/pipeline_components.py — Core infrastructure for the 4-stage pipeline.
LLM clients, dual-judge scoring, memory manager, PDF parser, and base classes.
"""

import os
import re
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import httpx

from config_exp5 import (
    DEEPSEEK_API_KEY, AUTODL_API_KEY,
    DEEPSEEK_BASE_URL, AUTODL_BASE_URL,
    GENERATOR_MODEL, JUDGE_MODEL_1, JUDGE_MODEL_2,
    MEMORY_DIR, ITERATION_ISSUES_PATH, PROMPT_ISSUES_PATH,
    PDF_PAPERS_DIR,
)

# =========================================================
# LLM Clients
# =========================================================
generator_client = OpenAI(
    api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
    timeout=httpx.Timeout(300.0)
)
judge_client = OpenAI(
    api_key=AUTODL_API_KEY, base_url=AUTODL_BASE_URL,
    timeout=httpx.Timeout(300.0)
)


def call_llm(client: OpenAI, model: str, system: str, prompt: str,
             temperature: float = 0.1, max_tokens: int = 8192, max_retries: int = 5) -> str:
    """Call LLM with exponential backoff."""
    sleep_time = 1
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [API] {model} attempt {attempt}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=temperature, timeout=180, max_tokens=max_tokens,
            )
            print(f"    [API] {model} OK")
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] All retries failed"


# =========================================================
# Dual-Judge Scoring
# =========================================================

def parse_score(text: Optional[str]) -> Dict[str, int]:
    """Parse Reliability: xx / Innovation: xx from judge output."""
    if text is None:
        return {"reliability": 0, "innovation": 0}
    try:
        rel = re.search(r"Reliability\s*:\s*(\d+)", text, re.IGNORECASE)
        inn = re.search(r"Innovation\s*:\s*(\d+)", text, re.IGNORECASE)
        return {
            "reliability": int(rel.group(1)) if rel else 0,
            "innovation": int(inn.group(1)) if inn else 0,
        }
    except Exception:
        return {"reliability": 0, "innovation": 0}


def parse_generic_score(text: Optional[str], key1: str = "Quality",
                        key2: str = "Completeness") -> Dict[str, int]:
    """Parse any two-dimension score like 'Quality: xx / Completeness: xx'."""
    if text is None:
        return {key1.lower(): 0, key2.lower(): 0}
    try:
        v1 = re.search(rf"{key1}\s*:\s*(\d+)", text, re.IGNORECASE)
        v2 = re.search(rf"{key2}\s*:\s*(\d+)", text, re.IGNORECASE)
        return {
            key1.lower(): int(v1.group(1)) if v1 else 0,
            key2.lower(): int(v2.group(1)) if v2 else 0,
        }
    except Exception:
        return {key1.lower(): 0, key2.lower(): 0}


def dual_judge_score(plan_text: str, label: str = "plan", verbose: bool = True) -> Dict:
    """Score a text with both judges (MiniMax-M2.5 + GPT-5.4)."""
    scoring_prompt = """You are an expert scientific reviewer.
Evaluate the given content from two aspects:
1. Method Reliability (0-100): technical correctness, experimental rigor, feasibility
2. Method Innovation (0-100): originality, novelty, creativity, uniqueness

Output:
Reliability: xx
Innovation: xx
Only output the scores."""

    if verbose:
        print(f"    [Judge1] {JUDGE_MODEL_1} scoring {label}...")
    j1_text = call_llm(judge_client, JUDGE_MODEL_1, "Expert scientific reviewer.", scoring_prompt + f"\n\n{plan_text[:6000]}", temperature=0.0)
    j1 = parse_score(j1_text)

    if verbose:
        print(f"    [Judge2] {JUDGE_MODEL_2} scoring {label}...")
    j2_text = call_llm(judge_client, JUDGE_MODEL_2, "Expert scientific reviewer.", scoring_prompt + f"\n\n{plan_text[:6000]}", temperature=0.0)
    j2 = parse_score(j2_text)

    combined_rel = round((j1["reliability"] + j2["reliability"]) / 2, 1)
    combined_inn = round((j1["innovation"] + j2["innovation"]) / 2, 1)
    reward = round(combined_rel * 0.6 + combined_inn * 0.4, 1)

    if verbose:
        print(f"    Scores: J1(R={j1['reliability']} I={j1['innovation']}) "
              f"J2(R={j2['reliability']} I={j2['innovation']}) "
              f"→ C(R={combined_rel} I={combined_inn}) reward={reward}")

    return {"judge1": j1, "judge2": j2, "combined": {"reliability": combined_rel, "innovation": combined_inn}, "reward": reward}


# =========================================================
# Memory Manager (persistent .md issue tracking)
# =========================================================

class MemoryManager:
    """Reads/writes persistent .md files to track issues across runs."""

    def __init__(self, issues_path: str):
        self.path = issues_path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(f"# Issues Log — {os.path.basename(self.path)}\n\n"
                        "| # | Timestamp | Question | Issue | Resolution |\n"
                        "|---|-----------|----------|-------|------------|\n")

    def read(self) -> str:
        """Read all past issues."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def append(self, question: str, issue: str, resolution: str = ""):
        """Append a new issue."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row = f"| {self._count() + 1} | {timestamp} | {question[:60]} | {issue[:120]} | {resolution[:120]} |\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(row)

    def _count(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.startswith("|") and not line.startswith("|---") and not line.startswith("| #"))
        except Exception:
            return 0

    def get_summary(self, max_lines: int = 20) -> str:
        content = self.read()
        lines = content.split("\n")
        # Return header + last N issue lines
        header = [l for l in lines if l.startswith("#") or l.startswith("| #") or l.startswith("|---")]
        issue_lines = [l for l in lines if l.startswith("|") and not l.startswith("| #") and not l.startswith("|---")]
        recent = issue_lines[-max_lines:] if len(issue_lines) > max_lines else issue_lines
        return "\n".join(header + recent)


# Global memory instances
_iteration_memory = MemoryManager(ITERATION_ISSUES_PATH)
_prompt_memory = MemoryManager(PROMPT_ISSUES_PATH)


def get_iteration_memory() -> MemoryManager:
    return _iteration_memory


def get_prompt_memory() -> MemoryManager:
    return _prompt_memory


# =========================================================
# PDF Parser
# =========================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file. Tries multiple backends."""
    # Try PyPDF2 first
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
            result = "\n".join(text)
            if len(result.strip()) > 200:
                return result
    except ImportError:
        pass
    except Exception:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
            result = "\n".join(text)
            if len(result.strip()) > 200:
                return result
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: raw text extraction (may be messy)
    try:
        with open(pdf_path, "rb") as f:
            raw = f.read()
        # Try to find text between stream/endstream
        text = raw.decode("latin-1", errors="ignore")
        # Filter printable ASCII
        lines = [l for l in text.split("\n") if len(l.strip()) > 30 and not l.strip().startswith("%")]
        return "\n".join(lines[:200])  # Limit to 200 lines
    except Exception:
        pass

    return f"[Could not extract text from {os.path.basename(pdf_path)}]"


def load_pdfs_from_directory(pdf_dir: str = PDF_PAPERS_DIR) -> List[Dict]:
    """Load all PDFs from a directory, extracting text from each."""
    os.makedirs(pdf_dir, exist_ok=True)  # Auto-create if missing
    papers = []
    if not os.path.isdir(pdf_dir):
        print(f"  [PDF] Directory not found: {pdf_dir}")
        return papers

    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(pdf_dir, fname)
            print(f"  [PDF] Extracting: {fname}")
            text = extract_text_from_pdf(fpath)
            papers.append({
                "source": "pdf",
                "filename": fname,
                "path": fpath,
                "text": text[:5000],
                "full_text": text,
            })
    print(f"  [PDF] Loaded {len(papers)} papers")
    return papers


# =========================================================
# Data Loading
# =========================================================

def load_bio_test_questions(path: str, n: int = 10) -> List[Dict]:
    """Load questions from bio_test.jsonl."""
    from config_exp5 import BIO_TEST_PATH
    if not path:
        path = BIO_TEST_PATH
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if n and n < len(data):
        random.seed(42)
        data = random.sample(data, n)

    return data


def extract_question(entry: Dict) -> str:
    """Extract clean research question from a bio_test entry."""
    for msg in entry.get("messages", []):
        if msg["role"] == "user":
            content = msg["content"]
            match = re.search(r"<user_request>\s*(.*?)(?:<think>|\$)", content, re.DOTALL)
            if match:
                return match.group(1).strip()[:500]
            return content[:500]
    return ""


# =========================================================
# Score Aggregation Utilities
# =========================================================

def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


class ScoreAccumulator:
    """Accumulates and averages scores across all 10 questions."""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}

    def add(self, key: str, value: float):
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value)

    def avg(self, key: str) -> float:
        return _avg(self.records.get(key, []))

    def summary(self) -> Dict:
        return {k: {"avg": _avg(v), "min": min(v), "max": max(v)} for k, v in self.records.items()}
