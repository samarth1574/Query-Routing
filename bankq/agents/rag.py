"""Retrieval-augmented generation: find the policy passages that ground an agent's answer.

The knowledge base is a folder of Markdown files (knowledge_base/*.md). Each file has a
small front-matter block with its category, and is split into passages at "## " headings.

Retrieval is TF-IDF (word 1–2 grams) + cosine similarity, optionally restricted to one
category. It is fast (sub-millisecond per query), needs no GPU and no network, and is easy
to swap for a dense embedding index behind the same `search` interface.

Indexing is embarrassingly parallel: each document is parsed independently, so
`KnowledgeBase.load` parses files with a thread pool.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass(frozen=True)
class Passage:
    id: str
    category: str
    source: str
    heading: str
    text: str

    def to_dict(self, score: float | None = None) -> dict:
        d = {"id": self.id, "category": self.category, "source": self.source, "heading": self.heading, "text": self.text}
        if score is not None:
            d["score"] = round(float(score), 4)
        return d


def parse_markdown(path: Path) -> list[Passage]:
    raw = path.read_text(encoding="utf-8")
    meta = {}
    m = _FRONT.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    category = meta.get("category", path.stem)
    passages = []
    for i, block in enumerate(re.split(r"^## ", raw, flags=re.M)):
        block = block.strip()
        if not block:
            continue
        heading, _, body = block.partition("\n")
        body = " ".join(body.split())
        if body:
            passages.append(Passage(f"{path.stem}#{i}", category, meta.get("title", path.stem), heading.strip(), body))
    return passages


class KnowledgeBase:
    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("Knowledge base is empty")
        self.passages = passages
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        # Headings are short and highly informative, so they are counted twice.
        self.matrix = self.vectorizer.fit_transform([f"{p.heading} {p.heading} {p.text}" for p in passages])
        self._by_category = {}
        for i, p in enumerate(passages):
            self._by_category.setdefault(p.category, []).append(i)

    @classmethod
    def load(cls, folder: Path, workers: int = 4) -> KnowledgeBase:
        files = sorted(Path(folder).glob("*.md"))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            parsed = list(pool.map(parse_markdown, files))
        return cls([p for ps in parsed for p in ps])

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_category)

    def search(self, query: str, category: str | None = None, k: int = 3, min_score: float = 0.0) -> list[dict]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        idx = self._by_category.get(category, []) if category else range(len(self.passages))
        ranked = sorted(idx, key=lambda i: -scores[i])[:k]
        return [self.passages[i].to_dict(scores[i]) for i in ranked if scores[i] > min_score]

    def search_many(self, queries: list[str], k: int = 3) -> list[list[dict]]:
        """Vectorised retrieval for a batch of queries: one sparse matrix product instead of N."""
        q = self.vectorizer.transform(queries)
        scores = (q @ self.matrix.T).toarray()
        out = []
        for row in scores:
            top = np.argsort(row)[::-1][:k]
            out.append([self.passages[i].to_dict(row[i]) for i in top if row[i] > 0])
        return out
