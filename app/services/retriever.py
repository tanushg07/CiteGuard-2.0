"""Evidence passage retrieval for one claim against the passages of ONE cited source (source-isolated).

Stages
1. Passage construction: each source paragraph is split into sentence windows (up to 3 sentences, stride 2, capped
   at `chunk_size` words) so the NLI premise stays short and focused. The full paragraph is kept as `context`.
   Bibliography, acknowledgement and similar non-evidence sections are excluded; exact duplicate passages dropped.
2. Lexical candidate retrieval: TF-IDF cosine blended with BM25 (index built once per source set).
3. Cross-encoder reranking of all lexical candidates (semantic relevance). The cross-encoder only RANKS passages;
   it never decides whether a citation is correct. Lexical overlap is never treated as support.
4. Near-duplicate removal among the selected passages; source page/section/paragraph and ranks are preserved.
"""
import logging
import math
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.model_health import ModelUnavailable
from app.core.models import cross_encoder
from app.core.schemas import ClaimCitationPair, RetrievedEvidence

NON_EVIDENCE = re.compile(r'^(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|works cited|literature cited|'
                          r'acknowledg(?:e)?ments?|funding|author contributions|competing interests)\b', re.I)
CITATION = re.compile(r'\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]')


def CrossEncoder(*args, **kwargs):
    return cross_encoder()


logger = logging.getLogger(__name__)


def _norm(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def _jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


class Retriever:
    def __init__(self, use_reranker: bool = True, chunk_size: int = 512, top_k: int = 3, window_sentences: int = 3):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.window_sentences = window_sentences
        self.use_reranker = use_reranker

        self.chunks: list[str] = []
        self.source_citations: dict[int, str] = {}
        self.source_pages: dict[int, int] = {}
        self.provenance = {}
        self._tfidf_matrix = None
        self._vectorizer: TfidfVectorizer | None = None
        self._docs: list[Counter] = []
        self._lengths: list[int] = []
        self._avg_length = 1.0
        self._df: Counter = Counter()
        self.stats = {'rerank_calls': 0, 'rerank_pairs': 0, 'rerank_failures': 0}

        self.reranker = None
        self.reranker_unavailable: dict | None = None  # why semantic reranking is off; surfaced in the health report
        if self.use_reranker:
            try:
                self.reranker = CrossEncoder()  # shared, probe-verified instance from the model registry
            except ModelUnavailable as exc:
                self.reranker_unavailable = exc.status.as_dict()
                logger.warning("Cross-encoder unavailable (%s at %s): %s. Lexical retrieval will only propose candidates.",
                               exc.status.cause, exc.status.stage, exc.status.error)
            except Exception as exc:  # noqa: BLE001
                self.reranker_unavailable = {'status': 'UNAVAILABLE', 'cause': 'INIT_ERROR', 'error': f'{type(exc).__name__}: {exc}'[:300]}
                logger.warning("Cross-encoder unavailable: %s", exc)
            if self.reranker is None:
                self.use_reranker = False
        else:
            self.reranker_unavailable = {'status': 'DISABLED', 'cause': 'DISABLED', 'error': 'Reranking disabled by configuration.'}

    def load_source(self, text, name):
        self.add_sources([{"name": name, "text": text}])

    def clear(self):
        self.chunks = []
        self.source_citations = {}
        self.source_pages = {}
        self.provenance = {}
        self._tfidf_matrix = None
        self._vectorizer = None
        self._docs, self._lengths, self._avg_length, self._df = [], [], 1.0, Counter()

    def _windows(self, paragraph: str) -> list[str]:
        from app.services.extractor import Extractor
        sentences = Extractor()._split_sentences(paragraph) or [paragraph]
        if len(sentences) <= self.window_sentences and len(paragraph.split()) <= self.chunk_size:
            return [paragraph]
        out, step = [], max(1, self.window_sentences - 1)
        for start in range(0, len(sentences), step):
            window = ' '.join(sentences[start:start + self.window_sentences])
            words = window.split()
            for i in range(0, len(words), self.chunk_size):
                out.append(' '.join(words[i:i + self.chunk_size]))
            if start + self.window_sentences >= len(sentences):
                break
        return out

    def add_sources(self, sources: list[dict[str, Any]]):
        """Build passages and the lexical index ONCE; retrieve() only transforms the query."""
        self.clear()
        all_chunks: list[str] = []
        citation_map: dict[int, str] = {}
        seen: set[str] = set()
        for src in sources:
            name = src.get("name", "Unknown Source")
            text = src.get("text", "")
            if not text.strip() or NON_EVIDENCE.match(src.get('section') or ''):
                continue
            raw = [c.strip() for c in text.split("\n\n") if c.strip()] or [text.strip()]
            for paragraph in raw:
                clean = ' '.join(paragraph.split())
                if len(clean) < 15:
                    continue
                for passage in self._windows(clean):
                    key = _norm(passage)
                    if len(passage) < 15 or key in seen:
                        continue
                    seen.add(key)
                    idx = len(all_chunks)
                    all_chunks.append(passage)
                    citation_map[idx] = name
                    self.source_pages[idx] = src.get('page_number')
                    self.provenance[idx] = {
                        **{k: src.get(k) for k in ('source_id', 'source_url', 'section', 'paragraph', 'extraction_method')},
                        'evidence_tier': src.get('evidence_tier') or 'full_text',
                        'context': clean[:2500] if clean != passage else None}
        if not all_chunks:
            logger.error("No valid chunks extracted from any source document.")
            return
        self.chunks = all_chunks
        self.source_citations = citation_map
        self._docs = [Counter(re.findall(r'\w+', chunk.lower())) for chunk in all_chunks]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) or 1.0
        self._df = Counter(term for doc in self._docs for term in doc)
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            self._tfidf_matrix = self._vectorizer.fit_transform(self.chunks)
        except ValueError:
            self.clear()
        logger.info("Retrieval index built: %d passages from %d source paragraphs.", len(self.chunks), len(sources))

    def _lexical(self, query: str) -> np.ndarray:
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        docs, lengths, average = self._docs, self._lengths, self._avg_length
        bm25 = np.zeros(len(docs))
        for term in set(re.findall(r'\w+', query.lower())):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
            for i, doc in enumerate(docs):
                frequency = doc[term]
                if frequency:
                    bm25[i] += idf * frequency * 2.5 / (frequency + 1.5 * (.25 + .75 * lengths[i] / average))
        if bm25.max() > 0:
            similarities = .5 * similarities + .5 * bm25 / bm25.max()
        return similarities

    def _evidence(self, claim, idx, lexical, lexical_rank, candidates, rerank=None, rerank_rank=None):
        prov = {k: v for k, v in self.provenance.get(idx, {}).items() if v is not None}
        doc_name = self.source_citations.get(idx, "Unknown Source")
        return RetrievedEvidence(
            claim_id=claim.id, source_document=doc_name, evidence_text=self.chunks[idx],
            relevance_score=float(rerank if rerank is not None else lexical),
            ranking_method='cross_encoder' if rerank is not None else 'lexical',
            retrieval_score=float(lexical), rerank_score=None if rerank is None else float(rerank),
            lexical_rank=lexical_rank, rerank_rank=rerank_rank, candidate_count=candidates,
            evidence_id=f"EVD-{idx:05d}", source_citation=doc_name, page_number=self.source_pages.get(idx), **prov)

    def retrieve(self, claim: ClaimCitationPair, top_k: int | None = None, candidate_k: int = 30) -> list[RetrievedEvidence]:
        if self._tfidf_matrix is None or self._vectorizer is None or not self.chunks:
            raise RuntimeError(
                "Source document not loaded; retrieval index not initialized. Call add_sources() before retrieve()."
            )
        k = top_k if top_k is not None else self.top_k
        query = CITATION.sub('', claim.text).strip() or claim.text
        if claim.citation_marker:
            query = query.replace(claim.citation_marker, '').strip() or query
        similarities = self._lexical(query)
        order = similarities.argsort()[::-1]
        lexical_rank = {int(idx): r for r, idx in enumerate(order, 1)}
        candidates = [int(i) for i in order[:min(candidate_k, len(self.chunks))]]
        if not candidates:
            return []
        selected: list[tuple[int, float | None, int | None]] = []
        if self.use_reranker and self.reranker is not None:
            self.stats['rerank_calls'] += 1
            self.stats['rerank_pairs'] += len(candidates)
            try:
                scores = [float(s) for s in self.reranker.predict([[query, self.chunks[i]] for i in candidates])]
                ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                selected = [(i, s, r) for r, (i, s) in enumerate(ranked, 1)]
            except Exception as e:  # noqa: BLE001 - fall back to lexical ranking, recorded per passage
                logger.warning("Cross-encoder failed for claim %s: %s. Falling back to lexical scores.", claim.id, e)
                self.stats['rerank_failures'] += 1
                selected = [(i, None, None) for i in candidates if similarities[i] > 0]
        else:
            selected = [(i, None, None) for i in candidates if similarities[i] > 0]
        results, kept = [], []
        for idx, score, rank in selected:
            norm = _norm(self.chunks[idx])
            if any(_jaccard(norm, other) >= .8 for other in kept):
                continue  # near-duplicate of a better-ranked passage (e.g. overlapping sentence windows)
            kept.append(norm)
            results.append(self._evidence(claim, idx, similarities[idx], lexical_rank[idx], len(candidates), score, rank))
            if len(results) >= k:
                break
        return results
