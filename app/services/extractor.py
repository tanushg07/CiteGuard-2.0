import re
from typing import Any

from app.core.claims import associate, predicate_components, split_clauses, validate_page
from app.core.schemas import ClaimCitationPair


class Extractor:
    """
    Extracts citation-bearing claims from document paragraphs and pairs them with
    their full paragraph context and citation markers.

    Key fix: deduplicates claims at the sentence level — a sentence is extracted
    at most ONCE even if it contains multiple citation patterns.
    """

    def __init__(self):
        # Bracketed numbers: [1], [1, 2], [1-3], [12, 14, 18]
        self.bracket_pattern = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
        # Author-year: (Smith, 2020), (Smith et al., 2020), (Alon and Yahav, 2021)
        self.author_year_pattern = re.compile(r"\([A-Za-z\s]+(?:et al\.)?,\s*\d{4}[a-z]?\)")
        # Inline: Vaswani et al. (2017)
        self.inline_author_year = re.compile(r"\b[A-Z][a-zA-Z\-]+(?:\s+(?:et al\.|and [A-Z][a-zA-Z]+))?\s*\(\d{4}[a-z]?\)")

    def _split_sentences(self, text: str) -> list[str]:
        """Splits paragraph into sentences using period/question/exclamation boundaries."""
        protected = text
        replacements = [
            ("et al.", "ET_AL_TOKEN"),
            ("i.e.", "I_E_TOKEN"),
            ("e.g.", "E_G_TOKEN"),
            ("Fig.", "FIG_TOKEN"),
            ("Tab.", "TAB_TOKEN"),
            ("Ref.", "REF_TOKEN"),
            ("vs.", "VS_TOKEN"),
            ("approx.", "APPROX_TOKEN"),
            ("al.", "AL_TOKEN"),
        ]
        for orig, rep in replacements:
            protected = protected.replace(orig, rep)

        raw_sentences = re.split(r"(?:(?<=[.!?])\s+(?=[A-Z0-9\"\'(\[])|(?<=[.!?])\n+)", protected)

        sentences = []
        for s in raw_sentences:
            s_clean = s
            for orig, rep in replacements:
                s_clean = s_clean.replace(rep, orig)
            s_clean = s_clean.strip()
            if len(s_clean) > 10:
                sentences.append(s_clean)

        return sentences

    def _has_citation(self, sentence: str) -> bool:
        """Returns True if the sentence contains at least one citation pattern."""
        return bool(
            self.bracket_pattern.search(sentence)
            or self.author_year_pattern.search(sentence)
            or self.inline_author_year.search(sentence)
        )

    def _primary_marker(self, sentence: str) -> str:
        """Returns the first citation marker found in the sentence."""
        for pat in [self.bracket_pattern, self.author_year_pattern, self.inline_author_year]:
            m = pat.search(sentence)
            if m:
                return m.group(0)
        return ""

    def extract(self, paragraphs: list[dict[str, Any]], references=None, enricher=None, enrich=True,
                total_pages: int | None = None, manuscript_id: str = '') -> list[ClaimCitationPair]:
        """
        Scans paragraphs for claims that contain academic citations.
        Each cited SENTENCE is extracted exactly ONCE (deduplicated), paired with
        its primary citation marker and paragraph context.
        """
        pairs: list[ClaimCitationPair] = []
        seen_sentences: set[str] = set()

        sentence_index = 0
        for para in paragraphs:
            if para.get("section", "").lower().strip() in {"references", "bibliography", "works cited"}:
                continue

            para_text = para["text"]
            # Never default to page 1: an unmapped or impossible page stays unknown and is flagged.
            page_number, page_status = validate_page(para.get("page_number"), total_pages)
            section = para.get("section", "Body")

            sentences = self._split_sentences(para_text)

            for sentence in sentences:
                sentence_index += 1
                if not self._has_citation(sentence):
                    continue

                located: dict[str, int] = {}
                for pattern in (self.bracket_pattern, self.author_year_pattern, self.inline_author_year):
                    for match in pattern.finditer(sentence):
                        for marker in self._expand_marker(match.group(0)):
                            located.setdefault(marker, match.start())
                clauses = split_clauses(sentence)
                sentence_id = f'CLM-{sentence_index:04d}'
                for assoc in associate(sentence, clauses, list(located.items())):
                    clause = clauses[assoc.clause_index]
                    claim_text = clause.proposition or clause.text
                    key = (claim_text, assoc.marker, page_number)
                    if key in seen_sentences:
                        continue
                    seen_sentences.add(key)
                    pairs.append(ClaimCitationPair(id=f'claim_{len(pairs)+1:04d}', text=claim_text,
                        context=para_text, citation_marker=assoc.marker, page_number=page_number, section=section,
                        page_mapping=page_status, manuscript_id=manuscript_id,
                        claim_group_id=sentence_id, sentence_id=sentence_id, sentence_text=sentence,
                        atomic_claim_id=f'{sentence_id}.{assoc.clause_index + 1}',
                        citation_id='CIT-' + re.sub(r'[^A-Za-z0-9]', '', assoc.marker),
                        sentence_index=sentence_index, paragraph=para.get('paragraph', 0),
                        association_confidence=assoc.association, association_type=assoc.association,
                        ambiguity_status=assoc.ambiguity, association_rationale=assoc.rationale,
                        clause_text=clause.text, decomposition_method=clause.split_method,
                        clause_relation=clause.relation, decomposition_rationale=clause.rationale,
                        decomposition_uncertain=clause.uncertain,
                        predicate_components=predicate_components(claim_text),
                        claim_type=self._claim_type(sentence)))
        from app.services.reference_metadata import CompositeEnricher, reference_for_marker
        for pair in pairs:
            entry = reference_for_marker(pair.citation_marker, references or [])
            if entry:
                pair.resolved_reference_id = f'REF-{references.index(entry) + 1:03d}'
        if not enrich:
            return pairs
        enricher = enricher or CompositeEnricher()
        for pair in pairs:
            raw = reference_for_marker(pair.citation_marker, references or [])
            pair.reference_metadata = enricher.enrich(raw)
        return pairs

    @staticmethod
    def _expand_marker(marker: str) -> list[str]:
        if not marker.startswith('['):
            return [marker]
        out = []
        for part in marker[1:-1].split(','):
            nums = re.findall(r'\d+', part)
            if len(nums) == 2:
                out.extend(f'[{n}]' for n in range(int(nums[0]), min(int(nums[1]), int(nums[0]) + 100) + 1))
            elif nums:
                out.append(f'[{nums[0]}]')
        return out

    @staticmethod
    def _claim_type(sentence: str) -> str:
        navigation = r'^(?:see|refer to|for (?:details|an overview)|we (?:thank|refer)|in (?:the )?(?:following|next) sections?\b|(?:in this (?:paper|section),? )?we will (?:describe|discuss|present|review))\b'
        if re.match(navigation, sentence, re.I):
            return 'not_verifiable'
        # Flattened tables are retained for inspection, but are unsafe NLI hypotheses.
        without_citations = re.sub(r'\[[\d,\s–-]+\]', '', sentence)
        words = without_citations.split()
        numeric_tokens = sum(bool(re.fullmatch(r'[\d.,%+−-]+', word)) for word in words)
        if numeric_tokens >= 10 and numeric_tokens / max(len(words), 1) > 0.15:
            return 'not_verifiable'
        return 'assertion_candidate'
