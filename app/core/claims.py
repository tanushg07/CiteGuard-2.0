"""Atomic-proposition decomposition and proposition-level citation association. Pure Python, no third-party imports.

Decomposition (split_clauses)
- A sentence is split only at a coordinating joint (';', ', and', ', but', ', while', ', whereas', ', yet', or a
  citation marker directly followed by 'and'/'while'/'whereas'/'but') AND only when the right-hand side is an
  independent clause: it has its own subject followed by a finite verb. Punctuation alone never splits.
- Never split when the split would change meaning: a sentence-initial subordinate clause ('Although ...',
  'If ...', 'Because ...') scopes over the whole sentence; 'respectively' binds the parts; a right-hand side that
  opens with a back-reference ('this', 'they', 'thus', 'which') cannot stand alone; causal, temporal and
  conditional connectives ('because', 'so that', 'after', 'if') are not joints, so those relations stay intact.
- Clause text is a verbatim slice of the manuscript sentence. The proposition stored for verification is that
  slice with citation markers removed and the sentence completed ('...laboratory [1, 2]' -> '...laboratory.').
- Coordinated predicates with ONE shared subject ('LLMs can search the literature, propose ideas and plan
  experiments') stay one proposition - splitting would require inventing text. predicate_components() derives
  subject+predicate sub-hypotheses from verbatim slices; they are used only to test partial support.

Association (associate)
- direct           : the marker sits inside the proposition it is paired with (or a one-marker, one-proposition sentence)
- sentence_level   : one proposition with several citation markers; each cited work may support all or part of it
- multi_proposition: a sentence-final marker in a multi-proposition sentence whose other propositions have no
                     citation of their own - it may cover several propositions (flagged for review)
- ambiguous_scope  : a proposition without its own citation, paired with the nearest marker (flagged for review)
A trailing citation is never assumed to support every proposition in a sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r'\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]|\([A-Za-z][A-Za-z\s.\-]*(?:et al\.)?,\s*\d{4}[a-z]?\)')

# Qualifiers/conditions scope over the whole sentence; splitting would detach them from half the claim.
QUALIFIER_START = re.compile(r'^(?:if|when|whenever|unless|although|though|while|whereas|because|since|given|assuming|'
                             r'in (?:contrast|general|particular|most cases)|for example|for instance|however|'
                             r'in the case of)\b', re.I)
# A right-hand clause that opens with a back-reference or a consequence marker cannot stand alone.
ANAPHORIC = frozenset('this that these those it they such he she its their which whose thus therefore hence so then '
                      'consequently accordingly'.split())
JOINT = re.compile(r'(?P<c>;\s+)|,\s+(?P<j>and|but|while|whereas|yet)\s+|'
                   r'(?<=\])\s+(?P<m>and|but|while|whereas)\s+', re.I)
RELATION = {'and': 'conjunction', 'but': 'contrast', 'yet': 'contrast', 'while': 'contrast_or_simultaneity',
            'whereas': 'contrast', ';': 'semicolon'}
MIN_WORDS = 6
AUX = frozenset('is are was were be been being has have had do does did can could may might must shall should will '
                'would cannot'.split())
DETERMINERS = frozenset('the a an this that these those its their our his her each every some many several no any all '
                        'both such other another'.split())
PREPOSITIONS = frozenset('of in on at by for with from to into onto over under between among through via across '
                         'about against during without within upon toward towards per than as'.split())
# Base forms of verbs common in scientific prose (inflections are generated). Used only to recognise a finite verb.
VERB_BASES = ('accelerate achieve affect allow analyse analyze apply argue assess assume attend automate avoid '
              'become begin build calculate carry cause challenge change check combine compare compute confirm '
              'consider consist contain contribute control correlate create decrease define demonstrate depend '
              'derive describe design detect determine develop differ discover display drive emerge enable '
              'encode enhance ensure establish estimate evaluate examine exhibit expand explain explore expose '
              'extend facilitate fail find focus follow form generate give govern guide help highlight hold '
              'identify illustrate implement improve include increase indicate induce influence inform '
              'introduce investigate involve lead learn leave limit link make manipulate measure model '
              'motivate need observe obtain occur offer operate optimize optimise outperform perform permit '
              'plan play predict prefer prevent produce propose prove provide raise reach read reduce reflect '
              'remain report represent require reveal run search select serve show solve study suffer suggest '
              'support surpass take tend test track train transform treat undergo use vary verify work yield '
              'automate assist adopt address align employ engage exploit fit gain grow increase lack lower '
              'mitigate monitor rely remove replace resolve respond rise scale share shift simulate span '
              'stabilize stem summarize target trigger underlie underpin capture characterize characterise '
              'quantify revise criticize criticise question argue cast apply').split()
IRREGULAR = frozenset('found made took gave led held ran grew rose knew saw shown shows drawn drew began begun '
                      'became brought built thought sought wrote written taught kept left lost met paid put set '
                      'said told understood won'.split())


def _verb_forms():
    forms = set()
    for base in VERB_BASES:
        forms.add(base)
        forms.add(base + 'es' if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')) else
                  base[:-1] + 'ies' if base.endswith('y') and base[-2] not in 'aeiou' else base + 's')
        if base.endswith('e'):
            forms.add(base + 'd')
        elif base.endswith('y') and base[-2] not in 'aeiou':
            forms.add(base[:-1] + 'ied')
        else:
            forms.add(base + 'ed')
            if re.fullmatch(r'[^aeiou]*[aeiou][bcdfgklmnprstvz]', base):
                forms.add(base + base[-1] + 'ed')
    return frozenset(forms | IRREGULAR)


VERB_FORMS = _verb_forms()
VERB_BASE_SET = frozenset(VERB_BASES)
MODALS = frozenset('can could may might must shall should will would cannot do does did'.split())
PERFECT = frozenset('have has had'.split())


def _is_participle(tok: str) -> bool:
    return tok in IRREGULAR or tok == 'been' or (tok.endswith('ed') and tok in VERB_FORMS)


@dataclass(frozen=True)
class Clause:
    index: int
    start: int          # offsets into the original sentence
    end: int
    text: str           # verbatim slice, citation markers retained
    split_method: str   # 'none' | 'semicolon' | 'coordinating_conjunction' | 'citation_boundary'
    proposition: str = ''
    relation: str = 'none'
    rationale: str = ''
    uncertain: bool = False


def _words(text: str) -> int:
    return len(CITATION.sub('', text).split())


def _tokens(text: str) -> list[str]:
    return [re.sub(r'[^\w\-]', '', t).lower() for t in CITATION.sub(' ', text).split() if re.sub(r'[^\w\-]', '', t)]


def finite_verb(text: str, window: int = 9) -> tuple[int, str] | None:
    """Index and kind ('lexicon'|'aux'|'guess') of the first finite verb that follows a subject, else None."""
    toks = _tokens(text)[:window]
    for i, tok in enumerate(toks):
        if i == 0:
            continue  # position 0 is the subject's head at best
        prev = toks[i - 1]
        if prev in DETERMINERS or prev in PREPOSITIONS or prev == 'to':
            continue  # 'the results', 'of use', 'to select' are not finite verbs
        if tok in AUX:
            return i, 'aux'
        if tok in VERB_FORMS:
            return i, 'lexicon'
        if re.fullmatch(r'[a-z]{3,}ed', tok) and prev not in AUX:
            return i, 'guess'
    return None


DISCOURSE = frozenset('however moreover furthermore additionally instead meanwhile nevertheless nonetheless conversely '
                      'similarly likewise notably importantly'.split())


def independent_clause(text: str) -> tuple[bool, str, bool]:
    """(is_independent, rationale, uncertain)."""
    toks = _tokens(text)
    while toks and toks[0] in DISCOURSE:
        # 'however, protection waned ...': a discourse adverb does not change who the subject is
        toks = toks[1:]
        text = re.sub(r'^\W*\w+\W*', '', text, count=1)
    if not toks:
        return False, 'empty', False
    first = toks[0]
    if first in ANAPHORIC:
        return False, f'right-hand side opens with "{first}", which refers back to the left-hand side', False
    # A right side opening with a base/past verb form ('and tested them', 'and improving results') continues the
    # left-hand predicate. '-s' forms are allowed because they are usually plural nouns ('results show ...').
    opens_with_verb = first in VERB_BASE_SET or first in IRREGULAR or (first in VERB_FORMS and first.endswith('ed'))
    if first.endswith('ing') or first in PREPOSITIONS or first == 'to' or first in AUX or opens_with_verb:
        return False, f'right-hand side opens with "{first}", continuing the left-hand predicate (shared subject)', False
    verb = finite_verb(text)
    if not verb:
        return False, 'no finite verb with its own subject was found on the right-hand side', False
    index, kind = verb
    subject = ' '.join(toks[:index])
    return True, f'right-hand side has its own subject ("{subject}") and finite verb ("{toks[index]}")', kind == 'guess'


def _proposition(text: str) -> str:
    clean = CITATION.sub('', text)
    clean = re.sub(r'\s+([,.;:])', r'\1', ' '.join(clean.split())).strip(' ,;')
    if not clean:
        return ''
    clean = clean[0].upper() + clean[1:]
    return clean if clean.endswith(('.', '?', '!')) else clean + '.'


def split_clauses(sentence: str) -> list[Clause]:
    text = sentence.strip()
    whole = [Clause(0, 0, len(sentence), text, 'none', _proposition(text), 'none',
                    'Single proposition: no coordinating joint with an independent right-hand clause.',
                    _words(text) > 60)]
    if QUALIFIER_START.match(text):
        return [Clause(0, 0, len(sentence), text, 'none', _proposition(text), 'none',
                       'Not split: a sentence-initial subordinate clause qualifies the whole sentence.', _words(text) > 60)]
    if re.search(r'\brespectively\b', text, re.I):
        return [Clause(0, 0, len(sentence), text, 'none', _proposition(text), 'none',
                       'Not split: "respectively" binds the coordinated parts together.', False)]
    cuts, rejected = [], []
    for m in JOINT.finditer(sentence):
        word = (m.group('j') or m.group('m') or ';').lower()
        method = 'semicolon' if m.group('c') else 'citation_boundary' if m.group('m') else 'coordinating_conjunction'
        left = sentence[cuts[-1][1] if cuts else 0:m.start()]
        right_full = sentence[m.end():]
        nxt = JOINT.search(sentence, m.end())
        right = sentence[m.end():nxt.start()] if nxt else right_full
        if _words(left) < MIN_WORDS or _words(right) < MIN_WORDS:
            rejected.append(f'"{word}" joint not split: a side has fewer than {MIN_WORDS} words')
            continue
        left_verb = finite_verb(left, window=60)
        subject_list = bool(re.search(r',[^,]+\band\b', CITATION.sub('', left)))
        if not left_verb or (left_verb[1] == 'guess' and subject_list):
            # 'Independent verification, precommitted predictions and acceptance criteria, and experimental
            # controls provide ...': the left side is a list of subjects sharing the right-hand predicate.
            rejected.append(f'"{word}" joint not split: the left-hand side has no finite verb of its own '
                            '(coordinated subjects share one predicate)')
            continue
        ok, why, uncertain = independent_clause(right)
        if not ok:
            rejected.append(f'"{word}" joint not split: {why}')
            continue
        cuts.append((m.start(), m.end(), method, RELATION.get(word, 'conjunction'), why, uncertain))
    if not cuts:
        if rejected:
            return [Clause(0, 0, len(sentence), text, 'none', _proposition(text), 'none',
                           'Single proposition. ' + '; '.join(rejected) + '.', _words(text) > 60)]
        return whole
    out, prev, method, relation, why, uncertain = [], 0, 'none', 'none', 'first proposition of the sentence', False
    for s, e, m, rel, reason, unc in cuts:
        out.append((prev, s, method, relation, why, uncertain))
        prev, method, relation, why, uncertain = e, m, rel, reason, unc
    out.append((prev, len(sentence), method, relation, why, uncertain))
    clauses = []
    for i, (a, b, m, rel, reason, unc) in enumerate(out):
        verbatim = sentence[a:b].strip()
        rationale = (f'Proposition {i + 1} of {len(out)}: ' +
                     ('first clause before a coordinating joint.' if i == 0 else f'split at a {m.replace("_", " ")} ({rel}); {reason}.'))
        clauses.append(Clause(i, a, b, verbatim, m, _proposition(verbatim), rel, rationale, unc))
    return clauses


def predicate_components(proposition: str) -> list[str]:
    """Sub-hypotheses for coordinated predicates sharing one subject (used only to test partial support).

    'X can search the literature, propose ideas and analyse data.' ->
    ['X can search the literature.', 'X can propose ideas.', 'X can analyse data.']
    Returns [] when the proposition is not a list of >= 2 predicates headed by verbs.
    """
    body = proposition.rstrip('.!?')
    words = body.split()
    toks = [re.sub(r'[^\w\-]', '', w).lower() for w in words]
    # Prefer an auxiliary/modal head ('can', 'have'); a bare lexicon verb is the fallback.
    head = next((i for i, tok in enumerate(toks[1:12], 1) if tok in AUX and toks[i - 1] not in DETERMINERS), None)
    if head is None:
        head = next((i for i, tok in enumerate(toks[1:12], 1) if tok in VERB_FORMS
                     and toks[i - 1] not in DETERMINERS | PREPOSITIONS), None)
    if head is None:
        return []
    j = head + 1
    while j < len(toks) and toks[j] in AUX | {'also', 'not', 'often', 'already', 'successfully', 'been', 'be'}:
        j += 1
    subject_and_aux = ' '.join(words[:head + 1] if toks[head] in AUX else words[:head])
    adverbs = ' '.join(w for w, t in zip(words[head + 1:j], toks[head + 1:j]) if t in AUX | {'been', 'be'})
    predicate = ' '.join(words[j:] if toks[head] in AUX else words[head:])
    parts = [p.strip() for p in re.split(r',\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+', predicate) if p.strip()]
    if len(parts) < 2 or len(parts) > 6:
        return []
    firsts = [re.sub(r'[^\w\-]', '', p.split()[0]).lower() for p in parts]
    if not all(f in VERB_FORMS for f in firsts):
        return []
    # Every coordinated predicate must agree with the head ('have generated ... and tested', 'can search ... and
    # plan'). This rejects coordination inside an infinitive: 'have used feedback to select and carry out ...'.
    h = toks[head]
    if h in PERFECT and not all(_is_participle(f) for f in firsts):
        return []
    if h in MODALS and not all(f in VERB_BASE_SET for f in firsts):
        return []
    if h not in AUX:
        kind = 'ed' if firsts[0].endswith('ed') else 's' if firsts[0].endswith('s') else 'base'
        if not all((f.endswith('ed') if kind == 'ed' else f.endswith('s') if kind == 's' else f in VERB_BASE_SET)
                   for f in firsts):
            return []
    if re.search(r'\bto\s*$', parts[0]):
        return []
    prefix = ' '.join(filter(None, [subject_and_aux, adverbs]))
    return [f'{prefix} {p}.' for p in parts]


@dataclass(frozen=True)
class Association:
    clause_index: int
    marker: str
    position: int
    association: str    # direct | sentence_level | multi_proposition | ambiguous_scope
    rationale: str
    ambiguity: str = 'unambiguous'   # unambiguous | shared | ambiguous
    review: bool = False


AMBIGUOUS_TYPES = frozenset({'ambiguous_scope', 'multi_proposition'})


def associate(sentence: str, clauses: list[Clause], located_markers: list[tuple[str, int]]) -> list[Association]:
    """located_markers: (marker, char offset in sentence). Returns one Association per (proposition, marker)."""
    if len(clauses) == 1:
        if len(located_markers) == 1:
            m, p = located_markers[0]
            return [Association(0, m, p, 'direct', 'The only citation in a single-proposition sentence.')]
        return [Association(0, m, p, 'sentence_level',
                            f'One proposition cites {len(located_markers)} works; each is checked independently and '
                            'may support all or part of it.', 'shared')
                for m, p in located_markers]

    def clause_of(pos: int) -> int:
        for c in clauses:
            if pos < c.end + 1:
                return c.index
        return clauses[-1].index

    own: dict[int, list[tuple[str, int]]] = {c.index: [] for c in clauses}
    for marker, pos in located_markers:
        own[clause_of(pos)].append((marker, pos))
    marked = [i for i, v in own.items() if v]
    unmarked = [c.index for c in clauses if not own[c.index]]
    last = clauses[-1].index
    sentence_final = bool(own[last]) and all(i == last for i in marked) and bool(unmarked)
    out: list[Association] = []
    for c in clauses:
        for marker, pos in own[c.index]:
            if sentence_final:
                out.append(Association(c.index, marker, pos, 'multi_proposition',
                                       f'Marker closes the sentence; propositions {", ".join(str(i + 1) for i in unmarked)} '
                                       'have no citation of their own, so it may cover several propositions.',
                                       'ambiguous', True))
            else:
                out.append(Association(c.index, marker, pos, 'direct',
                                       f'Marker appears inside proposition {c.index + 1} of {len(clauses)}.'))
    for c in clauses:
        if own[c.index] or not marked:
            continue
        following = [i for i in marked if i > c.index]
        donor = following[0] if following else marked[-1]
        for marker, pos in own[donor]:
            out.append(Association(c.index, marker, pos, 'ambiguous_scope',
                                   f'Proposition {c.index + 1} has no citation of its own; the nearest marker sits in '
                                   f'proposition {donor + 1}. It is not known whether it was intended to cover this one.',
                                   'ambiguous', True))
    return sorted(out, key=lambda a: (a.clause_index, a.position))


def validate_page(page, total_pages: int | None) -> tuple[int | None, str]:
    """Never invent a page: missing or impossible values become None with an explicit status."""
    if page is None or isinstance(page, bool):
        return None, 'missing'
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None, 'invalid'
    if page < 1 or (total_pages is not None and total_pages > 0 and page > total_pages):
        return None, 'out_of_range'
    return page, 'ok'


__all__ = ['CITATION', 'Clause', 'Association', 'AMBIGUOUS_TYPES', 'split_clauses', 'associate', 'predicate_components',
           'independent_clause', 'finite_verb', 'validate_page', 'field']
