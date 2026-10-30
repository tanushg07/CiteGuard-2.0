"""Whole-document representation; layout heuristics retain all pages and raw text."""
import os
import re

import pdfplumber

REFERENCE = re.compile(r'^(?:\d+[. ]+)?(?:references|bibliography|works cited|literature cited)\s*$', re.I)
SECTION = re.compile(r'^(?:(?i:section\s+\d+[:.]?.*)|[1-9]\d?(?:\.\d+)*\.?\s+[A-Z][A-Za-z -]{3,80}|(?i:abstract|introduction|related work|methodology|methods|experiments|results|discussion|conclusions?|limitations|acknowledg(?:e)?ments|appendi(?:x|ces)(?:\s+.*)?))$')
ARXIV = re.compile(r'(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/|\babs/)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)', re.I)
DOI = re.compile(r'10\.\d{4,9}/[^\s<>"\]]+', re.I)


MARKER = re.compile(r'^\s*(?:\[\s*\d+\s*\]|\d+[.)])\s*')
YEAR = re.compile(r'(?<![\d/.])(?:19|20)\d{2}(?![\d/])')
NAME_SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv'}


def normalize_reference_text(raw):
    """Undo PDF line-wrapping damage that breaks identifiers and titles. Never changes bibliographic content.

    - 'Cap- pellaro'                -> 'Cappellaro'          (hyphenation at a line break, lowercase continuation)
    - 'doi: 10.1038/ d41586-026-1'  -> 'doi: 10.1038/d41586-026-1'
    - 'https: //openreview.net'     -> 'https://openreview.net'
    """
    text = ' '.join(str(raw).split())
    text = re.sub(r'(?<=[a-z])- (?=[a-z])', '', text)
    text = re.sub(r'(10\.\d{4,9}/)\s+(?=\S)', r'\1', text)
    text = re.sub(r'\b(https?):\s*//\s*', r'\1://', text)
    text = re.sub(r'(?<=\d)\s*[–—−]\s*(?=\d)', '–', text)  # '2600– 2606' -> '2600–2606'
    return text


def _sentence_boundaries(text):
    """Offsets of '. ' boundaries that are not after a single-letter initial (e.g. 'Ross D. King')."""
    out = []
    for m in re.finditer(r'[.?!]\s+(?=[A-Z0-9"“‘(\[]|arXiv|doi:)', text):
        before = re.search(r'([\w-]+)$', text[:m.start()])
        word = before[1] if before else ''
        if text[m.start()] == '.' and (re.fullmatch(r'[A-Z]|[A-Z]-[A-Z]', word) or word.lower() in NAME_SUFFIX
                                       or word.lower() in {'vol', 'no', 'pp', 'eds', 'ed', 'proc', 'conf', 'int', 'j'}):
            continue
        out.append((m.start(), m.end()))
    return out


def _looks_like_authors(segment):
    words = re.findall(r"[^\s,]+", segment)
    if not words:
        return False
    if re.search(r'\bet al\b', segment) or ',' in segment or re.search(r'\band\b', segment):
        capitalised = sum(w[:1].isupper() or w.lower() in ('and', 'et', 'al.', 'al', 'van', 'von', 'de', 'der', 'la', 'di')
                          for w in words)
        return capitalised / len(words) >= .8
    return len(words) <= 4 and all(w[:1].isupper() for w in words)


def parse_authors(segment):
    segment = re.sub(r',?\s*et al\.?$', '', segment.strip().rstrip('.'))
    names = [n.strip() for n in re.split(r',\s*(?:and\s+)?|\s+and\s+|\s*&\s*', segment) if n.strip()]
    authors, surnames = [], []
    for name in names:
        tokens = [t for t in name.replace('.', '. ').split() if t.strip('.').lower() not in NAME_SUFFIX]
        if not tokens:
            continue
        authors.append(name)
        # 'Smith, J.' (surname-first) vs 'John Smith' / 'J. Smith' (surname-last)
        surname = tokens[0] if len(tokens) == 1 or re.fullmatch(r'[A-Z]\.?', tokens[-1].strip()) else tokens[-1]
        surnames.append(surname.strip('.,'))
    return authors, surnames


def reference_record(raw, index, page=None):
    """Best-effort structured fields of one bibliography entry. Unknown fields stay null; nothing is invented."""
    text = normalize_reference_text(raw)
    doi = DOI.search(text)
    arxiv = ARXIV.search(text)
    urls = [u.rstrip('.,;') for u in re.findall(r'https?://[^\s<>]+', text)]
    number = re.match(r'^\s*(\[\s*\d+\s*\]|\d+[.)])', text)
    body = MARKER.sub('', text)
    boundaries = _sentence_boundaries(body)
    segments, prev = [], 0
    for start, end in boundaries:
        segments.append((prev, body[prev:start + 1].strip()))
        prev = end
    segments.append((prev, body[prev:].strip()))
    authors, surnames, title, rest_offset = [], [], None, 0
    if segments and _looks_like_authors(segments[0][1]) and len(segments) > 1:
        authors, surnames = parse_authors(segments[0][1])
        title_index = 1
    else:
        title_index = 0
    title_variants = []
    if len(segments) > title_index:
        title = segments[title_index][1].strip().rstrip('.').strip() or None
        rest_offset = segments[title_index + 1][0] if len(segments) > title_index + 1 else len(body)
        # A title may itself contain '?' ('Can LLMs ...? A large-scale study ...'); keep the merged reading as an
        # alternative so identification can score both instead of guessing which one is right.
        if title and title.endswith(('?', '!')) and len(segments) > title_index + 1:
            title_variants.append((title + ' ' + segments[title_index + 1][1]).strip().rstrip('.'))
    rest = body[rest_offset:]
    masked = re.sub(r'https?://\S+|10\.\d{4,9}/\S+|arxiv:\S+', ' ', rest, flags=re.I)
    years = YEAR.findall(masked) or YEAR.findall(re.sub(r'https?://\S+|10\.\d{4,9}/\S+', ' ', body))
    venue = None
    venue_part = re.sub(r'^\s*In\s+(?:[^,]*,\s*editors?,\s*)?', '', rest)
    venue_part = re.split(r'\.\s+(?:doi|URL|arXiv|Originally)\b|\bdoi:|\s+URL\s', venue_part, maxsplit=1, flags=re.I)[0]
    vm = re.match(r'\s*([^,\d][^,]*?)(?:,|\s+\d+\s*\(|\.?$)', venue_part)
    if vm and not re.fullmatch(r'\s*(?:arxiv|preprint|unpublished manuscript|url).*', vm[1], re.I):
        venue = re.sub(r'\s+(?:19|20)\d{2}$', '', vm[1].strip(' .')) or None
    vol = re.search(r'\b(\d{1,5})\s*\((\d{1,6})\)\s*:\s*([A-Za-z]?\d+(?:–\d+)?)', masked)
    pages = vol[3] if vol else (re.search(r'\bpages?\s+(\d+(?:–\d+)?)', masked) or [None, None])[1]
    if not vol:
        vol = re.search(r'\b(\d{1,5})\s*:\s*([A-Za-z]?\d{3,}(?:–\d+)?)', masked)
        pages = pages or (vol[2] if vol else None)
    return {'reference_id': f'REF-{index:03d}', 'citation_key': f'[{re.search(r"\d+", number[0])[0]}]' if number else None,
            'raw_reference': raw, 'normalized_reference': text, 'page': page, 'title': title,
            'title_variants': title_variants,
            'authors': authors, 'author_surnames': surnames,
            'year': int(years[0]) if years else None, 'doi': doi[0].rstrip('.,;') if doi else None,
            'urls': urls, 'arxiv_id': arxiv[1] if arxiv else None,
            'venue': venue, 'publisher': None, 'volume': vol[1] if vol else None,
            'issue': vol[2] if vol and vol.re.groups == 3 else None, 'pages': pages,
            'parsing_method': 'bibliography_heuristic'}


def _pdfium_pages(path):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        pages = []
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            page.close()
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            # pdfium marks a line-break hyphen with U+FFFE (or \x02); the word continues after it.
            pages.append(re.sub(r'[\ufffe\x02]\n?', '', text))
        return pages
    finally:
        pdf.close()


# Page furniture that must never become part of a manuscript sentence or a source passage.
NOISE_LINE = re.compile(r'^(?:arXiv:\d{4}\.\d{4,5}(?:v\d+)?\s+\[[\w.\-]+\].*|.*\bviXra\b.*|'
                        r'[∗†‡§¶*]\s*(?:Corresponding authors?|E-?mail|These authors|Equal contribution|Present address).*|'
                        r'(?:Corresponding authors?|E-?mail):\s*\S+@\S+.*|\S+@\S+\.[a-z]{2,}|'
                        r'(?:Downloaded from|This article is protected by copyright|Electronic copy available at).*)$', re.I)


def starts_reference(line, references):
    """Does this bibliography line begin a new entry?

    The numbering style of the list is fixed by its first entry. In a bracket-numbered list only '[n]' starts an
    entry, and in a dot-numbered list only the *next expected* number does - so a wrapped line that begins with a
    year ('2004. doi: 10.1038/...') is a continuation, not reference number 2004.
    """
    if not references:
        return True
    first = references[0]
    if re.match(r'^\s*\[\s*\d+\s*\]', first):
        return bool(re.match(r'^\s*\[\s*\d+\s*\]', line))
    numbered = re.match(r'^\s*(\d+)[.)]\s', first)
    if numbered:
        current = re.match(r'^\s*(\d+)[.)]\s', line)
        return bool(current) and int(current[1]) == len(references) + int(numbered[1])
    return bool(re.match(r'^[A-Z][\w\'’-]+,\s+(?:[A-Z]\.|[A-Z][a-z]+)', line))


class DocumentParser:
    def __init__(self, file_path=None, ocr=True, ocr_recogniser=None):
        self.file_path = file_path
        self.extracted_references = []
        self.ocr = ocr
        self.ocr_recogniser = ocr_recogniser  # injectable (name, fn) for tests

    def _structure(self, pages, name, metadata=None):
        from app.services.extractor import Extractor
        paragraphs, references, reference_pages, sections = [], [], [], []
        in_references, section = False, 'Front matter'
        splitter = Extractor()
        for page_number, text in enumerate(pages, 1):
            if in_references and not re.search(r'^\s*(?:\[\d+\]|\d+[.)]\s)', text, re.M) and re.search(r'^Figure \d', text, re.M):
                in_references, section = False, 'Post-reference figures / appendix'
                sections.append({'title': section, 'page': page_number})
            block = []
            state = {'first_on_page': True}
            def flush():
                joined = ''
                for part in block:
                    # 'inter-' + 'vention' at a line break -> 'intervention' (lowercase continuation only)
                    if joined.endswith('-') and re.search(r'[a-z]-$', joined) and part[:1].islower():
                        joined = joined[:-1] + part
                    else:
                        joined = (joined + ' ' + part) if joined else part
                clean = ' '.join(joined.split())
                block.clear()
                if len(clean) <= 15:
                    return
                previous = paragraphs[-1] if paragraphs else None
                # A paragraph that starts lowercase at the top of a page continues the previous page's paragraph.
                if (state['first_on_page'] and previous and clean[:1].islower() and previous['section'] == section
                        and not previous['text'].rstrip().endswith(('.', '?', '!', ':'))):
                    previous['text'] += ' ' + clean
                    previous['page_end'] = page_number
                else:
                    paragraphs.append({'text': clean, 'page_number': page_number,
                                       'paragraph': len(paragraphs) + 1, 'section': section})
                state['first_on_page'] = False
            lines = [line.strip() for line in text.splitlines()]
            lengths = sorted(len(line) for line in lines if len(line) > 20)
            typical = lengths[len(lengths) // 2] if lengths else 0
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if NOISE_LINE.match(line):
                    continue  # arXiv stamp, corresponding-author footnote, download banner
                if REFERENCE.fullmatch(line):
                    flush()
                    in_references, section = True, 'References'
                    sections.append({'title': section, 'page': page_number})
                elif in_references and re.match(r'^(?:appendi(?:x|ces)\b|[A-Z]\s+[A-Z][a-z])', line, re.I if line.lower().startswith('append') else 0):
                    in_references = False
                    section = line
                    sections.append({'title': section, 'page': page_number})
                elif in_references:
                    if not line or re.fullmatch(r'\d+', line):
                        continue
                    if starts_reference(line, references):
                        references.append(line)
                        reference_pages.append(page_number)
                    else:
                        # Join wrapped lines; a line-break hyphen before a lowercase continuation is removed.
                        if references[-1].endswith('-') and line[:1].islower() and re.search(r'[a-z]-$', references[-1]):
                            references[-1] = references[-1][:-1] + line
                        else:
                            references[-1] += ' ' + line
                elif section.lower() == 'abstract' and re.match(r'^[∗†‡]|^\d+(?:st|nd|rd|th) Conference', line):
                    flush()
                    section = 'Front matter notes'
                    block.append(line)
                elif SECTION.fullmatch(line) and len(line) < 100 and not line.endswith('.'):
                    flush()
                    section = line
                    sections.append({'title': section, 'page': page_number})
                elif not line:
                    flush()
                elif not re.fullmatch(r'\d+', line):
                    block.append(line)
                    # Layout cue: a clearly short line ending a sentence closes its paragraph.
                    if typical and len(line) < .7 * typical and line.endswith(('.', '?', '!')) and len(block) > 1:
                        flush()
            flush()
        sentences = []
        for para in paragraphs:
            for sentence in splitter._split_sentences(para['text']):
                sentences.append({'text': sentence, 'sentence_index': len(sentences)+1,
                                  'page': para['page_number'], 'section': para['section'], 'paragraph': para['paragraph']})
        # Whitespace/line-wrap repair only (split DOIs, 'https: //', 'Cap- pellaro'); bibliographic content is unchanged.
        references = [normalize_reference_text(r) for r in references]
        self.extracted_references = references
        md = metadata or {}
        first_lines = [line.strip() for line in (pages[0] if pages else '').splitlines() if line.strip()]
        title = md.get('Title') if isinstance(md.get('Title'), str) else None
        title = title or (first_lines[0] if first_lines else name)
        author = md.get('Author') if isinstance(md.get('Author'), str) else None
        front = '\n'.join(first_lines[:30])
        title_pos = next((i for i,line in enumerate(first_lines) if title in line), -1)
        abstract_pos = next((i for i,line in enumerate(first_lines) if line.lower() == 'abstract'), min(len(first_lines), 15))
        author_lines = first_lines[title_pos+1:abstract_pos] if title_pos >= 0 else []
        authors = [name.strip() for line in author_lines if '@' not in line and not re.search(r'university|institute|google|research|department|brain',line,re.I) for name in re.split(r'[∗†‡]|\s{2,}',line) if len(name.strip().split()) >= 2]
        affiliations = [line for line in first_lines[:30] if re.search(r'\b(university|institute|laboratory|research|google|department)\b', line, re.I)]
        return {'document_name': name, 'metadata': {'title': title, 'authors': [author] if author else authors,
                    'affiliations': affiliations, 'abstract': ' '.join(p['text'] for p in paragraphs if p['section'].lower() == 'abstract') or None,
                    'keywords': md.get('Keywords') if isinstance(md.get('Keywords'), str) else None,
                    'publication_information': None, 'front_matter': front, 'method': 'PDF metadata and heading heuristics'},
                'pages': [{'page': i+1, 'text': t} for i,t in enumerate(pages)],
                'sections': sections, 'paragraphs': paragraphs, 'sentences': sentences,
                'reference_records': [reference_record(r,i+1,reference_pages[i]) for i,r in enumerate(references)],
                'references': references, 'total_pages': len(pages)}

    def _ocr(self, path, pages):
        """Image-only PDF: OCR fallback. Raises RuntimeError with an explicit cause when OCR cannot run."""
        if not self.ocr:
            raise RuntimeError('OCR_DISABLED: no extractable text in this PDF and OCR fallback is disabled.')
        from app.services.ocr import OCRUnavailable, ocr_pdf
        from backend.config import get_settings
        settings = get_settings()
        if settings.ocr_mode == 'off':
            raise RuntimeError('OCR_DISABLED: no extractable text in this PDF and OCR_MODE=off.')
        try:
            name, ocr_pages = ocr_pdf(path, settings.ocr_max_pages, self.ocr_recogniser)
        except OCRUnavailable as exc:
            raise RuntimeError(f'OCR_UNAVAILABLE: this PDF has no extractable text (likely scanned). {exc}') from exc
        if not any(p.strip() for p in ocr_pages):
            raise RuntimeError('OCR_NO_TEXT: OCR ran but recognised no text in this PDF.')
        return ocr_pages, f'ocr:{name}'

    def parse_text(self, raw_text, source_name='Pasted Text'):
        return self._structure([raw_text], source_name)

    def parse(self, file_path=None):
        path = file_path or self.file_path
        if not path:
            raise RuntimeError('No file path provided for DocumentParser.')
        try:
            engine = 'pdfium'
            try:
                # pdfium follows the content stream (column order) and is ~50x faster than pdfplumber on long PDFs;
                # pdfplumber's extract_text interleaves the two columns of two-column layouts line by line.
                pages = _pdfium_pages(path)
            except Exception:  # noqa: BLE001 - fall back to pdfplumber below
                pages, engine = None, 'pdfplumber'
            with pdfplumber.open(path) as pdf:
                if pages is None or len(pages) != len(pdf.pages):
                    pages, engine = [page.extract_text(x_tolerance=1) or '' for page in pdf.pages], 'pdfplumber'
                metadata = dict(pdf.metadata) if isinstance(pdf.metadata, dict) else {}
                first = pdf.pages[0]
                if isinstance(first.chars, list) and first.chars:
                    sizes = [c['size'] for c in first.chars if c.get('upright', True) and c.get('text','').isalpha()]
                    if sizes:
                        largest = max(sizes)
                        title_chars = [c for c in first.chars if c.get('size',0) >= largest-.5]
                        if title_chars:
                            box = (min(c['x0'] for c in title_chars), min(c['top'] for c in title_chars),
                                   max(c['x1'] for c in title_chars), max(c['bottom'] for c in title_chars))
                            title_text = first.crop(box).extract_text(x_tolerance=1)
                            if title_text and len(title_text) > 8:
                                metadata['Title'] = title_text.replace('\n',' ')
            method = f'pdf_text:{engine}'
            # Fewer than 5 words per page on average: an image-only (scanned) PDF, possibly with a stray text layer.
            if sum(len(text.split()) for text in pages) < 5 * max(1, len(pages)):
                pages, method = self._ocr(path, pages)
            document = self._structure(pages, os.path.basename(path), metadata)
            document['extraction'] = {'method': method, 'pages': len(pages),
                                      'words': sum(len(p.split()) for p in pages)}
            return document
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f'Failed to parse PDF {path}: {exc}') from exc
