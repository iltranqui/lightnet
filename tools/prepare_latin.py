#!/usr/bin/env python3
"""Prepare pinned Latin snapshots for lightnet LM; never runs model training.

Usage: python3 tools/prepare_latin.py [--root datasets/latin]
Generated outputs are immutable: use a new --out for another preparation.
"""
import argparse
import collections
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

LANG = '{http://www.w3.org/XML/1998/namespace}lang'
SKIP = {'note', 'rdg', 'rdgGrp', 'teiHeader', 'fw', 'speaker', 'stage',
        'del', 'head', 'figDesc', 'figure', 'ref', 'bibl', 'label'}
BLOCK = {'p', 'l', 'ab', 'lb', 'pb', 'div', 'sp'}
FOLD = str.maketrans({'æ': 'ae', 'œ': 'oe', 'v': 'u', 'j': 'i'})


def local(tag):
    return tag.rsplit('}', 1)[-1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n'))
    lines = [' '.join(line.split()) for line in text.splitlines()]
    return '\n'.join(line for line in lines if line) + '\n'


def folded_words(text):
    return re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', text.lower().translate(FOLD)))


def audit_key(text, folded=False):
    text = text.lower()
    if folded:
        text = text.translate(FOLD)
    return ' '.join(re.findall(r'[^\W\d_]+', unicodedata.normalize('NFC', text)))


def shingles(text, folded=False):
    words = audit_key(text, folded).split()
    # 128-bit fingerprints bound memory while retaining negligible collision risk.
    return {hashlib.blake2b(' '.join(words[i:i+5]).encode(), digest_size=16).digest()
            for i in range(max(0, len(words) - 4))}


def work_id(edition):
    name = edition.removeprefix('urn:cts:').removeprefix('latinLit:').removeprefix('greekLit:')
    return 'urn:cts:latinLit:' + '.'.join(name.split('.')[:2])


def split_for(author_id, author):
    if author_id == 'phi1017' or 'seneca' in author.lower():
        return 'validation'
    if author_id == 'phi1351' or 'tacitus' in author.lower():
        return 'test'
    bucket = int(digest(('latin-pilot-v1:' + author_id).encode())[:8], 16) % 100
    return 'validation' if bucket < 5 else 'test' if bucket < 10 else 'train'


def extract_tei(path, annotated=False):
    metadata = {'author': '', 'title': '', 'licences': [], 'links': []}
    if annotated:
        passages = []
        # Clear completed passages: large annotation files need not stay in RAM.
        for _, elem in ET.iterparse(path, events=('end',)):
            tag = local(elem.tag)
            if tag in ('author', 'title') and not metadata[tag]:
                metadata[tag] = ' '.join(''.join(elem.itertext()).split())
            if tag in ('licence', 'license'):
                metadata['licences'].append(elem.get('target', '') + ' ' + ''.join(elem.itertext()))
            if tag == 'link':
                metadata['links'].append(elem.get('target', ''))
            if tag == 'ab':
                tokens = [''.join(w.itertext()).strip() for w in elem.iter()
                          if local(w.tag) in ('w', 'pc')]
                tokens = [t for t in tokens if t and not (t.startswith('{') and t.endswith('}'))]
                passage = ' '.join(tokens)
                passage = re.sub(r'\s+([,.;:!?\)\]])', r'\1', passage)
                passage = re.sub(r'([\(\[])\s+', r'\1', passage)
                passages.append(passage)
                elem.clear()
        return normalize('\n'.join(passages)), metadata

    root = ET.parse(path).getroot()
    for elem in root.iter():
        tag = local(elem.tag)
        if tag in ('author', 'title') and not metadata[tag]:
            metadata[tag] = ' '.join(''.join(elem.itertext()).split())
        if tag in ('licence', 'license'):
            metadata['licences'].append(elem.get('target', '') + ' ' + ''.join(elem.itertext()))
    body = next((e for e in root.iter() if local(e.tag) == 'body'), None)
    if body is None:
        raise ValueError('missing TEI body')

    def surface(elem):
        tag = local(elem.tag)
        language = elem.get(LANG, 'lat').lower()
        if tag in SKIP or language not in ('lat', 'la', 'latin'):
            return ''
        if tag == 'choice':
            for preferred in ('orig', 'sic', 'abbr', 'reg', 'corr', 'expan'):
                child = next((e for e in elem if local(e.tag) == preferred), None)
                if child is not None:
                    return surface(child)
        result = re.sub(r'\s+', ' ', elem.text or '')
        for child in elem:
            result += surface(child) + re.sub(r'\s+', ' ', child.tail or '')
        if tag in BLOCK:
            result += '\n'
        return result

    text = re.sub(r' +([,.;:!?])', r'\1', surface(body))
    return normalize(text), metadata


def read_conllu(directory, source):
    groups = {}
    order = {'train': 0, 'dev': 1, 'test': 2}
    for path in sorted(directory.glob('*.conllu'), key=lambda p: order[p.stem.rsplit('-', 1)[-1]]):
        for block in path.read_text(encoding='utf-8').split('\n\n'):
            comments = dict(re.findall(r'^# ([\w_]+) = (.*)$', block, re.M))
            if not comments:
                continue
            text = comments.get('text')
            if not text:
                raise ValueError(f'{path}: sentence without surface text')
            if source == 'ud-llct':
                match = re.search(r"document_id='([^']+)'", comments.get('reference', ''))
                if not match:
                    raise ValueError(f'{path}: missing charter reference')
                key, split = 'llct:' + match[1], 'diagnostic_llct'
            else:
                sent_id = comments['sent_id']
                key = 'circse:' + sent_id.rsplit('-', 1)[0]
                split = 'validation' if 'Seneca' in sent_id else 'test' if 'Tacitus' in sent_id else 'diagnostic_circse'
            if key not in groups:
                groups[key] = {'id': key, 'work_id': key, 'split': split, 'source': source,
                               'license': 'CC-BY-SA-4.0', 'text': [], 'raw_paths': set(),
                               'author': 'Seneca' if 'Seneca' in key else 'Tacitus' if 'Tacitus' in key else 'LLCT charters',
                               'author_id': 'phi1017' if 'Seneca' in key else 'phi1351' if 'Tacitus' in key else key}
            span = re.search(r"span='(\d+)'", comments.get('reference', ''))
            position = int(span[1]) if span else int(comments['sent_id'].rsplit('-', 1)[-1])
            groups[key]['text'].append((position, text))
            groups[key]['raw_paths'].add(str(path))
    result = []
    for doc in groups.values():
        doc['text'] = normalize('\n'.join(t for _, t in sorted(doc['text'], key=lambda item: item[0])))
        doc['raw_paths'] = sorted(doc['raw_paths'])
        result.append(doc)
    return result


def source_licence(metadata, source):
    licences = ' '.join(metadata['licences']).lower()
    if any(x in licences for x in ('by-nc', 'noncommercial', 'non-commercial', 'all rights reserved')):
        return None
    for version in ('4.0', '3.0', '2.5'):
        if f'/by-sa/{version}' in licences:
            return f'CC-BY-SA-{version}'
        if f'/by/{version}' in licences:
            return f'CC-BY-{version}'
    if source == 'perseus' and not licences:
        return 'CC-BY-SA-4.0 (repository default; see raw component)'
    return None


def write_tsv(path, documents):
    with path.open('w', encoding='utf-8') as out:
        out.write('id\twork_id\tsplit\tsource\tlicense\tsha256\ttext_path\n')
        for doc in documents:
            fields = [str(doc[k]) for k in ('id', 'work_id', 'split', 'source', 'license', 'sha256', 'text_path')]
            if any('\t' in f or '\n' in f for f in fields):
                raise ValueError('invalid TSV metadata')
            out.write('\t'.join(fields) + '\n')


def prepare(root, out):
    if out.exists():
        raise ValueError(f'{out} exists; use a fresh --out to avoid mixing preparations')
    out.mkdir(parents=True)
    (out / 'text').mkdir()
    (out / 'sample').mkdir()
    excluded, documents, seen_works, seen_text = [], [], {}, {}
    revisions = {row.split('\t')[0]: row.split('\t')[2]
                 for row in (root / 'sources.tsv').read_text().splitlines()[1:]}
    raw_hashes = {}

    def reject(path, reason, **extra):
        excluded.append({'path': str(path), 'reason': reason, **extra})

    for source in ('perseus', 'lascivaroma'):
        base = root / 'raw' / source
        paths = sorted((base / 'data').rglob('*-lat*.xml')) if source == 'perseus' else sorted((base / 'lemmatized/xml').glob('*.xml'))
        for index, path in enumerate(paths):
            if index % 100 == 0:
                print(f'extract {source} {index}/{len(paths)}', flush=True)
            edition = path.stem.removeprefix('urn:cts:latinLit:').removeprefix('urn:cts:greekLit:')
            if not re.fullmatch(r'[^.]+\.[^.]+\..*-lat\d+', edition):
                reject(path, 'not a named Latin edition')
                continue
            work = work_id(edition)
            if work in seen_works:
                reject(path, 'alternate edition or annotated mirror', kept=seen_works[work])
                continue
            try:
                text, meta = extract_tei(path, source == 'lascivaroma')
            except (ET.ParseError, ValueError) as exc:
                reject(path, 'XML extraction error', detail=str(exc))
                continue
            licence = source_licence(meta, source)
            if not licence:
                reject(path, 'unresolved or restricted original-text licence', metadata=meta)
                continue
            if len(re.findall(r'[^\W\d_]+', text)) < 50:
                reject(path, 'fewer than 50 surface words')
                continue
            exact = digest(audit_key(text, True).encode())
            if exact in seen_text:
                reject(path, 'folded normalized exact duplicate', kept=seen_text[exact])
                continue
            author_id = edition.split('.')[0]
            doc = {'id': source + ':' + edition, 'work_id': work,
                   'author_id': author_id, 'author': meta['author'], 'title': meta['title'],
                   'split': split_for(author_id, meta['author']), 'source': source,
                   'license': licence, 'text': text, 'raw_paths': [str(path)],
                   'revision': revisions[source], 'metadata': meta}
            documents.append(doc)
            seen_works[work] = doc['id']
            seen_text[exact] = doc['id']

    for source in ('ud-llct', 'ud-circse'):
        documents.extend(read_conllu(root / 'raw' / source, source))
    # Prefer native literary text to an annotation overlay of the same held-out work.
    # CIRCSE remains a separate diagnostic; neither its published train split nor
    # its sentence boundaries are used to allocate LM train/validation/test.
    for doc in documents:
        if doc['source'] == 'ud-circse':
            doc['split'] = 'diagnostic_circse'

    # Build exact five-word shingle membership over all nontraining text, then
    # screen each training work. This audits short quotations as well as longer
    # reuse; substantial overlap quarantines the complete training work.
    heldout = [d for d in documents if d['split'] != 'train']
    audit, quarantine = [], set()
    for folded in (False, True):
        print(f'audit five-word shingles latin_fold={folded}', flush=True)
        owners = collections.defaultdict(set)
        heldout_sizes = []
        for i, doc in enumerate(heldout):
            unique = shingles(doc['text'], folded)
            heldout_sizes.append(len(unique))
            for shingle in unique:
                owners[shingle].add(i)
        for doc in documents:
            if doc['split'] != 'train':
                continue
            current = shingles(doc['text'], folded)
            counts = collections.Counter(i for s in current for i in owners.get(s, ()))
            for i, matches in counts.items():
                # Containment uses the smaller work: catches excerpts of long works.
                other_size = heldout_sizes[i]
                containment = matches / max(1, min(len(current), other_size)) if other_size else 0
                if matches >= 5:
                    audit.append({'train_id': doc['id'], 'heldout_id': heldout[i]['id'],
                                  'folded': folded, 'shared_5grams': matches,
                                  'containment': containment, 'action': 'quarantine' if matches >= 20 and containment >= .2 else 'low_overlap_reported'})
                if matches >= 20 and containment >= .2:
                    quarantine.add(doc['id'])
        del owners
    kept = []
    for doc in documents:
        if doc['id'] in quarantine:
            reject(doc['id'], 'substantial heldout five-word overlap')
            continue
        data = doc.pop('text').encode('utf-8')
        doc['sha256'] = digest(data)
        doc['bytes'] = len(data)
        doc['words'] = len(re.findall(r'[^\W\d_]+', data.decode()))
        doc['text_path'] = 'text/' + digest(doc['id'].encode())[:24] + '.txt'
        (out / doc['text_path']).write_bytes(data)
        doc['revision'] = revisions[doc['source']]
        doc['raw_sha256'] = {}
        for path in doc['raw_paths']:
            if path not in raw_hashes:
                raw_hashes[path] = file_hash(Path(path))
            doc['raw_sha256'][path] = raw_hashes[path]
        kept.append(doc)
    write_tsv(out / 'documents.tsv', kept)
    for split in sorted({d['split'] for d in kept}):
        write_tsv(out / f'{split}.tsv', [d for d in kept if d['split'] == split])
    # A reproducible, bounded sample. At most 4096 UTF-8 bytes per work,
    # round-robin by author, 512 KiB total; no heldout file is consulted.
    by_author = collections.defaultdict(list)
    for doc in sorted(kept, key=lambda d: digest(('sample-v1:' + d['id']).encode())):
        if doc['split'] == 'train':
            by_author[doc['author_id']].append(doc)
    sample, total, round_index = [], 0, 0
    while total < 512 * 1024:
        progress = False
        for author in sorted(by_author):
            docs = by_author[author]
            if round_index >= len(docs):
                continue
            parent = docs[round_index]
            cap = min(4096, 512 * 1024 - total)
            data = (out / parent['text_path']).read_bytes()[:cap].decode('utf-8', errors='ignore').encode()
            if len(data) < 128:
                continue
            item = {**parent, 'id': 'sample:' + parent['id'], 'parent_id': parent['id'],
                    'sha256': digest(data), 'bytes': len(data),
                    'text_path': 'sample/' + digest(parent['id'].encode())[:24] + '.txt'}
            (out / item['text_path']).write_bytes(data)
            sample.append(item)
            total += len(data)
            progress = True
        if not progress:
            break
        round_index += 1
    write_tsv(out / 'tokenizer-sample.tsv', sample)
    (out / 'tokenizer-sample.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False) + '\n')
    with (out / 'provenance.jsonl').open('w', encoding='utf-8') as stream:
        for doc in kept:
            stream.write(json.dumps(doc, ensure_ascii=False) + '\n')
    (out / 'excluded.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + '\n')
    (out / 'overlap-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    report = {'version': 1, 'normalization': 'NFC, canonical line endings, whitespace; latin_fold=false; macrons retained',
              'source_revisions': revisions, 'extractor_sha256': file_hash(Path(__file__)),
              'split_policy': 'Seneca validation; Tacitus test; other CTS author IDs sha256 latin-pilot-v1 modulo 100: <5 validation, <10 test, else train; diagnostics separate',
              'sample_policy': 'training only; round-robin authors, SHA256 sample-v1 work order; prefix <=4096 UTF-8 bytes/work; <=512KiB total',
              'sample_bytes': total, 'sample_documents': len(sample),
              'excluded_reasons': dict(collections.Counter(x['reason'] for x in excluded)),
              'near_duplicate_rule': 'quarantine train work if >=20 shared five-word shingles and >=20% containment of smaller work, original or folded view',
              'overlap_pairs_reported': len(audit), 'train_works_quarantined': len(quarantine),
              'splits': {}}
    for split in sorted({d['split'] for d in kept}):
        group = [d for d in kept if d['split'] == split]
        report['splits'][split] = {'documents': len(group), 'bytes': sum(d['bytes'] for d in group),
                                   'words': sum(d['words'] for d in group),
                                   'authors': len({d['author_id'] for d in group})}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'PREPARATION_COMPLETE').write_text('latin-preparation-v1\n')
    # Pin preparation outputs, excluding this self-referential checksum file.
    with (out / 'SHA256SUMS').open('w') as stream:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'SHA256SUMS':
                stream.write(file_hash(path) + '  ' + str(path.relative_to(out)) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/latin'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.out or args.root / 'prepared')
