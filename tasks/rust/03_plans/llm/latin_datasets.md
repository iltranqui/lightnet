# Latin text datasets for the Latin pilot

Compiled with gpt-6.1-sol (reasoning effort high) from fetched pages, unreviewed.

Candidate corpora, treebanks and eval sets for the 30-60M parameter Latin pilot (Task 10 in `tasks/rust/03_plans/llm/plan_llm_decoder.md`). Facts come from pages fetched on 2026-10-23; `[SEARCH]` marks search-summary-only evidence and `UNVERIFIED` marks anything not stated on a fetched page. Ranking, split, mix and cleaning advice is judgement, not measurement. No corpus was downloaded.

## Table of Contents

- [1. Deduped source table](#1-deduped-source-table)
- [2. Overlap map](#2-overlap-map)
- [3. Pilot ranking](#3-pilot-ranking)
- [4. Train/validation/test split](#4-trainvalidationtest-split)
- [5. Licence-risk table](#5-licence-risk-table)
- [6. Cleaning by source](#6-cleaning-by-source)
- [7. Recommended mix](#7-recommended-mix)
- [8. Evaluation sets and use](#8-evaluation-sets-and-use)
- [9. Sources or claims not verified](#9-sources-or-claims-not-verified)

## 1. Deduped source table

Evidence date: **2026-10-23**. **[FETCHED]** means a fetched page was read; **[SEARCH]** means search-summary evidence only. Sizes retain their original units; downloads, words, tokens and documents are not interchangeable. **UNVERIFIED** means the notes do not establish the fact.

| Name / URL | Content | Size with unit | Format | Licence / terms | Status |
|---|---|---|---|---|---|
| [Perseus canonical-latinLit](https://github.com/PerseusDL/canonical-latinLit) | Latin literature; period/genre UNVERIFIED | UNVERIFIED | TEI-style XML | Default CC BY-SA 4.0 unless otherwise indicated; “offer Perseus any modifications”; component copyright varies | [FETCHED]; work in progress; macrons/u-v/i-j/OCR UNVERIFIED |
| [Perseus Hopper downloads](https://www.perseus.tufts.edu/hopper/opensource/download) / [CLTK Perseus](https://github.com/cltk/lat_text_perseus) | Perseus distributions; CLTK includes English translations, extent UNVERIFIED | Hopper: all texts 447 MB; Greek/Roman 125 MB; Latin share UNVERIFIED. CLTK: UNVERIFIED | XML; CLTK author folders and JSON tooling | Stated Creative Commons ShareAlike 3.0 | [FETCHED]; overlapping distributions, not additive |
| [Latin Library](https://www.thelatinlibrary.com/) / [terms](https://www.thelatinlibrary.com/about.html) / [CLTK mirror](https://github.com/cltk/lat_text_latin_library) | Latin from Alcuin to Virgil; CLTK says archaic–modern; no translations | CLTK about 108 MB; site size UNVERIFIED | HTML; mirror `.txt` | Site: personal/educational use and precautions toward public-domain status. Mirror README: public domain; referenced licence file unseen | [FETCHED]; scanner artifacts/typos acknowledged; macrons/u-v/i-j UNVERIFIED |
| [Corpus Corporum: secondary description](https://wiki.digitalclassicist.org/Corpus_Corporum) | Latin repository | [SEARCH] about 226 million words/30 corpora; about 215M Latin words in Feb 2025; HistCorp subset 5,271 texts/~80M tokens. All UNVERIFIED by fetch | Standardised TEI XML; some fails TEI validation | Noncommercial download unless provider/copyright restrictions apply; wiki CC-BY does **not** license texts | [FETCHED] wiki only; direct repository UNVERIFIED |
| [Fece228 latin-literature-dataset-170M](https://huggingface.co/datasets/Fece228/latin-literature-dataset-170M) | Corpus Corporum derivative; raw and CLTK-preprocessed parts; raw includes other languages/references/contractions | Total files 6.15 GB; category `100M<n<1B`, unit UNVERIFIED | UNVERIFIED | Not stated; inherited permissions UNVERIFIED | [FETCHED]; one `train` split; raw/preprocessed are alternative representations |
| [CC-100 Latin](https://data.statmt.org/cc-100/) | Latin file; provenance from Common Crawl UNVERIFIED on fetched page | `la.txt.xz`: **609M compressed**, unit UNVERIFIED | `.txt.xz`; sentence-per-line UNVERIFIED | No IP claims over corpus preparation; no formal licence named | [FETCHED]; Wikipedia-like content/web noise UNVERIFIED |
| [Fece228 cc100-latin](https://huggingface.co/datasets/Fece228/cc100-latin) | Claimed CC-100 derivative removing lorem ipsum and duplicate sentences | [SEARCH] about 609M → 390M **tokens**, UNVERIFIED | UNVERIFIED | UNVERIFIED | [SEARCH]; HTTP 401; distinct from CC-100’s compressed `609M` |
| [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | Latin web documents | 48,968 documents; 44,176,580 tokens | Parquet | Follows mC4/OSCAR terms; harmful-use restriction stated | [FETCHED]; gated, conditions/login required |
| [mC4](https://huggingface.co/datasets/allenai/c4) | Latin subset | Viewer approximately 1.68M rows; Latin byte/token size UNVERIFIED | Gzipped JSON; streaming | ODC-BY plus Common Crawl terms | [FETCHED]; public |
| [OSCAR 22.01](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201) | Latin from Nov/Dec 2021 Common Crawl | 4.1 MB; 4,397 documents; 187,446 words | UNVERIFIED | CC0 metadata/annotations only; underlying-text permission UNVERIFIED | [FETCHED]; gated, access temporarily suspended |
| [OSCAR 23.01](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | Latin web records | 2.0 MB; 14,384 documents; 307,865 words | JSON-style records, Zstandard | CC0 annotations/packaging only; underlying text not owned by dataset authors | [FETCHED]; gated, access temporarily suspended |
| [Vicipaedia statistics](https://la.wikipedia.org/wiki/Specialis:Statistics) / [dump](https://dumps.wikimedia.org/lawiki/latest/) | Modern/neo-Latin encyclopaedic text | 142,580 content pages; 22,633,483 content words. Dump `103877453`, unit UNVERIFIED | MediaWiki XML bz2 | UNVERIFIED on fetched pages | [FETCHED]; dump dated 01-Oct-2026; macrons/u-v/i-j UNVERIFIED |
| [OGL First1KGreek](https://github.com/OpenGreekAndLatin/First1KGreek) | Greek-focused XML; Latin content UNVERIFIED | UNVERIFIED | XML | CC BY-SA 4.0 for this repository | [FETCHED]; other OGL Latin repositories UNVERIFIED; OGL/Perseus relationship [SEARCH] |
| [Patrologia Latina overview](https://en.wikipedia.org/wiki/Patrologia_Latina) | Tertullian–Innocent III; digital scans and machine-readable alternatives listed | 217 volumes, 1841–1855; digital text size UNVERIFIED | Scans; Zurich machine-readable repository; OGL XML | PL copyright status UNVERIFIED | [FETCHED] overview; reprint errors stated; scan OCR quality UNVERIFIED |
| [Corpus Thomisticum / Index Thomisticus](https://www.corpusthomisticum.org/) | Aquinas study resource | UNVERIFIED | Download/format UNVERIFIED | “All rights reserved”; permission UNVERIFIED | [FETCHED] homepage; Index page yielded no content |
| [UD Latin](https://universaldependencies.org/la/index.html): PROIEL | Mostly Vulgate NT; Caesar, Cicero, Palladius selections | UNVERIFIED | CoNLL-U | CC BY-NC-SA 3.0 | [FETCHED] index/repository |
| UD Perseus / [AGLDT mirror](https://github.com/cltk/latin_treebank_perseus) | Latin literary treebanks | UD: 2,273 trees, 29,138 words/tokens. LDT 1.5: ~53,143 words, 8 texts | CoNLL-U / XML | CC BY-NC-SA 2.5 | [FETCHED]; overlapping versions, not additive |
| UD ITTB | Aquinas treebank; source corpus includes 61 related authors | Changelog v2.7: 450,515 annotated tokens, possibly outdated | CoNLL-U | CC BY-NC-SA 3.0 | [FETCHED] |
| UD UDante | Dante letters, treatises, poetry | 1,721 sentences; 57,242 syntactic words | CoNLL-U | CC BY-NC-SA 3.0 | [FETCHED] |
| UD LLCT | Tuscan charters, AD 774–897 | 521 charters; 242,411 tokens: train 194,143/dev 24,189/test 24,079 | CoNLL-U | CC BY-SA 4.0 | [FETCHED]; 223 sentences omitted versus LLCT2 |
| UD CIRCSE | Four Seneca plays and Tacitus’s *Germania* | ~29,054 tokens, computed in notes | CoNLL-U | CC BY-SA 4.0 in README metadata | [FETCHED] |
| [EvaLatin](https://aclanthology.org/2024.lt4hala-1.21/) | 2024 parsing/emotion tasks; earlier morphology tasks [SEARCH] | UNVERIFIED | Data format/download UNVERIFIED | Dataset licence UNVERIFIED; paper CC BY 4.0 only | [FETCHED] paper overview |
| [LiLa Lemma Bank](https://zenodo.org/records/8300851) | Lemma lexicon | ZIP 15.1 MB; [SEARCH] 215,102 lemmas, UNVERIFIED | Fetched format UNVERIFIED; [SEARCH] RDF Turtle/SQL | Record: “Other (Open)”, description CC-BY-SA; exact 4.0 version [SEARCH], UNVERIFIED | [FETCHED]; GitHub 404 |
| [LASLA–LiLa linking record](https://zenodo.org/records/5961378) | Linking triples; not necessarily corpus text | ZIP 108.1 MB; [SEARCH] LASLA text portion ~1.7M tokens, UNVERIFIED | Turtle; [SEARCH] LASLA DAT/APN/BPN, UNVERIFIED | Record conflicts: CC BY 4.0 versus CC BY-NC-SA 4.0; corpus terms UNVERIFIED | [FETCHED] triples; actual corpus [SEARCH] only |
| [lascivaroma](https://github.com/lascivaroma/latin-lemmatized-texts) | Classical/late Latin; automatic Pie-Extended lemmatisation | 21,327,783 tokens; 17,885,059 excluding punctuation | TEI-like XML, word annotations | Repo CC BY 4.0; annotations CC BY-SA 4.0; texts retain original licences | [FETCHED]; original-source identities UNVERIFIED; suspected Perseus/OGL overlap UNVERIFIED |
| [PHI Latin](https://latin.packhum.org/) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | HTTP 403; download availability UNVERIFIED |
| [Digital Latin Library](https://digitallatin.org/) | Critical editions from all eras | UNVERIFIED | Text inventory/download UNVERIFIED | UNVERIFIED | [FETCHED] homepage only |
| Vulgate: Clementine / PROIEL | Clementine 1598 plaintext [SEARCH]; PROIEL NT above | UNVERIFIED | Plaintext [SEARCH] | Clementine “public domain” [SEARCH], UNVERIFIED; other editions’ rights UNVERIFIED | Clementine page not fetched |
| [Latin BERT reference](https://arxiv.org/abs/2009.10053) | Classical–21st-century corpus | Roughly 642.7 million words | Corpus format/availability UNVERIFIED | Corpus licence UNVERIFIED | [FETCHED] abstract; source breakdown and ~50% IA balance [SEARCH] |
| [Internet Archive Latin query](https://archive.org/advancedsearch.php?q=language%3Alat&rows=0&output=json) | All media with `language:lat` | 140,098 items; not a text count | Text/OCR availability UNVERIFIED per item | Per-item licences vary; UNVERIFIED | [FETCHED] query only; OCR quality UNVERIFIED |
| Project Gutenberg Latin | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | DNS failed; listing UNVERIFIED |
| [Tatoeba](https://tatoeba.org/en/downloads) | Sentence downloads | Latin count UNVERIFIED | UNVERIFIED | Main files CC BY 2.0 FR; some sentences CC0 1.0 | [FETCHED]; OPUS Latin-English sizes UNVERIFIED, pages 404 |
| Musisque Deoque / TLL | Poetry / other content UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Musisque connection refused; TLL not fetched |
| LiMe / LatinCy / Camena etc. | LiMe: [SEARCH] 325 Milanese criminal documents; LatinCy: [SEARCH] pipelines, not corpus; Camena unresearched | LiMe text size UNVERIFIED | UNVERIFIED | UNVERIFIED | [SEARCH] or unresearched |

## 2. Overlap map

| Family | Relationship and evidence |
|---|---|
| Perseus → Hopper / CLTK Perseus | Same source family [FETCHED]; do not sum as independent text |
| Latin Library → CLTK Latin Library | Mirror relationship [FETCHED] |
| Corpus Corporum → Fece228 literature | Explicit derivative [FETCHED]; raw and preprocessed parts also duplicate underlying works |
| CC-100 → Fece228 cc100-latin | Claimed derivative [SEARCH]; UNVERIFIED |
| Perseus ↔ Latin Library / lascivaroma | Cross-site overlap requires audit; lascivaroma source identities UNVERIFIED |
| Wikipedia ↔ CC-100 / mC4 / CulturaX / OSCAR | Overlap identified in notes; exact documents/extent UNVERIFIED |
| Perseus / Vulgate / Aquinas ↔ treebanks | Annotation overlays on underlying texts [FETCHED]; AGLDT and UD Perseus also overlap |
| Latin BERT ↔ IA / PL / Thomisticum / Latin Library / Perseus / Wikipedia | Source breakdown [SEARCH]; exact membership UNVERIFIED |
| PL ↔ Zurich / OGL / IA scans | Alternative digital manifestations listed [FETCHED]; exact edition/text overlap UNVERIFIED |

**Judgement:** deduplicate at underlying work/edition level across repositories, then audit near duplicates with 5-word shingles.

## 3. Pilot ranking

**Judgement — ordered by quality visibility, usable scale, rights clarity and extraction effort:**

| Rank | Source | Reason |
|---|---|---|
| 1 | Perseus canonical-latinLit | Structured XML and explicit default licence; size/quality still UNVERIFIED, with component exceptions requiring review |
| 2 | lascivaroma | Substantial stated token count and annotated structure; automatic annotations, source-rights checks and overlap removal add work |
| 3 | Latin Library/CLTK | Convenient ~108 MB plaintext mirror; acknowledged errors and conflicting rights signals weaken licence safety |
| 4 | Vicipaedia | Stated word count and downloadable dump; useful register diversity, but licence UNVERIFIED here |
| 5 | Corpus Corporum/Fece228 | Broad volume candidate; noncommercial/provider restrictions, derivative rights and preprocessing need resolution |
| 6 | mC4/CulturaX | Useful web supplement; mC4 has stated terms, while CulturaX adds gating/inherited conditions; cleaning burden UNVERIFIED |
| 7 | CC-100/OSCAR/IA | Rights uncertainty, access limitations or unverified text quality make them later-stage candidates |

**Judgement:** reserve treebanks for evaluation initially. Defer sources with missing text access or unresolved permissions. No source is established here as unconditionally cleared for commercial use.

## 4. Train/validation/test split

**Proposed rules — judgement:**

- Allocate approximately **90%/5%/5% of eligible, deduplicated text**, keeping every work intact and every known author in one split.
- Group alternate editions, mirrors, excerpts, annotated versions and near duplicates before assigning splits.
- Example validation group: **all Seneca**, including *Hercules Furens*, *Agamemnon*, *Oedipus*, *Phoenissae*. Example test group: **all Tacitus**, including *Germania*. Exclude their corresponding texts from every training source.
- For anonymous works, group by work and linked textual witnesses; do not claim author independence where authorship is UNVERIFIED. Keep web/article evaluation separate when author grouping cannot be established.
- Ignore inherited dataset splits when they violate these rules; Fece228’s single `train` split is not an evaluation design.
- Fit the **8,192/16,384 BPE vocabulary on training text only**. Apply the chosen u/v and i/j policy consistently, preserving macrons. Create 512/1,024-token windows only after splitting.
- Audit cross-split 5-word shingles, including a folded comparison view; adjudicate shared boilerplate and genuine textual reuse before freezing splits.

## 5. Licence-risk table

**Risk classifications below are judgement from the stated terms, not verified legal clearance.**

| Category | Sources | Commercial / noncommercial implication |
|---|---|---|
| Explicit share-alike | Perseus canonical; LLCT; CIRCSE; First1KGreek | No NC label in the cited licence names; component rights and actual Latin availability still matter |
| Explicit noncommercial + share-alike | PROIEL, UD Perseus/AGLDT, ITTB, UDante | Restrict to an appropriately cleared noncommercial track |
| Noncommercial/provider restrictions | Corpus Corporum | Commercial use unsupported by these notes; provider exceptions also affect noncommercial use |
| Layered licences | lascivaroma; CulturaX; mC4 | Repository/annotation/database terms do not settle every underlying text’s permissions |
| Attribution or partial CC0 | Tatoeba | Track sentence-level licence provenance |
| CC0 covers metadata only | OSCAR | Underlying-text clearance remains UNVERIFIED |
| Conflicting/ambiguous | Latin Library/CLTK; LASLA linking record; LiLa exact version | Resolve discrepancies before inclusion |
| Restrictive notice | Corpus Thomisticum | Permission UNVERIFIED; exclude pending permission |
| No sufficient text licence established | Fece228 derivatives, CC-100, Wikipedia, PL, IA items, EvaLatin data, DLL, PHI, Gutenberg, LiMe, other unverified sources | Neither commercial nor noncommercial reuse established by these notes |

## 6. Cleaning by source

**Proposed actions — judgement; inspect before assuming any defect exists.**

| Source | Cleaning actions |
|---|---|
| Perseus/Hopper/CLTK | Extract Latin text from XML; separate notes, apparatus and translations; exclude English folders; reconcile duplicate versions |
| Latin Library/CLTK | Remove HTML/navigation; inspect acknowledged scanner artifacts/typos; retain work/author boundaries |
| Corpus Corporum/Fece228 | Tolerate and log invalid TEI; choose one raw/preprocessed representation; inspect references, contractions and Greek/Hebrew passages |
| lascivaroma | Reconstruct surface text; remove annotation attributes; resolve duplicated enclitics marked with `{}`; preserve provenance |
| Wikipedia | Extract article prose from MediaWiki XML; remove markup, navigation and template debris; audit crawl duplicates |
| CC-100/Fece228 cc100/mC4/CulturaX/OSCAR | Inspect language quality, boilerplate, lorem ipsum and duplicates; retain provenance; verify sentence/document boundaries |
| PL/IA | Separate editions and scan-derived copies; inspect OCR and reprint errors; exclude unusable passages |
| UD/AGLDT | Reconstruct text without annotation columns; retain work IDs; prevent underlying-text leakage |
| LiLa/LASLA triples | Keep lexical/linking data separate from prose; do not train on serialized triples as ordinary Latin |
| Tatoeba | Filter Latin sentences; retain licence identifiers; deduplicate sentences |
| EvaLatin/DLL/Thomisticum/PHI/Gutenberg/others | Establish access, text format and permissions before designing extraction |

**Judgement, all sources:** preserve raw originals; retain macrons during Unicode handling; record folding decisions, extraction versions and exclusions. Avoid silently “correcting” historical spelling.

## 7. Recommended mix

**Judgement — approximate training-token proportions after cleaning and deduplication, not measured availability:**

- **80%** eligible Perseus canonical text.
- **20%** eligible, nonoverlapping lascivaroma text with original-source permissions checked.

**Judgement:** if the second portion cannot be cleared or adds little unique text, use **100% eligible Perseus** for the first pilot. Treat this as a pipeline pilot; available unique volume is UNVERIFIED. Keep treebanks out initially, and introduce broader/web sources only in separately tracked experiments after rights and quality checks.

## 8. Evaluation sets and use

**Proposed uses — judgement:**

| Evaluation set | Use / limit |
|---|---|
| Frozen author/work-held-out Latin | Primary LM loss; compare vocabulary sizes, context lengths and folding choices under identical splits |
| CIRCSE Seneca/Tacitus groups | Named literary holdouts; include only authors absent from training |
| LLCT | Legal/charter domain diagnostic; its published split is not proof of author/work independence |
| PROIEL, UD Perseus, ITTB, UDante | Morphology/syntax diagnostics under NC-SA terms; remove underlying-text leakage |
| EvaLatin | Parsing/emotion benchmark candidate; earlier morphology tasks [SEARCH]; access/licence UNVERIFIED |
| LiLa Lemma Bank | Lexical coverage and lemma diagnostics, not continuous-text perplexity |
| Tatoeba | Optional sentence-level diagnostic after Latin selection/licence tracking; count UNVERIFIED |

**Judgement:** report results separately by evaluation domain and licence track. Parsing or emotion scores require a defined task evaluation procedure; they are not supplied by LM loss alone.

## 9. Sources or claims not verified

- **Failed/inaccessible:** Fece228 cc100-latin (401), PHI (403), Gutenberg (DNS), direct Corpus Corporum site (DNS), Musisque Deoque (connection refused), CLTK documentation/OGL homepage/LiLa GitHub/OPUS pages (404), Index Thomisticus page (no content).
- **[SEARCH] only:** Fece228 cc100 cleaning/token counts; Corpus Corporum scale and HistCorp subset; Clementine plaintext/public-domain claim; LASLA corpus scale/formats/terms; LiLa lemma count/formats/exact licence version; Latin BERT source breakdown/balance; earlier EvaLatin tasks; LiMe; LatinCy; Perseus Zenodo releases and OGL/Perseus relationship.
- **Fetched overview, underlying resource UNVERIFIED:** OGL Latin repositories, PL digital texts/rights, IA text availability/licences/OCR quality, DLL editions/downloads/licences, EvaLatin data access/licence, Latin BERT corpus availability/licence.
- **Other unresolved:** Wikipedia licence; Fece228 inherited permissions/formats; Latin Library licence discrepancy; LASLA licence conflict; Tatoeba Latin count; unlisted sizes and orthographic properties remain **UNVERIFIED**.
- **Not researched/fetched:** TLL, Camena and other neo-Latin sources.
