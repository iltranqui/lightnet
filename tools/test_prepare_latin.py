import tempfile
import unittest
from pathlib import Path

from prepare_latin import extract_tei, folded_words, read_conllu, split_for, work_id


class ExtractionTests(unittest.TestCase):
    def xml(self, text, annotated=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.xml'
            path.write_text(text, encoding='utf-8')
            return extract_tei(path, annotated)[0]

    def test_perseus_excludes_notes_languages_and_apparatus(self):
        text = self.xml('''<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><title>Metadata</title></teiHeader><text><body>
          <p>Arma <note>English note</note>uirumque
          <foreign xml:lang="grc">λόγος</foreign> cano <app><lem>Troiae</lem>
          <rdg>variant</rdg></app>.</p><p xml:lang="eng">Translation</p>
          <l>māter <hi>mea</hi></l></body></text></TEI>''')
        self.assertEqual(text, 'Arma uirumque cano Troiae.\nmāter mea\n')

    def test_annotated_enclitic_copy_is_not_surface_text(self):
        text = self.xml('''<TEI><text><body><ab><w>breuibusque</w>
          <w>{breuibusque}</w><w pos="PUNC">,</w><w>māter</w>
          <w pos="PUNC">.</w></ab></body></text></TEI>''', True)
        self.assertEqual(text, 'breuibusque, māter.\n')

    def test_editions_share_work_and_author_split(self):
        self.assertEqual(work_id('urn:cts:latinLit:phi0474.phi051.perseus-lat1'),
                         work_id('phi0474.phi051.perseus-lat2'))
        self.assertEqual(split_for('phi1017', 'Seneca'), 'validation')
        self.assertEqual(split_for('phi1351', 'Tacitus'), 'test')
        self.assertEqual(split_for('phi0474', 'Cicero'), split_for('phi0474', 'Cicero'))

    def test_folded_audit_preserves_macrons(self):
        self.assertEqual(folded_words('Vīta jūvat æquōs.'), ['uīta', 'iūuat', 'aequōs'])

    def test_conllu_groups_charters_across_published_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split, sentence in [('train', 'In Dei nomine.'), ('test', 'Ego subscripsi.')]:
                (Path(tmp) / f'la_llct-ud-{split}.conllu').write_text(
                    f"# sent_id = {split}-s1\n# text = {sentence}\n"
                    "# reference = document_id='36:1047'-span='1'\n\n", encoding='utf-8')
            docs = read_conllu(Path(tmp), 'ud-llct')
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0]['text'], 'In Dei nomine.\nEgo subscripsi.\n')
            self.assertEqual(docs[0]['split'], 'diagnostic_llct')


if __name__ == '__main__':
    unittest.main()
