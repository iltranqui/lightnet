use std::fs;

use lightnet_lm::{
    BASE_VOCAB_SIZE, BOS_ID, BYTE_BASE, ByteBpe, EOS_ID, NormalizationSpec, PAD_ID, UNK_ID,
    normalize_text,
};

fn plain() -> ByteBpe {
    ByteBpe::train(&[], BASE_VOCAB_SIZE, NormalizationSpec::default()).expect("base tokenizer")
}

#[test]
fn lm_tokenizer_roundtrips_bytes() {
    let tokenizer = plain();
    let all: Vec<u8> = (0..=255u8).collect();
    let expected: Vec<u32> = (0..=255u32).map(|byte| BYTE_BASE + byte).collect();
    assert_eq!(tokenizer.encode_bytes(&all), expected);
    assert_eq!(tokenizer.decode_bytes(&expected).expect("decode"), all);

    for byte in 0..=255u8 {
        let id = BYTE_BASE + u32::from(byte);
        assert_eq!(
            tokenizer.decode_bytes(&[id]).expect("single byte"),
            vec![byte]
        );
    }

    for control in [PAD_ID, BOS_ID, EOS_ID, UNK_ID] {
        assert!(tokenizer.decode_bytes(&[control]).is_err());
    }
    assert!(tokenizer.decode_bytes(&[BASE_VOCAB_SIZE as u32]).is_err());

    let trained =
        ByteBpe::train(&[b"ababab".to_vec()], 262, NormalizationSpec::default()).expect("train");
    assert_eq!(
        trained
            .decode_bytes(&trained.encode_bytes(&all))
            .expect("decode"),
        all
    );
}

#[test]
fn lm_tokenizer_preserves_unicode() {
    let spec = NormalizationSpec::default();
    let text = "Lūna caelum, Ἀθῆναι, e\u{0301}, 🦀, 漢字";
    let normalized = normalize_text(text, &spec);
    assert_eq!(normalized, "Lūna caelum, Ἀθῆναι, é, 🦀, 漢字");

    let tokenizer = ByteBpe::train(&[text.as_bytes().to_vec()], 300, spec).expect("train");
    let ids = tokenizer.encode(text).expect("encode");
    assert_eq!(
        tokenizer.decode_bytes(&ids).expect("decode"),
        normalized.as_bytes()
    );

    // Characters never seen during training still round-trip through bytes.
    for unseen in ["Ω", "ꙮ", "\u{10FFFF}"] {
        let ids = tokenizer.encode(unseen).expect("encode unseen");
        assert_eq!(
            tokenizer.decode_bytes(&ids).expect("decode unseen"),
            unseen.as_bytes()
        );
    }

    // Line endings are canonical even without Latin folding.
    assert_eq!(normalize_text("a\r\nb\rc", &spec), "a\nb\nc");
}

#[test]
fn lm_tokenizer_latin_policy() {
    let fold = NormalizationSpec { latin_fold: true };
    assert_eq!(normalize_text("VJV æ œ", &fold), "UIU ae oe");
    assert_eq!(normalize_text("Ærōma Œdipus", &fold), "AErōma OEdipus");

    // Fold happens before NFC, so a decomposed macron recomposes.
    assert_eq!(normalize_text("v\u{0304}", &fold), "ū");
    assert_eq!(normalize_text("u\u{0304}", &fold), "ū");
    assert_eq!(normalize_text("j\u{0304}", &fold), "ī");
    // Macrons on other vowels are preserved.
    assert_eq!(normalize_text("āēīōū", &fold), "āēīōū");

    let keep = NormalizationSpec::default();
    assert_eq!(normalize_text("VJV æ œ", &keep), "VJV æ œ");
    assert_eq!(normalize_text("Ærōma", &keep), "Ærōma");
}

#[test]
fn lm_tokenizer_deterministic_merges() {
    let documents = vec![b"banana bandana".to_vec(), b"ananas".to_vec()];
    let first = ByteBpe::train(&documents, 300, NormalizationSpec::default()).expect("train");
    let second = ByteBpe::train(&documents, 300, NormalizationSpec::default()).expect("train");
    assert_eq!(first, second);
    assert_eq!(first.artifact_hash(), second.artifact_hash());
    assert_eq!(first.merge_count(), second.merge_count());
    assert_eq!(first.requested_vocab_size(), 300);
    assert!(first.vocab_size() > BASE_VOCAB_SIZE);
    assert!(first.vocab_size() <= 300);

    let mut reversed = documents.clone();
    reversed.reverse();
    let permuted = ByteBpe::train(&reversed, 300, NormalizationSpec::default()).expect("train");
    assert_eq!(first.artifact_hash(), permuted.artifact_hash());

    // "Stop when no pair repeats": isolated bytes never reach the request.
    let unique: Vec<Vec<u8>> = (32u8..=126).map(|byte| vec![byte]).collect();
    let limited = ByteBpe::train(&unique, 400, NormalizationSpec::default()).expect("train");
    assert_eq!(limited.vocab_size(), BASE_VOCAB_SIZE);
    assert!(limited.vocab_size() < limited.requested_vocab_size());

    // Save/load preserves the artifact and its hash.
    let tokenizer = ByteBpe::train(&documents, 300, NormalizationSpec::default()).expect("train");
    let path =
        std::env::temp_dir().join(format!("lightnet-lm-bpe-{}-{}.txt", std::process::id(), 1));
    tokenizer.save(&path).expect("save");
    let loaded = ByteBpe::load(&path).expect("load");
    assert_eq!(loaded, tokenizer);
    assert_eq!(loaded.artifact_hash(), tokenizer.artifact_hash());
    fs::remove_file(&path).expect("cleanup");
}

#[test]
fn lm_tokenizer_rejects_corrupt_artifacts() {
    let tokenizer =
        ByteBpe::train(&[b"ababab".to_vec()], 262, NormalizationSpec::default()).expect("train");
    assert_eq!(tokenizer.merge_count(), 2);
    let artifact = tokenizer.to_artifact();
    assert_eq!(
        ByteBpe::from_artifact(&artifact)
            .expect("round-trip artifact")
            .artifact_hash(),
        tokenizer.artifact_hash()
    );

    assert!(ByteBpe::from_artifact(&artifact.replace("version 1", "version 99")).is_err());
    assert!(ByteBpe::from_artifact(&artifact.replace("lightnet-bpe", "other-bpe")).is_err());
    assert!(ByteBpe::from_artifact("").is_err());

    let header = "lightnet-bpe\nversion 1\nlatin_fold 0\nrequested_vocab_size 263\nmerges 2\n";
    // A merge that references itself is a forward reference.
    assert!(ByteBpe::from_artifact(&format!("{header}260 260 4\n261 260 4\n")).is_err());
    // Duplicate merge IDs.
    assert!(ByteBpe::from_artifact(&format!("{header}260 4 5\n260 4 5\n")).is_err());
    // Control tokens are not merge operands.
    assert!(ByteBpe::from_artifact(&format!("{header}260 1 5\n261 260 4\n")).is_err());
    // Truncated merge list.
    assert!(ByteBpe::from_artifact(&format!("{header}260 4 5\n")).is_err());
    // Malformed and trailing lines.
    assert!(ByteBpe::from_artifact(&format!("{header}260 4\n261 260 4\n")).is_err());
    assert!(ByteBpe::from_artifact(&format!("{header}260 4 5\n261 260 4\njunk\n")).is_err());

    let short = "lightnet-bpe\nversion 1\nlatin_fold 0\nrequested_vocab_size 260\nmerges 1\n\
                 260 4 5\n";
    assert!(ByteBpe::from_artifact(short).is_err());
}

#[test]
fn lm_tokenizer_special_literals_are_text() {
    let tokenizer = ByteBpe::train(
        &[b"a Bos token and an EOS token".to_vec()],
        300,
        NormalizationSpec::default(),
    )
    .expect("train");
    for literal in ["<|bos|>", "<s>", "[BOS]", "</s>", "<pad>", "<unk>"] {
        let ids = tokenizer.encode(literal).expect("encode literal");
        for control in [PAD_ID, BOS_ID, EOS_ID, UNK_ID] {
            assert!(
                !ids.contains(&control),
                "{literal} produced control {control}"
            );
        }
        assert_eq!(
            tokenizer.decode_bytes(&ids).expect("decode literal"),
            literal.as_bytes()
        );
    }
}

#[test]
fn lm_tokenizer_merges_apply_in_rank_order() {
    let tokenizer =
        ByteBpe::train(&[b"ababab".to_vec()], 262, NormalizationSpec::default()).expect("train");
    assert_eq!(tokenizer.vocab_size(), 262);
    // (a,b) is merge rank 0, (260,260) is rank 1; non-overlapping left-to-right.
    assert_eq!(tokenizer.encode_bytes(b"ababab"), vec![261, 260]);
    assert_eq!(
        tokenizer.decode_bytes(&[261, 260]).expect("decode"),
        b"ababab"
    );
}
