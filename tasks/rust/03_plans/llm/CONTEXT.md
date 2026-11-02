# llm

Language-model line in `src/lightnet-lm`: Latin pilot, decoder, encoder-decoder, recursive decoder. Drafted by Codex, unreviewed.

## Table of Contents

- [Files](#files)

## Files

| File | Holds |
|---|---|
| [ROADMAP_LLM.md](ROADMAP_LLM.md) | Goal document: state snapshot, loop, versioned roadmap todo (CPU from `v0.1.067`, GPU from `v0.2.227`), delegation and user gates |
| [plan_llm_decoder.md](plan_llm_decoder.md) | Decoder-only model, resumable training, cached generation, Latin pilot |
| [plan_llm_encdec.md](plan_llm_encdec.md) | Encoder and decoder stacks, encoder-decoder composition |
| [plan_llm_recursive.md](plan_llm_recursive.md) | Weight-shared recurrent decoder |
| [plan_llm_training_recipe.md](plan_llm_training_recipe.md) | Pretraining, mid-training and chat recipe for a 30-60M Latin decoder |
| [llm_data.md](llm_data.md) | Tokenizer artifact contract and tokenizer rules |
| [latin_datasets.md](latin_datasets.md) | Candidate Latin corpora, treebanks and eval sets (unreviewed) |
