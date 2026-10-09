# Built-in local answer model: quality gate (2026-10-01)

Model: Qwen3-1.7B Q4_K_M via Homebrew llama-server, `-c 8192`, thinking off.
Corpus: `fixtures/docs` (6 docs, 34 chunks), real embedder + reranker, headless `ragcore ask`
with the `local` connection active. Machine: Apple Silicon Mac (this dev machine).

## Resource behaviour (measured)
- Before first use: no llama-server process.
- Loaded: ~2.1 GB RSS (1.7B Q4, context 8192). Unchanged after an answer.
- Released: process gone 62 s after the last lease ends (IDLE_S = 60).
- Speed: prompt ~580 tok/s, generation ~77 tok/s. The slow-speed banner threshold is 5 tok/s,
  so this machine never triggers it.
- `ragcore ask` left one llama-server per call running after exit (5 orphans). Fixed:
  `_ask` now stops the built-in model in `finally` (test_cli).

## Answer quality (5 questions, verdict by reading the answer against the doc)
| Question | Verdict |
|---|---|
| what does reranking do | Good: correct, cited the reranking doc p. 3 |
| why use hybrid search instead of only vector search | Bad: pasted a cross-encoder passage; did not answer the question |
| how should chunk size be chosen | Good: correct (count with the embedder's tokenizer), valid citation |
| how do I evaluate a retrieval system | OK: on topic, vague, valid citation |
| what is the capital of France (not in corpus) | Leaks world knowledge ("Paris") with an invalid citation id; the validator drops the citation, so the UI shows its "no citations" trust warning |

Citation validity: every citation that reached the output after validation pointed at a
retrieved passage; the one fabricated id (`document_id:1:page1`) was dropped as designed.

## Verdict
Usable for simple factual lookups over a library, weak on comparative ("why X over Y")
questions and does not refuse out-of-corpus questions. Good enough to ship as the
zero-setup default because the speed banner and existing connections give a way up, but a
4B-class model (spec says tiers are a later change) would likely fix the comparative and
refusal failures. Worth a follow-up: tighten the system prompt to refuse when passages do not
answer the question, which also helps remote models.

## Not verified here
- GUI flows (onboarding card, Settings row, banner): typechecked and unit-tested only.
- Force-quitting the app mid-answer: not exercised in the GUI, but `sidecars.rs` kills ragcore's
  whole process group (`libc_kill(-pid)`), which includes the llama-server it spawned.
