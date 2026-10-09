# Base-paper comparison experiments (Intent2Act vs Bloom, CHI 2026)

Base paper: Jörke et al. (2026), *Bloom: Designing for LLM-Augmented Behavior Change Interactions*, CHI '26,
https://doi.org/10.1145/3772318.3790506. All numbers quoted in the FYP report come from
`reports/metrics/base_paper_comparison.json`, built by `experiments/build_comparison.py`.

| Exp. | Question | Run | Output |
|---|---|---|---|
| E1 | Is a dedicated emotion model better than letting a general LLM read emotion? | `make_batches.py` → label-blind batches labelled by Claude under `INSTRUCTION.md` (answers/) → `score.py` | `e1_emotion/results.json` |
| E2 | How does the Intent2Act safety guard do on Bloom's own safety benchmark? | `run_guard.py` (needs the mentor-ai OpenAI key; cached) → `score.py` | `e2_safety/results.json` |
| E3 | What does the per-message AI pipeline cost? | `cost_model.py` | `e3_cost/results.json` |

Then: `uv run python experiments/build_comparison.py && uv run python experiments/make_figures.py`.

## E1 — emotion: DeBERTa-v3 (ours) vs LLM-only (Claude, zero-shot)
- Sample (seed 42): 647 GoEmotions test texts (up to 100 per class; balanced, so macro F1 differs from the full
  test-set figure), 350 ISEAR test texts (70 per class), 108 mixed-emotion texts, 154 Mentor-style texts.
  Texts from all sources were shuffled together under neutral ids; gold labels are only in `key.jsonl`.
- The LLM baseline is Claude (Anthropic), run as four fresh sub-agents that saw only the fixed instruction and the
  batches (never the key, the code or the data folders), and were told not to use code or keyword rules.
- Results (macro F1, 95% bootstrap CI): GoEmotions **0.767** [0.733, 0.798] vs LLM 0.599 [0.560, 0.637];
  ISEAR **0.812** [0.770, 0.849] vs LLM **0.913** [0.880, 0.941]; mixed top-2 covers all gold 0.481 vs 0.435.
- **Mentor-style set excluded from claims**: its texts and draft labels were written by Claude, so Claude's score
  on it (0.968) is circular.
- Reading: a frontier LLM is better at implicit, situation-described emotion (ISEAR); the dedicated model is better
  on conversational text (GoEmotions) and about equal on mixed emotion. Its real advantages are cost ($0 per
  message), latency (≈89 ms CPU, no network), privacy (text stays on the server) and calibrated confidence
  (ECE 0.017) that lets the app use the signal only when it is reliable.
- Caveats: Claude is not GPT-4o (Bloom's model); in-session labelling is not a scripted API run (instruction and
  raw answers archived).

## E2 — safety guard on Bloom's benchmark
- Data: `StanfordHCI/Bloom/safety/data` (validation 400 after removing Bloom's 100 few-shot training rows; test 100;
  author-corrected test 100). Research/educational use only — **not redistributed** (`data/` is gitignored).
  Bloom's proprietary prompts were not used; our prompt is our own wording of the published taxonomy (Table 2).
- Guard = the deployed prompt (`mentor-ai/apps/api/templates/prompts/safety-guard.txt`), gpt-4o-mini, temperature 0,
  JSON schema, MENTOR_REPLY mode.
- Protocol: v1 prompt written from the taxonomy only, scored on all splits (`results_v1_zero_shot.json`); one
  revision made from **validation** errors only (as Bloom tuned on validation); the final prompt was then run on the
  test splits once.
- Final (relaxed F1, ours vs Bloom): validation **0.945** vs 0.93; test 0.800 vs 0.86; corrected test 0.841 vs 0.93.
  Precision 0.90–0.95 vs Bloom 0.78–0.90; recall 0.79–0.94 vs Bloom 0.96–0.98. Weakest categories: inaccurate /
  out-of-scope advice and mental health.
- One check costs ≈ $0.00028 (1,719 input + 38 output tokens), median latency 2.1 s from this machine.
- API spend for E2: ≈ $0.28 in total (v1 run, one validation tuning run, final test run).

## E3 — cost per 1,000 chat messages (estimate)
Bloom-style (GPT-4o chat + 5 GPT-4o safety classifiers per reply) ≈ **$34.43**; Intent2Act (gpt-4o-mini chat +
gpt-4o-mini guard on message and reply + local emotion model) ≈ **$1.16** → about 30× cheaper (safety alone ≈ 42×).
Assumptions are in `e3_cost/cost_model.py` (3,000/250 chat tokens for both; Bloom's classifier prompts assumed the
same size as ours, which is conservative).
