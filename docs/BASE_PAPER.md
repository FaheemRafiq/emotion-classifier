# Intent2Act — Base Paper Analysis and Viva Notes

**Project:** Intent2Act: Bridging the Intention–Behaviour Gap with an Emotion-Aware, Proof-Verified AI Mentor
(previously "Mentor AI").

> Numbers marked **E1**, **E2**, **E3** come from the comparison experiments in `experiments/` (method in
> `experiments/README.md`) and `reports/metrics/base_paper_comparison.json`. Every other Intent2Act number comes from
> `reports/metrics/complex_eval_v2-deberta-v3-base-e6-20260928.json` and `final_test_v2-deberta-v3-base-e6-20260928.json`.

---

## 1. The base paper

> Jörke, M., Genç, D., Teutschbein, V., Sapkota, S., Chung, S., Schmiedmayer, P., Campero, M. I., King, A. C.,
> Brunskill, E., & Landay, J. A. (2026). **Bloom: Designing for LLM-Augmented Behavior Change Interactions.**
> In *Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)*, Barcelona, Spain,
> pp. 1–27. ACM. https://doi.org/10.1145/3772318.3790506 (preprint arXiv:2510.05449; code and safety benchmark:
> https://github.com/StanfordHCI/Bloom)

**Summary.** Bloom is a Stanford iOS app for physical activity. It combines an LLM health coach called "Beebo"
(GPT-4o, `gpt-4o-2024-11-20`) with established behaviour-change design patterns:
- weekly plans;
- an ambient garden display on the lock screen;
- LLM-written notifications;
- data visualisations.

The team also produced a five-category taxonomy of coaching harms and a safety benchmark of 600 examples. They ran
a four-week randomised field study with 54 participants, comparing Bloom against the same app without the LLM.
The results:
- The LLM improved **mindsets**: participants saw more benefit in exercise, enjoyed it more and were more
  self-compassionate.
- Both groups increased physical activity, and the share meeting 150 min/week doubled (36% → 72%).
- But there was **no advantage for the LLM condition in actual short-term physical activity**.
- Participants said that **plans, ambient displays and notifications — not chat — were the main sources of
  accountability** (finding Q1).

**Why Bloom is the right base paper:**
1. It is **recent and peer-reviewed**: CHI is the top-ranked venue in Human–Computer Interaction, and the paper is
   indexed in Google Scholar and the ACM Digital Library.
2. It addresses **the same problem** as our project: using an LLM coach to turn people's intentions into consistent
   behaviour.
3. It is a **built and evaluated system**, so there is a concrete baseline we can measurably upgrade.
4. It **publishes a benchmark** (the safety dataset), so one part of the comparison can be like-for-like.
5. It **states its own open problems**, and those are exactly what Intent2Act addresses:
   - no behavioural advantage from the LLM;
   - accountability came from structure, not chat;
   - gamification and social features left out;
   - iOS / Apple Watch only.

---

## 2. Problem lineage

1. **The intention–behaviour gap.** People often intend to build a habit (exercise, study, sleep) but do not act
   on that intention consistently.
2. **Human coaching works but does not scale.** Programmes such as Stanford's *Active Choices* are effective but
   expensive and limited by coach availability.
3. **LLM coaches.** GPTCoach (Jörke et al., CHI 2025) showed that an LLM can run a motivational-interviewing
   onboarding conversation. That was a single-session lab study with N = 16.
4. **Bloom (CHI 2026)** moved from chat-only to LLM + UI patterns over four weeks. Mindsets improved, but **behaviour
   did not improve more than in the no-LLM control**.
5. **Open problem left by Bloom:** what makes LLM coaching change *behaviour*, and not only *feelings about*
   behaviour?
   - Bloom's own evidence says accountability is the key.
   - In Bloom, accountability rested on self-marked plans and wearable logs, with no verification.
   - The coach also could not perceive the user's emotional state beyond what the prompt inferred.
6. **Intent2Act** targets exactly this mechanism:
   - **verified accountability**: proof submission with AI + human review;
   - **emotion awareness**: a dedicated emotion model;
   - **gamified consistency**: streaks, XP and levels;
   - **safety**: a guard on both the user's message and the mentor's reply.

---

## 3. Bloom in detail

| Aspect | Bloom |
|---|---|
| Platform | iOS (React Native + Swift), Apple Watch required in the study; Python FastAPI backend on Google Cloud Run; Firestore |
| LLM | GPT-4o (`gpt-4o-2024-11-20`) for every LLM call |
| Coach design | "Beebo"; dialogue-state prompt chain (Active Choices content) + motivational-interviewing prompt chain (MISC strategies) |
| Modes | Onboarding, weekly check-in, at-will chat |
| Tools | `query_health_data` (HealthKit), `generate_plan`; at-will chat can `add_workout` / `delete_workout` |
| Other UI | Weekly FITT plans, garden ambient display, LLM notifications (morning, post-activity, evening), plan/progress summaries |
| Memory | Summary-based (timestamped summary after each conversation) |
| Safety | Taxonomy of 5 harm categories from red-teaming interviews with 4 experts; one few-shot GPT-4o classifier **per category**, applied to every coach message; harmful messages are rewritten by a revision prompt |
| Safety results | Test set (100): overall strict F1 0.86 (recall 0.96); corrected test: F1 0.92; after revision, 0.18% still harmful |
| Study | 4-week randomised between-subjects field study, N = 54 (LLM vs no-LLM control), May–June 2025 |
| Behaviour result | PA increased in both groups; **no advantage for the LLM condition** (H2/H3 inconclusive) |
| Mindset result | LLM group: stronger PA-adequacy mindset, more enjoyment, more self-compassion, hope, active identity |
| Plans | More varied plans; completion 87% (LLM) vs 83% (control); more plan edits |
| Engagement | 5.6× daily app time in the LLM group |
| Stated limitations | Short (4 weeks), small N, underpowered for between-group effects; iOS + Apple Watch owners only; self-selected, compensated, high-income sample; no fine-tuning or multi-agent designs; **gamification and social features not explored**; the coach sometimes hallucinated answers about app features |

**Bloom's harm taxonomy** (Table 2 of the paper; wording paraphrased):
1. Risk of bodily harm or overexertion
2. Body image and eating disorders
3. Mental health
4. Negative mindsets and feedback (guilt or self-worth as motivation)
5. Inaccurate or out-of-scope advice

> Licence note: Bloom's prompts and `taxonomy.pdf` are © Stanford and need written approval to reuse. Intent2Act
> **does not use them**. We only use the published category names and definitions (cited) and the benchmark data,
> which is offered for research and educational use. The data is not redistributed.

---

## 4. Supporting literature ("Google Research legacy")

| Ref | Paper | How Intent2Act relates |
|---|---|---|
| S1 | Khasentino, J., Belyaeva, A., Liu, X., … McLean, C. Y. (2025). A personal health large language model for sleep and fitness coaching. *Nature Medicine*, 31, 3394–3403. https://doi.org/10.1038/s41591-025-03888-0 (Google Research, PH-LLM, fine-tuned Gemini) | Google shows that a specialised LLM can reach expert level on coaching *knowledge*. Intent2Act adds the missing *action* loop: plans, proof and accountability. |
| S2 | Heydari, A. A., et al. (2025). The Anatomy of a Personal Health Agent. arXiv:2508.20148 (Google Research, preprint) | Google's multi-agent design includes a health-coach agent using motivational interviewing. It has no verified accountability and no dedicated emotion model, which are both Intent2Act features. |
| S3 | Merrill, M. A., Paruchuri, A., Rezaei, N., … Liu, X. (2026). Transforming wearable data into personal health insights using large language model agents. *Nature Communications*, 17, 1143. https://doi.org/10.1038/s41467-025-67922-y (Google Research, PHIA) | An LLM agent that reasons over personal data. Intent2Act likewise feeds the user's progress data (tasks, streaks, proof reviews) to its mentor. |
| S4 | Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*, 4040–4054. https://doi.org/10.18653/v1/2020.acl-main.372 (Google Research) | Our emotion model is trained on Google's GoEmotions, so the project builds directly on Google Research data. |
| S5 | Jörke, M., Sapkota, S., Warkenthien, L., … Landay, J. A. (2025). GPTCoach: Towards LLM-Based Physical Activity Coaching. *CHI 2025*, 1–46. https://doi.org/10.1145/3706598.3713819 | Bloom's predecessor (single-session lab study, N = 16). Shows the lineage. |
| S6 | Meywirth, S., Janson, A., & Söllner, M. (2025). Personalized Coaching for Lifestyle Behavior Change through Large Language Models: A Qualitative Study. *HICSS-58*, 3379–3388. https://hdl.handle.net/10125/109250 | User requirements for LLM coaches: personalisation, tone choice, accountability, human oversight. Intent2Act implements them. |

---

## 5. Gap → upgrade (core argument)

| # | Bloom (base) | Intent2Act (upgrade) | Evidence |
|---|---|---|---|
| U1 | LLM coaching gave no behaviour advantage; accountability came from plans, and completion was **self-marked or wearable-logged** | **Proof-verified tasks** (image / audio / video / text). An AI pre-review auto-approves only at confidence ≥ 0.85; everything else goes to an admin. Progress, streaks and XP are gated on approved proof. | System design (Ch. 3, Ch. 6) |
| U2 | **No emotion sensing**; tone is fixed by the prompt | Dedicated **DeBERTa-v3 emotion model** (7 classes); its result is injected into every mentor prompt. Test macro-F1 **0.721**, implicit-emotion (ISEAR) macro-F1 **0.781** vs 0.498 for DistilBERT, **89 ms on CPU**, **$0 per message**. | Ch. 4; **E1**: better than a zero-shot LLM on conversational text (0.77 vs 0.60), worse on implicit situations (0.81 vs 0.91), similar on mixed emotion |
| U3 | Physical activity only; iPhone + Apple Watch owners only | **Any habit** (fitness, study, sleep, …); web/PWA on any device; no wearable needed | Design |
| U4 | Gamification **explicitly excluded** (left as future work) | XP, levels, streaks, streak freezes, milestone rewards | Design, screenshots |
| U5 | Safety: 5 separate GPT-4o classifiers on the **coach's reply only**, plus a revision call | One `gpt-4o-mini` **safety guard** on the same 5 categories, applied to the **user's message** (crisis escalation with helplines, no LLM improvisation) **and** the mentor's reply (logged for admin oversight); fails open | **E2** on Bloom's benchmark: relaxed F1 0.945 vs 0.93 (validation), 0.84 vs 0.93 (corrected test); **E3**: ≈42× cheaper |
| U6 | Weekly check-ins adapt the plan; at-will chat can add or delete workouts | Weekly check-in built on **verified** completion data, plus an `update_challenge_for_user` tool that edits only future tasks after the user explicitly agrees | Design, test cases |
| U7 | GPT-4o for every LLM call (chat, notifications, 5 safety calls, revision) | Per-task model routing (chat, nudges, proof review, safety) to cheaper models, plus a local classifier at no marginal cost | **E3**: ≈ $1.16 vs $34.43 per 1,000 messages (≈30×) |
| U8 | One persona (Beebo) | Several mentor personas with different tones (requested by HICSS participants, S6) | Design |

### Honest positioning (say this in the viva and the report)
- **What we claim:** Intent2Act is designed around the mechanism Bloom identified (accountability through structure),
  and it strengthens that mechanism with verification, emotion awareness and gamification. This is shown through
  design and measured component benchmarks (E1–E3).
- **What we do not claim:** we have **not** run a randomised behavioural study like Bloom's, so we do not claim
  proven behaviour change.
- **Proposed future study:** 4 weeks, two arms (Intent2Act with vs without proof verification). Outcomes:
  - verified task-completion rate;
  - self-reported habit strength (Self-Report Habit Index, SRHI);
  - retention.
- **Safety comparison:** Bloom's filters are already strong (corrected-test F1 0.93). Our single cheap guard matches
  them on validation (0.945 vs 0.93) but is lower on the small test set (0.84 vs 0.93, wide confidence interval),
  with higher precision and lower recall. We claim **comparable detection at about 1/40 of the safety cost**, plus
  *input-side* crisis escalation that Bloom does not have. We do not claim to beat Bloom's classifiers.
- **Emotion comparison:** E1 shows that a frontier LLM reads implicit, situational emotion better than our model
  (ISEAR 0.91 vs 0.81). Our model is better on conversational text (0.77 vs 0.60) and similar on mixed emotion. We
  therefore justify the dedicated model by **cost, latency, privacy and calibrated confidence**, not by accuracy alone.

---

## 6. Viva questions and answers

**Q1. What is your base paper and why that one?**
Bloom, from Stanford, published at CHI 2026, the top venue in Human–Computer Interaction (DOI
10.1145/3772318.3790506). It solves the same problem we do: using an LLM coach to help people turn intentions into
regular behaviour. It is a complete system with a published evaluation and a public safety benchmark, so we can
compare against it. Its main finding, that the LLM improved mindsets but not behaviour, gives us a clear gap to
upgrade.

**Q2. In one sentence, how is your project an upgrade?**
Bloom showed that accountability, not chat, drives behaviour, but its accountability relied on unverified,
self-marked plans. Intent2Act adds **proof-verified accountability**, **emotion-aware mentoring** from a dedicated
model, **gamification** that Bloom left out, and a **cheaper two-sided safety guard**, for any habit and on any
device.

**Q3. Bloom found no behaviour gain. How do you know yours would do better?**
We do not claim proven behaviour change, because we have not run a randomised study. What we can say is that
Bloom's own participants named plans and structure as the source of accountability (finding Q1). Intent2Act
strengthens exactly that, because a task only counts once proof is approved. We measured the parts we could measure
(emotion detection, safety, cost), and we propose a four-week, two-arm study as future work.

**Q4. Why not just let GPT-4o detect emotions instead of training a model?**
We tested exactly this in E1 on 1,105 human-labelled texts. To be honest: a frontier LLM is better on implicit,
situation-described emotion (ISEAR 0.91 vs our 0.81), but our model is clearly better on conversational text
(GoEmotions 0.77 vs 0.60) and similar on mixed emotions. The deciding factors are practical. Our model runs in about
89 ms on a CPU, costs nothing per message and keeps the text on our server; an extra LLM call adds 1–3 seconds, costs
money on every message and sends the text to a third party. Our model also gives calibrated confidence (ECE 0.017), so
the mentor only uses the emotion when it is reliable (threshold 0.6). A hybrid (local model first, LLM when unsure) is
future work.

**Q5. How is your safety guard evaluated, and why on Bloom's benchmark?**
We ran the deployed guard on Bloom's published benchmark: 400 validation and 100 test examples across the same five
harm categories, so it is a like-for-like comparison with their Table 3. Like Bloom, we tuned the prompt once on
validation only and ran the test set once. Relaxed F1 was 0.945 vs Bloom's 0.93 on validation, and 0.84 vs 0.93 on
the corrected test (only 100 items, so the interval is wide). Our guard is more precise but misses more harmful replies,
mainly confident medical advice. The trade-off is cost: one `gpt-4o-mini` call (about $0.00028) instead of five GPT-4o
calls, roughly 42 times cheaper. We also check the *user's* message, so a crisis gets a fixed, safe reply with
helplines before the LLM writes anything.

**Q6. What does "Google research legacy" mean in your work?**
Our emotion model is trained on GoEmotions, a Google Research dataset (ACL 2020). We also position the project
against Google's recent personal-health LLM work:
- PH-LLM in *Nature Medicine* 2025, which shows coaching *knowledge*;
- the Personal Health Agent, a multi-agent coach;
- PHIA in *Nature Communications* 2026, which reasons over personal data.

None of these verifies that the user actually did the action, and that is our contribution.

**Q7. How did you prevent data leakage in the emotion model?**
- The GoEmotions train/validation/test splits are frozen and fingerprinted with SHA-256.
- Model selection used validation data only.
- The test set was used once per model version, and every rerun is logged in `forced_reruns.log`.
- ISEAR was split 70/15/15 before training.
- The complexity slices were defined by fixed rules before any evaluation.

**Q8. What are the ethical risks?**
- The emotion model is a signal, not a diagnosis, and it is only used above a confidence threshold.
- Crisis messages get a fixed escalation reply with helplines, and the admins are alerted.
- Flagged mentor replies are logged for human review.
- Proof media is private to the user and the reviewers.
- Safety data from Bloom is used for research only and is not redistributed.

**Q9. Why do admins review proof, and does it scale?**
An AI pre-review handles the easy cases. It auto-approves only when its confidence is at least 0.85; everything else
is escalated to a human. This keeps quality high while most of the load stays automated. To scale further, we would
add peer or mentor review and sampling audits.

**Q10. What is the intention–behaviour gap?**
It is the well-documented finding that people's intentions only weakly predict what they actually do. Many people
who intend to exercise or study do not act on it. Intent2Act is built to close that gap through specific daily
tasks, verified completion, streaks, and timely, emotion-aware mentoring.

**Q11. What are the limitations of your evaluation?**
- No randomised user study.
- The Mentor-style emotion test set has draft labels.
- The LLM-only baseline in E1 was labelled by Claude, not GPT-4o; the Mentor-style set was left out of E1 because
  Claude wrote it.
- E2's test set has only 100 items, and our guard's recall there is below Bloom's.
- The safety benchmark covers physical-activity coaching, while Intent2Act covers any habit.
- Sarcasm remains hard for the emotion model.

**Q12. What happens if the emotion service or the safety guard is down?**
Both fail open: the chat continues without the extra signal, and a warning is logged. Both have timeouts, so a
slow service cannot block the conversation.

**Q13. Why DeBERTa-v3 instead of DistilBERT?**
On ordinary Reddit text they tie (macro-F1 0.721 vs 0.717). On text where the emotion has to be inferred from the
situation, DeBERTa is far better:
- ISEAR macro-F1 0.781 vs 0.498 (+28 points);
- Mentor-style check-ins 0.810 vs 0.602 (provisional labels).

Mentor messages look much more like the second kind of text. The cost is speed: 89 ms on CPU, which is still fine
for chat.

**Q14. How would you run a proper user study?**
- Four weeks, two arms, with randomised assignment:
  1. full Intent2Act;
  2. the same app without proof verification.
- Primary outcome: verified task-completion rate.
- Secondary outcomes: SRHI habit strength, streak length and retention.
- Get ethics approval and a pre-registered analysis plan, and aim for at least 30 participants per arm.

**Q15. What would you do with six more months?**
- Run that study.
- Add emotion-aware motivational-interviewing strategies and per-message emotion history.
- Improve sarcasm detection.
- Add Urdu / Roman-Urdu support.
- Fine-tune a small model as an on-device safety guard.
