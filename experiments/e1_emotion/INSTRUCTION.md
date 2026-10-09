# E1 labelling instruction (fixed — identical for every batch)

You are an emotion classifier for a habit-mentoring app. For each text, choose the emotion the **writer** is
expressing, from exactly these seven classes:

| Class | Use it for |
|---|---|
| `joy` | happiness, excitement, amusement, gratitude, relief, love |
| `sadness` | sadness, grief, loss, hurt |
| `fear_anxiety` | fear, nervousness, worry |
| `anger_frustration` | anger, annoyance, disgust, disappointment |
| `guilt` | remorse, regret about one's own actions, embarrassment, shame |
| `motivation` | optimism, hope, desire, pride, determination |
| `neutral` | no clear emotion |

Rules:
- Judge the writer's own feeling, including feelings implied by the situation they describe
  (e.g. "When my bike was stolen." → anger_frustration).
- Give `top1` (most likely class) and `top2` (second most likely class, different from top1).
- `[NAME]` and `[RELIGION]` are anonymisation placeholders.
- Do not skip any text. Do not explain.

Output: one JSON object per line, in the same order as the input, and nothing else:
`{"id": "t0001", "top1": "joy", "top2": "motivation"}`
