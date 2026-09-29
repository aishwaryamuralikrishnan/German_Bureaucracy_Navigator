# Evaluation run 2026-09-17_1412

23 questions · created 2026-09-17T14:12:40 · total cost $0.25 (OpenRouter credits used by this run — agent, judge, embeddings and reranker)

## 1 · Retrieval — hit-rate@5 and MRR (20 questions with an expected document)

| configuration | hit-rate | MRR |
|---|---:|---:|
| dense only | 95 % | 0.85 |
| BM25 only | 95 % | 0.90 |
| hybrid (RRF) | 100 % | 0.85 |
| hybrid + reranker | 100 % | 0.97 |

reranker: openrouter:cohere/rerank-4-fast · embeddings: openai/text-embedding-3-small · 66.7 s

## 2 · Faithfulness (judge google/gemini-2.5-flash via ragas)

mean **0.95** · probes 0.91 · others 0.99

scored by: 23 × ragas

Lowest scores:

- Q18 0.45 — 6 unsupported claim(s)
- Q07 0.70 — 9 unsupported claim(s)
- Q31 0.87 — 2 unsupported claim(s)
- Q26 0.90 — 1 unsupported claim(s)
- Q16 0.92 — 1 unsupported claim(s)

## 3 · Tool selection — 21 / 23 correct (91 %)

| group | questions | correct | failures |
|---|---:|---:|---|
| knowledge | 2 | 2 | — |
| mixed | 17 | 15 | Q11 (missing search_knowledge_base); Q24 (missing search_knowledge_base, convert_currency.gross_or_net: expected gross, got 'unknown') |
| tool_only | 1 | 1 | — |
| not_covered | 3 | 3 | — |

## 4 · Not-covered questions handled — 3 / 3

| id | notice shown | citations | handled |
|:---:|:---:|:---:|:---:|
| Q27 | yes | 0 | yes |
| Q28 | yes | 0 | yes |
| Q30 (hard) | no | 1 | yes |

agent model: openai/gpt-4o-mini · 806.0 s · tokens in/out 241,508 / 7,171 · answers revised by the grounding step: 5 / 23
questions with a failed tool call in the final attempt: 0 / 23 (re-run once after a failure: 0)
