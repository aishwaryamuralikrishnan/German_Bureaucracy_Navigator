# 🇩🇪 German Bureaucracy Navigator

A **bilingual** (English / German) RAG + tool-calling assistant for people who have moved to Germany or are about to.
It answers questions about address registration (*Anmeldung*), residence permits and the EU Blue Card, student visas, tax ID and tax classes,
health insurance, banking and SCHUFA, driving licences, recognition of qualifications and family reunification —
with inline citations to a curated knowledge base, calculators for deadlines and salaries, and a live exchange-rate
lookup. Ask in English, in German, or in English peppered with German terms (*"Do I need a Wohnungsgeberbestätigung
for the Anmeldung?"*): the language is detected per message, official German terms are kept and glossed, and the
answer comes back in the language you asked in (or the one you pin in the sidebar).

### At a glance

| | |
|---|---|
| **Try it** | **[Live app on Streamlit Community Cloud](https://germanbureaucracynavigator-nv4spe5owjrbx5kkumhffp.streamlit.app/)** (first load after idle takes about a minute while the knowledge base is built) · run locally in five minutes with [docs/TECHNICAL.md](docs/TECHNICAL.md) |
| **Results** | Hit-rate@5 **100 %** · MRR **0.97** · faithfulness **0.95** · tool selection **21 / 23** · not-covered handling **3 / 3** — 23-question labelled set, run of 17 Sep 2026, $0.25 ([§4](#4-evaluation)) |
| **Stack** | LangChain 1.x + LangGraph · OpenRouter · ChromaDB · RAGAS · Streamlit |
| **How it fits together** | [Architecture diagram](#architecture) below · [agent & tools](#1-the-langchain-agent) · [grounding check](#2-grounding-check) · [guardrails](#3-guardrails-and-out-of-domain-questions) |

![Demo: an English question about moving to Munich on 20 December 2026 answered with the registration deadline from the deadline calculator, the responsible office, late-registration consequences and the documents to bring, each cited to a knowledge-base passage; a German question from a married couple in Berlin on tax class 3/5 versus 4/4, who may switch, and the monthly net in class 3 from the net-salary calculator, answered in German; a Kindergeld question the knowledge base does not cover, answered with a referral to the Familienkasse and no amounts](docs/img/navigator_demo_full.gif)

Setup, commands and troubleshooting live in **[docs/TECHNICAL.md](docs/TECHNICAL.md)**.

<a id="architecture"></a>
![Architecture](docs/img/architecture.png)

## 1. The LangChain agent

The agent is LangChain's `create_agent` running on LangGraph: the model receives a bilingual system prompt with the
DE/EN glossary, the citation and refusal rules, and five tools with Pydantic schemas. It decides per turn which tools
to call (up to six), grounds every factual sentence in a passage or tool result returned *in that turn*, and cites
passages inline as `[S1]`, `[S2]`. Every tool returns the same JSON envelope (`ok`, `data`, `sources`, `warnings`,
`error`), so the UI can render tool traces uniformly and the model can relay a tool failure in plain words.

### 1.1 Knowledge base

![Knowledge base](docs/img/knowledge_base.png)

The knowledge base is 18 hand-compiled markdown documents (103 chunks) covering ten topics, each written from an
official source — federal law texts, BAMF, BZSt, Make-it-in-Germany, city portals — and carrying YAML front matter
with its topic, jurisdiction (federal, Berlin, Munich, Rostock), language and source URL. Documents are English with
the German administrative terms embedded, plus a German-language address-registration document, so both languages have
matching vocabulary to retrieve against.

### 1.2 Hybrid (advanced) RAG

#### 1.2.1 Building the knowledge base

![Building the knowledge base](docs/img/kb_build.png)

The knowledge base is built once, by `scripts/ingest.py` or automatically when the app finds the index empty.

Each of the 18 markdown documents is split on its headings and then, where a section is still long, into pieces of
about 1,800 characters with a 250-character overlap, every chunk prefixed with its *title > section* path — 103 chunks
in all.

Each chunk is embedded with `text-embedding-3-small` via OpenRouter and stored in **Chroma** as one record holding the
chunk id, the vector, the full text and the metadata (topic, jurisdiction, language, source URL), so Chroma is both the
vector index and the only stored copy of the text.

#### 1.2.2 Retrieval pipeline

![Retrieval pipeline](docs/img/hybrid_rag.png)

German terms in an English knowledge base are bridged before the search: a small model rewrites the question into up
to three variants — the German official term from the glossary, an English phrasing and, where relevant, the law
paragraph — so *"blocked account"* also searches for *Sperrkonto* and vice versa. Each variant then goes through the
hybrid search described in [§1.2.3](#123-hybrid-search), which returns the 20 best candidates.

A Cohere cross-encoder (`rerank-4-fast`, via OpenRouter) reads question and passage together and re-orders them, which
is what lifts the expected document to rank 1 in almost every case (MRR 0.85 → 0.97).

Finally a passage is flagged *weak* when it shares no discriminative keyword with the question and does not stand out
semantically, or when the reranker scores it below 0.65; if every returned passage is weak, the tool tells the model
that the knowledge base does not cover the question — the mechanism behind the not-covered handling in §3.

#### 1.2.3 Hybrid search

![Hybrid search](docs/img/hybrid_search.png)

Bureaucracy questions come in two kinds — exact administrative terms and law references (*Sperrkonto*, *§ 17 BMG*),
which keyword search finds and embeddings often blur, and paraphrased everyday questions (*"where do I tell the city I
moved?"*) that only semantic search catches — so the retriever runs both over the same chunks.

**Keyword search** (BM25) tokenises the question the same way as the chunks and scores every chunk on how often it
contains the question's rare words, using an index built in memory at app start from the texts in Chroma.

**Semantic search** embeds the question and asks Chroma for the nearest vectors by cosine similarity, with the
jurisdiction filter applied inside the search.

The two ranked lists are fused with Reciprocal Rank Fusion, which rewards passages both methods rate highly without
needing their scores to be comparable; the evaluation confirms the combination — dense alone and BM25 alone each miss
one question in twenty, hybrid finds all of them.

### 1.3 Tools

**`search_knowledge_base`** — the retrieval tool above. Arguments: `query`, optional `city` (jurisdiction filter) and
`topic` hint, `k` (1–8, default 5); output: numbered passages `[S1]…[Sk]` with title, section, text, relevance score
and a weak/ok flag, plus a coverage verdict and a LOW COVERAGE warning that forbids citing or stating facts.

Retrieval is a tool rather than a fixed step before the model so that the agent decides *whether* to search — a
net-salary or currency question needs no passages — and *how*: it writes the query, sets the city filter and topic
hint, and can search again when a question spans topics.

**`check_blue_card_salary`** — checks a salary against the EU Blue Card thresholds (§ 18g AufenthG) for a given year.
Arguments: `gross_annual_salary_eur`, `year`, and flags for shortage occupation, recent graduate and IT specialist
without degree; output: both the general and the reduced threshold with met/gap, an eligibility verdict for the
described case, the legal basis and warnings when the salary is within €1,500 of a threshold.

**`calculate_deadline`** — turns an event date into the legal or recommended deadline, skipping weekends and the
regional public holidays of the given city. Arguments: `event` (moved in, residence-permit expiry, visa expiry,
child birth, Elterngeld application, job start), `event_date`, optional `city`; output: the deadline, days
remaining, urgency, whether it is statutory, the legal basis and a plain-language description.

**`estimate_net_salary`** — estimates German take-home pay for 2026 from the income-tax tariff (§ 32a EStG) and the
social-insurance rates. Arguments: `gross_annual_eur`, `tax_class` 1–6, `federal_state`, `church_member`,
`has_children`, `health_insurance` public/private (and premium), `age`; output: monthly and annual net with a full
deduction breakdown and an "estimate ±3 %" warning.

**`convert_currency`** — converts an amount with the ECB reference rate from the free Frankfurter API (mirrors,
caching, dated rates back to 1999). Arguments: `amount`, `from_currency`, `to_currency`, `period` (one-off, monthly,
annual), `gross_or_net`, optional `on_date`; output: the converted amount, rate and rate date, monthly/annual
equivalents, and warnings that the figure is nominal (no cost-of-living or tax adjustment) and that banks add a
spread.

## 2. Grounding check

After the agent has written its answer, a second, small model reads it next to the passages and tool results of
that turn and lists every concrete claim they do not support; those claims are removed or labelled *"Not from the
knowledge base"* before the answer is shown, with the original kept behind an expander. A deterministic figure guard
protects against an over-zealous judge: a flagged claim whose numbers, dates or durations all appear in the evidence
is kept, and a rewrite that would drop a supported figure is rejected — so no correct threshold or deadline is ever
deleted by mistake.

![Grounding check](docs/img/grounding_check.png)

## 3. Guardrails and out-of-domain questions

Five checks run before the model sees a message: length, redaction of personal data (tax ID, IBAN, passport and
phone numbers, e-mail addresses are replaced by placeholders), prompt-injection attempts in English or German,
requests for help with fraud such as registering at an address you do not live at or forging documents, and a topic
gate for questions unrelated to German bureaucracy. A message that fails a check is answered with a fixed text and the
model is never called, so a refusal cannot be talked around; the agent itself is rate-limited per session.

![Guardrails](docs/img/guardrails.png)

Questions that are on topic but outside the knowledge base — Kindergeld, citizenship, Elterngeld — pass the
guardrails and are handled by the retrieval itself: when every returned passage is a weak match, the search tool
tells the model to say the knowledge base does not cover the question and to name the responsible authority, without
stating any amount, condition or procedure; the grounding check then removes anything the model added from memory
anyway.

![Not covered by the knowledge base](docs/img/not_covered.png)

## 4. Evaluation

![Evaluation metrics](docs/img/evaluation_metrics.png)

The harness runs on a labelled set of 23 questions (`data/eval/questions.yaml`): each carries the document that
answers it, the tools that must and must not be called with their expected arguments, and whether the knowledge base
covers it at all — three questions are deliberately outside it. It has two halves: a **retrieval-side** evaluation
that runs only the retriever, and an **end-to-end** evaluation in which the real agent answers every question exactly
as in the app, scored by two deterministic checks and one LLM-judged metric.

**Hit-rate@5** is the share of questions whose expected document appears among the five passages returned. It is
computed for four retriever configurations — dense only, BM25 only, hybrid, hybrid + reranker — from identical query
variants and embeddings, so the bars differ only by retrieval method.

**MRR** (mean reciprocal rank) rewards how high the first expected document sits: rank 1 scores 1.0, rank 2 scores
0.5, not found scores 0, averaged over the questions. It separates "found somewhere in the top five" from "found at the
top", which is what the model actually reads first.

**Tool selection accuracy** is the share of questions where every expected tool was called, no forbidden tool was
called, and the expected arguments were present (numbers compared numerically, strings case-insensitively). It is
fully deterministic and catches both missed calculations and misread inputs such as a gross salary passed as
"unknown".

**Faithfulness** uses RAGAS: the final answer is split into statements and each is checked, by natural-language
inference, against the passages and tool results the agent retrieved for that question — and only against those, so
a statement that is true in the real world but absent from the sources counts as unsupported. The judge is
`gemini-2.5-flash`, a different model family from the answering `gpt-4o-mini`, so it is not lenient towards its own
phrasing. Only *factual* statements are scored; statements that merely refer the user to an authority or say what the
knowledge base lacks are labelled *referral*/*meta* and shown but not scored, because an honest answer to an uncovered
question consists of exactly such sentences.

**Not-covered handling** counts, for the three questions outside the knowledge base, whether the app behaved as
designed: the search reported weak coverage, the answer cited no passage, and the judge found no unsupported factual
statement in it. Saying "not covered" and then listing amounts or application steps from memory therefore fails,
while naming the responsible authority never counts against the answer. One of the three is a *hard* probe the
knowledge base mentions in passing, where the app may quote that passage as long as it adds nothing beyond it.

### Results (run of 17 September 2026, $0.25)

| Hit-rate@5 | MRR | Mean faithfulness | Tool selection | Not-covered handled |
|:---:|:---:|:---:|:---:|:---:|
| **100 %** (20/20) | **0.97** | **0.95** (probes 0.91 · others 0.99) | **21 / 23** | **3 / 3** |

![Retrieval results](docs/img/results_retrieval.png)

![Faithfulness results](docs/img/results_faithfulness.png)

Beyond the default set, nine further not-covered questions in two robustness sets (Elterngeld, Hundesteuer in German,
pension refund, Arbeitslosengeld with a tempting salary figure, tax-return processing time, Führungszeugnis) were all
handled, which is what gives us confidence that the behaviour generalises rather than being fitted to the three probes.

**Why the remaining misses are acceptable.** Both tool-selection failures are the same pattern: for Q11 and Q24 the
calculators answered the whole question (Blue Card threshold, deadline, net salary), the model skipped the
knowledge-base search that the label requires, and the answers were still complete and fully grounded — the one real
slip is Q24's `gross_or_net='unknown'` where the user had written "gross", a misread input that the metric exists to
catch. The five faithfulness scores below 1.0 come from three kinds of strictness rather than from invented facts:
Q18 is a deliberately hard probe on the Opportunity Card, where the knowledge base has one sentence and the judge
marks the prompt-mandated cost-of-living caveats as "unsupported"; Q07 applied the document list for the *initial*
student permit to its *extension*, a reasonable inference the sources do not state; and Q16, Q26 and Q31 each lose one
statement to a small inference the sources do not spell out — that meeting the Blue Card threshold also satisfies
the family-reunification income condition, that cover starting on day one is "continuous", that a bank may not refuse
for a missing SCHUFA *history* when the passage says a poor SCHUFA *score*. In none of the 23 answers did a number,
date or condition reach the user that was not in a passage or tool result, which is the property the metric exists
to protect.

## 5. Limitations and future scope

The net-salary estimate is an approximation (±3 %) that states its assumptions with every result, the ECB publishes
reference rates for about 30 currencies so other currencies are refused, and currency conversions are nominal: the
app has no cost-of-living data and no foreign tax rules, so it will not say whether a German offer leaves you better
off than a salary abroad. The knowledge base is deliberately small and topic-bound; questions on Kindergeld,
Elterngeld, citizenship or unemployment benefit are answered with a referral, and the grounding check, being a small
model, occasionally lets an inferred detail through (Q07) or is stricter than the sources warrant.

Planned next steps: LangSmith tracing for per-turn cost and latency; growing the knowledge base from
`data/sources.yaml` with the fetch script and adding the family-benefit topics the probes showed users ask about;
a Qdrant backend behind the same retriever interface; and scheduled re-runs of the evaluation to catch regressions
when thresholds change each January.

---

*Not legal or tax advice. Data from OpenStreetMap © OpenStreetMap contributors (ODbL).*
