# German Bureaucracy Navigator — technical guide

Setup, operation and internals. The project overview, architecture and evaluation results are in the
[README](../README.md).

Contents: [Requirements](#1-requirements) · [Installation](#2-installation-windows-powershell) ·
[Build the knowledge base](#3-build-the-knowledge-base-once) · [Run the app](#4-run-the-app) · [Run the tests](#5-run-the-tests) ·
[Run the evaluation](#6-run-the-evaluation) · [Project layout](#7-project-layout) · [Assignment mapping](#8-how-the-pieces-map-to-the-assignment) ·
[Configuration tips](#9-configuration-tips) · [Troubleshooting](#10-troubleshooting) · [Deploy on Streamlit Community Cloud](#11-deploy-on-streamlit-community-cloud)

## 1. Requirements

| Requirement | Notes |
|---|---|
| **Python 3.11 or 3.12** | 3.13 works for most packages but ChromaDB wheels lag behind; 3.11/3.12 is the safe choice. |
| **OpenRouter API key** | Create one at <https://openrouter.ai/keys>. Set it as the system environment variable **`OPENROUTER_API_KEY`** (Windows: *Settings → System → About → Advanced system settings → Environment Variables*). Restart VS Code afterwards so it picks up the variable. Alternatively copy `.env.example` to `.env` and paste the key there. |
| **Credits on OpenRouter** | Chat calls use the model you pick in the sidebar (default `openai/gpt-4o-mini`). Embeddings use `openai/text-embedding-3-small` — indexing the seed knowledge base costs well under €0.01. A full evaluation run costs about $0.25. |
| **Internet access** | For OpenRouter and, for the currency tool, the free Frankfurter API (ECB rates, no key needed). |
| **VS Code + Python extension** | Optional: the *Streamlit* extension adds "Run with Streamlit" to the right-click menu. |

## 2. Installation (Windows, PowerShell)

### Option A — global Python, individual pip commands

```powershell
cd C:\Users\<you>\German_Bureaucracy_navigator
python --version                      # should be 3.11.x or 3.12.x
python -m pip install --upgrade pip

python -m pip install "langchain>=1.0,<2.0"
python -m pip install "langchain-openai>=1.0,<2.0"
python -m pip install "langchain-chroma>=1.0,<2.0"
python -m pip install "langchain-text-splitters>=1.0,<2.0"
python -m pip install "langgraph>=1.0,<2.0"
python -m pip install "openai>=1.50,<3.0"
python -m pip install "chromadb>=1.0,<2.0"
python -m pip install rank-bm25
python -m pip install "streamlit>=1.50,<2.0"
python -m pip install "pydantic>=2.7,<3.0" pyyaml requests tenacity python-dateutil holidays python-dotenv
python -m pip install matplotlib reportlab altair   # Evaluation page + PDF report
python -m pip install ragas datasets  # only to RUN the evaluation (scripts\evaluate.py)
python -m pip install trafilatura     # only for scripts\fetch_sources.py
python -m pip install pytest ruff     # optional: tests / linting

python -c "from langchain.agents import create_agent; import chromadb, streamlit; print('ok')"
```

Make sure the interpreter selected in VS Code (bottom-right status bar, or `Ctrl+Shift+P` →
**Python: Select Interpreter**) is the same `python` you installed into.

### Option B — virtual environment (equivalent, keeps packages isolated)

```powershell
cd C:\Users\<you>\German_Bureaucracy_navigator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # app + evaluation harness + tests; requirements.txt alone runs the app
```

Then select `.venv` as the interpreter in VS Code.

## 3. Build the knowledge base (once)

```powershell
python scripts\ingest.py
```

This reads every markdown file in `data/processed/`, chunks it by headings, embeds the chunks via
OpenRouter and stores them in Chroma (`data/chroma/`). **Stop the Streamlit app before running this
from a terminal** (Chroma's database must not be written by two processes) — or use the
*Rebuild knowledge base* button on the Knowledge Base page, which runs inside the app. The repo ships with
18 hand-compiled seed documents so the app works immediately; when the index is empty the main page builds it
automatically on first load (about a minute, one build at a time).

To add official web pages (list in `data/sources.yaml`):

```powershell
python scripts\fetch_sources.py     # downloads + extracts to data/processed/fetched/*.md
python scripts\ingest.py            # re-index
```

## 4. Run the app

**The file to launch is `streamlit_app.py` in the project root.**

- VS Code: right-click `streamlit_app.py` → **Run with Streamlit**, or
- Terminal (with the venv active): `streamlit run streamlit_app.py`

The app opens at <http://localhost:8501>. The *Knowledge Base* and *Evaluation* pages appear in the sidebar
automatically (Streamlit multipage via the `pages/` folder).

## 5. Run the tests

```powershell
pytest            # 98 tests, no network (fakes for the LLM and the embedder), ~6 s
ruff check navigator pages scripts tests
```

## 6. Run the evaluation

Run it from a terminal with the Streamlit app **closed** (Chroma is single-writer):

```powershell
python scripts\evaluate.py                                   # both halves, default set, ≈ 15 min, ≈ $0.25
python scripts\evaluate.py --only retrieval                  # ≈ 2 min, a few cents
python scripts\evaluate.py --only e2e --run reports\eval\<folder>   # add the e2e half to an existing run
python scripts\evaluate.py --only e2e --questions data\eval\questions_not_covered_v2.yaml   # 3 fresh not-covered probes
python scripts\evaluate.py --only e2e --questions data\eval\questions_not_covered_v3.yaml   # 3 more (tool temptation, partial coverage, German procedure)
python scripts\evaluate.py --ids Q11,Q18 --no-judge          # a few questions, deterministic checks only
python scripts\evaluate.py --model openai/gpt-4o-mini --judge anthropic/claude-haiku-4.5 --workers 3
```

Every run writes `reports/eval/<timestamp>/` with `meta.json` (incl. the run's cost, read from the OpenRouter
key's credit usage just before and just after the run, so it covers agent, judge, embeddings and reranker),
`retrieval.json`, `e2e.json` (per-question answers, tool calls, claims and verdicts), `summary.md` and
`report.pdf`. Open the **Evaluation** page in the app to browse runs, compare two runs, click a bar in the
faithfulness chart to see the details of that question, and download the PDF.

Notes: RAGAS 0.4 imports a Vertex AI class that the current `langchain-community` no longer ships; the harness
registers a harmless stand-in module before importing RAGAS. If RAGAS cannot be imported or fails on your judge
model, the harness switches to a built-in judge that runs the same two-step procedure (`--judge-backend builtin` to
force it, `--judge-backend ragas` to forbid the fallback). A judge that is the same model as the answering model is
allowed but warned about — a model is lenient towards its own phrasing. A tool that fails during a question (a
timeout, say) triggers one automatic re-run of that question; a second failure is reported in the PDF and the page.

The README figures are regenerated with `python docs\make_figures.py` (update the `RESULTS` block after a new run).

## 7. Project layout

```
streamlit_app.py            ← entry point (chat UI)
pages/1_Knowledge_Base.py   ← inventory, rebuild with progress, retrieval playground
pages/2_Evaluation.py       ← browse evaluation runs: retrieval metrics, faithfulness, tool selection, not-covered handling, PDF
ui/                         ← Streamlit rendering (source cards, tool traces) + session state + theme colours
.streamlit/config.toml      ← app theme (project-level; also used on Streamlit Community Cloud)
navigator/
  config.py                 ← loads config/settings.yaml, .env, API key
  agent/    llm.py (ChatOpenAI → OpenRouter) · prompts.py · graph.py (create_agent + streaming runner)
  tools/    5 LangChain @tools returning a uniform JSON ToolResult
  core/     pure business logic (tax tariff, Blue Card rules, deadline rules) — unit-tested
  clients/  frankfurter.py (ECB exchange rates: mirrors, caching, typed errors)
  rag/      loaders · chunking · embeddings (OpenRouter) · vectorstore (Chroma) · hybrid (BM25+RRF) · coverage (query-relative relevance) · rerank · retriever · ingest
  guardrails/ input validation · PII redaction · injection detection · topic gate · output checks · grounding check (LLM judge) · figures (figure guard)
  eval/     questions (YAML loader) · retrieval_eval (hit-rate/MRR × 4 configs) · e2e_eval (agent run + checks) · faithfulness (RAGAS / built-in) · cost · report (summary.md + PDF) · store
config/settings.yaml        ← models, chunk sizes, top-k, rate limits, feature flags
data/processed/*.md         ← knowledge base (markdown with YAML front matter)
data/reference/*.yaml       ← parameter tables for the calculators + glossary (NOT embedded)
data/sources.yaml           ← official URLs for scripts/fetch_sources.py
data/eval/questions.yaml    ← evaluation question set (23) · questions_not_covered_v2/v3.yaml (2 × 3 fresh not-covered probes)
docs/                       ← this guide, make_figures.py, img/ (README figures)
reports/eval/<run>/         ← evaluation runs: meta.json, retrieval.json, e2e.json, summary.md, report.pdf
scripts/                    ← fetch_sources.py, ingest.py, check_openrouter.py, evaluate.py (evaluation harness)
tests/                      ← pytest suite (uses fakes, no network)
requirements.txt            ← runtime dependencies (what Streamlit Community Cloud installs) · requirements-dev.txt adds evaluation, tests, lint
```

## 8. How the pieces map to the assignment

| Requirement | Implementation |
|---|---|
| Knowledge base, embeddings, chunking, similarity search | `data/processed`, `rag/chunking.py` (heading-aware + recursive), `rag/embeddings.py`, `rag/vectorstore.py` (Chroma, cosine) |
| Advanced RAG | Hybrid BM25 + dense search fused with Reciprocal Rank Fusion, bilingual query expansion, jurisdiction metadata filtering, conservative weak-match detection (no keyword overlap AND no semantic standout, or low reranker relevance), Cohere cross-encoder reranking via OpenRouter (`rag/hybrid.py`, `rag/retriever.py`, `rag/coverage.py`, `rag/rerank.py`) |
| ≥ 3 tool calls | 5 tools: 1 retrieval (knowledge base), 3 calculators (Blue Card, deadlines, net salary), 1 live API (currency conversion via ECB rates — nominal only, carries a gross/net flag and cost-of-living warnings) (`navigator/tools/`) |
| Domain prompts | `agent/prompts.py` — bilingual, glossary-injected, citation rules, refusal rules |
| Security | `navigator/guardrails/` — PII redaction, injection & fraud detection, off-topic gate, rate limiting, output citation/PII checks, grounding check + revision with a deterministic figure guard, indirect-injection defence in the prompt |
| LangChain + OpenRouter | `agent/llm.py` (`ChatOpenAI(base_url=openrouter)`), `create_agent` from LangChain 1.x, `@tool` with Pydantic schemas |
| Error handling & validation | typed exceptions (`utils/errors.py`), Pydantic tool inputs, retries and mirror failover for external APIs, readable tool errors the agent can relay |
| UI: context & sources | *Sources* expander with passage, section, relevance and link; `[S1]` citations inline |
| UI: tool results | *Tool calls* expander with arguments, structured rendering (metrics, tables), raw JSON |
| UI: progress | `st.status` steps per tool call, token streaming, progress bars for ingestion |
| Evaluation | `navigator/eval/` + `scripts/evaluate.py` + Evaluation page: hit-rate@5, MRR, RAGAS faithfulness, tool selection, not-covered handling |

## 9. Configuration tips

- **Change the model:** edit `llm.models` in `config/settings.yaml`. Every id must be enabled for your OpenRouter key
  (OpenRouter → Settings → allowed models); the defaults are `openai/gpt-4o-mini`, `anthropic/claude-haiku-4.5`,
  `google/gemini-2.5-flash`, `openai/gpt-4.1-mini`.
- **Reranker:** on by default via OpenRouter's rerank endpoint (`cohere/rerank-4-fast`, about $0.002 per question;
  `cohere/rerank-4-pro` is the stronger option). Set `retrieval.reranker.provider: none` to switch it off, or
  `local` with `BAAI/bge-reranker-v2-m3` after `pip install sentence-transformers torch` for an offline reranker.
  `retrieval.reranker.min_relevance` (0.65) is the third weak-coverage signal — re-calibrate it if you change the reranker
  (covered questions' best passage scored ≥ 0.74, not-covered ≤ 0.62 with `rerank-4-fast`).
- **Embedding model:** `openai/text-embedding-3-small` by default; `openai/text-embedding-3-large` or
  `qwen/qwen3-embedding-8b` are drop-in alternatives (change `embeddings.model`, then rebuild the index).
- **Disable the LLM topic gate** (saves one cheap call per message): `guardrails.topic_gate.enabled: false`.
- **Grounding check + revision** (`guardrails.grounding_check`): `revise: true` (default) rewrites the answer so
  unsupported claims are removed or prefixed "Not from the knowledge base:" and keeps the original behind an expander;
  `revise: false` only shows them as a 🔎 warning; `enabled: false` switches the check off. The figure guard
  (`navigator/guardrails/figures.py`) always runs with the revision.
- **Query expansion** (extra call per retrieval): `retrieval.query_expansion.enabled`.

## 10. Troubleshooting

**`ingest.py` seems stuck after "Collection reset" / the app says the knowledge base is empty.**
Older versions reset the index before calling OpenRouter, so a stalled embedding call left it empty.
The current version first checks the connection (`ping` step), embeds everything with per-batch
progress and only then writes — a failure leaves the old index untouched. To find the cause:

```powershell
python scripts\check_openrouter.py     # validates key, credits, embedding model and chat model
```

Typical results: **HTTP 402** = no OpenRouter credits left; **HTTP 401** = key not picked up (restart
VS Code after setting the environment variable); a timeout = network/proxy problem. If the Streamlit
app was running while you ran `ingest.py` in a terminal, close it and rebuild (or use the *Rebuild* button on the
Knowledge Base page).

**Requests to OpenRouter sometimes hang for exactly the timeout, then the retry succeeds in under a second.**
That is a pooled HTTP connection the server or a proxy closed while idle. The clients drop idle connections
after 10 s and single query embeddings use a 12 s timeout, so a stall costs seconds, not half a minute. If it
persists, check for a corporate proxy/VPN and try `python scripts\check_openrouter.py` a few times in a row.

**"The knowledge base has no specific passage on this question."** Retrieval found only weak matches.
A passage is weak when it shares **no keywords** with the question (no BM25 hit) **and** its similarity
does not *stand out* from the question's similarity to the whole corpus (robust z-score below
`retrieval.coverage_z`, default 2.5 — see `navigator/rag/coverage.py`), **or** — when the reranker runs — its
cross-encoder relevance is below `retrieval.reranker.min_relevance` (default 0.65). The third signal exists because
a single shared word ("children" in an Anmeldung passage) used to make a Kindergeld question look covered. If the
message appears for a covered topic, check the "standout z" and relevance values in the *Retrieval playground*; if the
topic is genuinely missing, add a markdown file to `data/processed/` and rebuild.

**The answer states a fact that is not in the sources (e.g. "typically valid for one year").** The system prompt
forbids this and the grounding check removes or labels such sentences, but a model can still slip one in.
If the fact matters, add it to a markdown file in `data/processed/` with an official source and rebuild — then the
model can cite it instead of guessing.

**A correct number disappeared from an answer after the grounding revision.** The figure guard should prevent this
(a revision may never drop a number, date or duration that appears in the evidence). If it happens, the number was
probably written in a format the normaliser does not recognise — open an issue with the answer and the source.

**New or edited markdown files are not used.** The index is only rebuilt on demand: run
`python scripts\ingest.py` or press *Rebuild knowledge base* on the Knowledge Base page.

**Streamlit logs "Serialization of dataframe to Arrow table was unsuccessful".** A table column mixed numbers and
text. The Evaluation page renders its tables as static, string-typed tables precisely to avoid this; if you add a
table, keep each column to one type.

## 11. Deploy on Streamlit Community Cloud

The app runs unchanged on [Streamlit Community Cloud](https://share.streamlit.io) (free, public URL) — the live
instance is at <https://germanbureaucracynavigator-nv4spe5owjrbx5kkumhffp.streamlit.app/>. What the
code already does for that environment: the API key is read from the environment *or* from Streamlit's
secrets store (`navigator/config.py`); an empty knowledge base is built automatically on first load, because the
Chroma index is not in the repository and the cloud container starts with an empty disk after every restart
(`streamlit_app.py`); `requirements.txt` holds only what the app needs (the evaluation harness lives in
`requirements-dev.txt`); and `.streamlit/config.toml` carries the theme.

**Steps**

1. Push the repository to GitHub (public or private — Community Cloud reads either). `.gitignore` already keeps
   `.env`, `.streamlit/secrets.toml` and `data/chroma/` out.
2. Create an OpenRouter key **for the deployment only** (Settings → Keys) and give it a credit limit — the app is
   public and every visitor's questions run on that key. Enable the models the app uses for that key
   (`openai/gpt-4o-mini`, `openai/text-embedding-3-small`, `cohere/rerank-4-fast`).
3. On <https://share.streamlit.io> → **Create app** → *Deploy a public app from GitHub*: choose the repository and
   branch, set **Main file path** to `streamlit_app.py`, and under **Advanced settings** pick **Python 3.12** and
   paste into **Secrets**:

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-…"
   ```

4. Deploy. The first build installs the requirements (a few minutes); the first page load then builds the
   knowledge base (about a minute) and the app is ready. Secrets and Python version can be changed later under
   the app's *Settings*; a push to the branch redeploys automatically.

**Good to know**

- The container sleeps after a period without visitors and restarts on the next visit; the knowledge base is
  rebuilt on that first visit (a few cents of embeddings). Committing `data/chroma/` would avoid the rebuild but
  ties the repository to one Chroma version and adds binary churn — not worth it at this size.
- Evaluation runs in `reports/eval/` are shown read-only; only the run quoted in the README is committed.
  `scripts/evaluate.py` is meant for a local machine (it needs `requirements-dev.txt` and about 15 minutes).
- The per-session rate limit (`app.rate_limit` in `config/settings.yaml`) is the only usage brake inside the app;
  the credit limit on the OpenRouter key is the hard stop.
- If the logs show `sqlite3` older than 3.35 (Chroma refuses to start), add `pysqlite3-binary` to
  `requirements.txt` and, at the very top of `streamlit_app.py`, `import pysqlite3, sys; sys.modules["sqlite3"] =
  pysqlite3`. Current Community Cloud images ship a recent SQLite, so this is normally not needed.

