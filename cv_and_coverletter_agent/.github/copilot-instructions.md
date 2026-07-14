# CV/Cover Letter Agent - Copilot Instructions

## 🎯 Project Overview
Building an agent that takes one or more job ads as input and generates a tailored CV and cover letter, using a structured personal facts database and a learned style profile so the output reads like the user actually wrote it — not a generic AI draft. Every generated claim must be traceable to the facts database (no invented experience), and every draft passes through a separate evaluator before being returned.

## 🏗 Tech Stack
- **LLM calls**: Deepseek API via the OpenAI-compatible async client
- **Facts store**: structured JSON/YAML file(s) validated with **Pydantic** models — not a database at this stage, this is a personal tool
- **Style profile**: extracted once from sample writing (past CVs, cover letters, free-written messages), stored as a structured profile (tone, sentence-length distribution, recurring phrases, structural habits), and including `anchor_snippets: list[str]` with 3-4 raw high-quality paragraphs from the user's own writing so voice and cadence can be preserved consistently
- **Job ad parsing**: LLM-based extraction into a structured schema (requirements, keywords, company info, tone signals)
- **Output**: Markdown, later converted to docx/PDF for actual submission
- **CLI**: `argparse` or `click` for local use
- **Testing**: pytest with pytest-asyncio

## 🛠 Core Development Rules

### Async/Await for All LLM and I/O Calls
- ALL Deepseek API calls MUST be async (async client)
- Use `asyncio.gather()` when processing multiple job ads in batch mode
- File I/O (facts DB, style profile, job ad files) can stay sync — no need to over-engineer this

### Error Handling Pattern
```python
try:
    result = await some_async_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", extra={"context": "..."})
    raise CustomError(f"Meaningful message") from e
```
- Wrap ALL LLM API calls in try-except (rate limits, timeouts, malformed responses)
- Use specific exception types, not generic `Exception`
- Never let a failed generation silently return an empty/partial CV — fail loudly

### Type Hints and Schemas Required
- ALL function parameters and return values need type hints
- Facts database, style profile, job ad parse, and evaluator output are all **Pydantic models**, not free-form dicts — this is what makes the evaluator's checks possible
- Example:
```python
class FactsEntry(BaseModel):
    id: str
    category: Literal["experience", "project", "skill", "education"]
    description: str
    evidence_url: str | None = None  # e.g. GitHub link, for verifiable claims
```

## 🔑 Harness Rules (the ground-truth layer)

- **The facts database is the single source of truth.** The generator is NEVER allowed to introduce a skill, achievement, or metric that isn't present in the facts database. This is the #1 anti-pattern to prevent.
- **The style profile is data, not prompt text.** Don't paste raw sample letters into every prompt — extract a compact structured summary once (see Phase 2 in PROJECT_INSTRUCTIONS.md) and reuse it. Keeps prompts smaller and outputs more consistent.
- **The job ad parser output is also a Pydantic model.** Downstream steps (generator, evaluator) key off structured fields (`required_skills: list[str]`), not off re-reading raw job ad text every time.
- **Every external input (job ad text, resume samples) goes through validation before hitting the LLM.** Empty input, garbled scraped HTML, non-English postings — handle gracefully, don't crash the pipeline.

## 🔁 Loop Rules (the control layer)

- **Generator and evaluator are separate LLM calls, never the same call grading itself.** The generator writes; a distinct evaluator prompt (or ideally partially deterministic checks) scores it.
- **The evaluator checks, in order of priority:**
  1. Hallucination check — does every specific claim trace back to a `FactsEntry` id?
  2. Requirement coverage — does the draft address the job ad's key requirements?
  3. Style match — does it match the extracted style profile (sentence length variance, tone)?
  4. "AI-sounding" check — flag generic phrases (e.g. "I am excited to leverage my skills"), uniform paragraph rhythm, or anything that reads templated
    5. Corporate cliche and boilerplate filter (deterministic Python check) — if banned words like `delve`, `tapestry`, `testament`, `pioneer`, `bespoke`, `seamlessly`, `foster`, `ultimate`, `furthermore`, or `moreover` appear, fail the draft automatically
- **Retry logic must be bounded.** Define a max retry count (e.g. 3) and a clear stop condition. Never let the loop retry indefinitely on a stubborn failure — log it and surface to the user instead.
- **Each retry gets a targeted correction note**, not a blind "try again" — e.g. "Claim X in paragraph 2 has no matching facts entry, remove or rephrase."

### Anti-Patterns - NEVER Do These
- ❌ Don't let the generator invent achievements, metrics, or skills not in the facts database
- ❌ Don't use the same prompt/call for generation and evaluation
- ❌ Don't dump raw sample cover letters into the prompt as "style reference" — extract a structured profile instead
- ❌ Don't retry without a bounded stop condition
- ❌ Don't hardcode the user's personal data (name, employers, specific achievements) into prompt template strings — load from the facts database file so the template stays reusable
- ❌ Don't skip the hallucination check even for "obviously safe" drafts — this is the one failure mode that actually damages a real application
- ❌ Don't catch generic `Exception` — use specific types
- ❌ Don't hardcode API keys — use environment variables

## 📁 File Organization
```
src/
├── facts/            # facts_schema.py, facts_loader.py, sample facts.yaml
├── style/             # style_extractor.py (one-time), style_profile.py (schema)
├── job_ads/           # parser.py, schema.py
├── generation/        # cv_generator.py, cover_letter_generator.py
├── evaluation/        # evaluator.py (hallucination, coverage, style, AI-tone checks)
├── loop/              # orchestrator.py (generate → evaluate → retry/stop)
├── pipeline/           # batch.py (multi-ad mode)
└── main.py            # CLI entrypoint only
```

## 🔧 Development Workflow
1. **Build one phase at a time** — see PROJECT_INSTRUCTIONS.md for order
2. **Test each module immediately**, especially the evaluator — it's the part most likely to have subtle bugs that let bad output through
3. **Review Copilot's code for the harness/loop sections manually** — these are the parts to actually learn, not just accept
4. Keep functions small, single responsibility
5. Keep files under 1,000 lines
6. No dead code in committed files
