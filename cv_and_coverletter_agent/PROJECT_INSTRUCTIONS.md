# CV/Cover Letter Agent - Step-by-Step Build Guide
*Personal reference for guiding Copilot through project development*

## 🎯 Project Mission
Build a tool that:
1. Stores a structured, verified database of the user's real experience
2. Learns the user's writing style from samples
3. Parses a job ad (or many) into structured requirements
4. Generates a tailored CV + cover letter
5. Verifies the draft against facts and style before returning it, retrying if it fails

## 🧭 Where to vibe-code freely vs. where to slow down
Phases 0–3 are mostly plumbing — good candidates for fast, Copilot-driven iteration.
**Phase 4 onward is the harness/loop core** — write the logic yourself in plain English or pseudocode first, then have Copilot implement it. This is marked explicitly below.

---

## 📦 PHASE 0: Environment Setup

### Required Packages
```bash
pip install openai             # Deepseek API via OpenAI-compatible async client
pip install pydantic           # schemas for facts, style profile, job ad, evaluator output
pip install pyyaml             # facts database file format
pip install python-dotenv      # environment variables
pip install click              # CLI

pip install pytest pytest-asyncio pytest-mock
pip install black ruff
```

### Environment Variables (.env file)
```bash
DEEPSEEK_API_KEY=your_api_key
```

---

## 📋 PHASE 1: Facts Database
**Goal**: A single, structured, verified source of truth for everything that can appear in a generated document.

### Step 1.1: Define the Schema
**Guide Copilot to build**: `src/facts/facts_schema.py`

**What to tell Copilot**:
```
"Create Pydantic models for a personal facts database:
- FactsEntry: id, category (experience/project/skill/education/certification),
  title, description, technologies (list[str]), evidence_url (optional),
  start_date, end_date (optional)
- FactsDatabase: list[FactsEntry], loaded from a YAML file
- Include a validator that ensures every entry has a unique id"
```

### Step 1.2: Populate and Load
**Guide Copilot to build**: `src/facts/facts_loader.py` + a real `facts.yaml`

Populate this yourself with your actual, verifiable experience (Plejd internship, NuvioTV contribution, home automation projects, education) — this file is the ground truth the whole system depends on. Don't let Copilot invent placeholder entries you forget to replace.

**Validation checklist**:
- [ ] Every entry has a unique id
- [ ] Every claim you'd want in a CV traces to exactly one entry
- [ ] No entry contains vague/unverifiable claims ("improved performance" without a number, unless that's genuinely the honest description)

---

## ✍️ PHASE 2: Style Profile Extraction
**Goal**: Learn how you actually write, once, rather than re-reading samples every generation.

### Step 2.1: Build the Extractor
**Guide Copilot to build**: `src/style/style_extractor.py`

**What to tell Copilot**:
```
"Create a one-time style extractor that:
- Takes a list of the user's past writing samples (cover letters, free-written messages)
- Sends them to Deepseek with a prompt asking it to analyze and return structured
  style attributes: average sentence length + variance, tone (formal/casual scale),
  recurring phrases to preserve, recurring phrases to avoid, typical paragraph structure
- Returns a StyleProfile Pydantic model
- Saves the result to a local file so this only needs to run once (or when the
  user wants to refresh it with new samples)"
```

### Step 2.2: Define the Schema
```
"Create a StyleProfile Pydantic model with fields for: tone_description,
avg_sentence_length, sentence_length_variance, characteristic_phrases (list[str]),
phrases_to_avoid (list[str]), structural_notes (str), anchor_snippets (list[str])

anchor_snippets should preserve 3-4 raw, high-quality paragraphs extracted from
the user's writing samples so later generation can better match voice and cadence."
```

---

## 📄 PHASE 3: Job Ad Parsing
**Goal**: Turn raw job ad text/URL into structured requirements.

### Step 3.1: Build the Parser
**Guide Copilot to build**: `src/job_ads/parser.py`

**What to tell Copilot**:
```
"Create an async job ad parser that:
- Takes raw job ad text (or fetches from a URL) as input
- Sends it to Deepseek with a prompt to extract: company name, role title,
  required_skills (list[str]), nice_to_have_skills (list[str]),
  tone_signals (str — e.g. 'startup casual' vs 'corporate formal'), key_responsibilities
- Returns a structured JobAd Pydantic model
- Handles malformed/empty input gracefully"
```

### Step 3.2: Batch Mode
**Guide Copilot to build**: `src/job_ads/batch.py`
```
"Create a function that accepts a list of job ad texts/URLs and parses them
concurrently using asyncio.gather(), returning a list of JobAd models"
```

---

## 🔑 PHASE 4: Generation — this is where harness engineering starts
**Goal**: Produce a first draft, grounded strictly in the facts database.

**Before prompting Copilot**: write out, in your own words, exactly what context the generator should receive (which facts entries, the style profile, the parsed job ad) and what it's explicitly forbidden from doing (inventing claims). This is the harness design — the interface between your ground-truth data and the LLM call. Do this part deliberately.

### Step 4.1: Build the Generator
**Guide Copilot to build**: `src/generation/cv_generator.py` and `src/generation/cover_letter_generator.py`

**What to tell Copilot** (after you've drafted your own version of this):
```
"Create an async cover letter generator that:
- Takes FactsDatabase, StyleProfile, and JobAd as input
- Selects the most relevant facts entries for this specific job ad (don't dump the whole database into the prompt)
- Constructs a prompt that instructs Deepseek to write ONLY using the selected facts entries, in the tone/structure described by the style profile
- Explicitly instructs: 'Do not state any skill, achievement, or experience not present in the provided facts. If something isn't covered, do not invent it.'
- Returns raw draft text"
```

---

## 🔁 PHASE 5: Evaluation — this is where loop engineering starts
**Goal**: Catch hallucinations, poor job-fit, and AI-sounding prose before the draft reaches the user.

**Before prompting Copilot**: decide your own pass/fail criteria for each check below and roughly how strict each should be. This is the judgment call that makes the loop actually useful instead of just theater.

### Step 5.1: Build the Evaluator
**Guide Copilot to build**: `src/evaluation/evaluator.py`

**What to tell Copilot**:
```
"Create an async evaluator that takes a generated draft, the FactsDatabase, the
JobAd, and the StyleProfile, and runs four checks:
1. Hallucination check — a separate Deepseek call comparing each specific claim in
   the draft against the facts database, flagging anything unsupported
2. Requirement coverage — does the draft address the job ad's required_skills?
3. Style match — does sentence length variance and tone match the style profile?
4. AI-tone check — flag generic filler phrases and uniform paragraph rhythm
5. Corporate Cliche and Boilerplate Filter (deterministic Python check) — if the
  draft contains any banned words such as delve, tapestry, testament, pioneer,
  bespoke, seamlessly, foster, ultimate, furthermore, or moreover, the evaluator
  automatically fails the draft
Return an EvaluationResult Pydantic model: passed (bool), issues (list[str]),
per-check scores"
```

---

## ♻️ PHASE 6: Loop Orchestration
**Goal**: Tie generation and evaluation into a bounded retry loop.

### Step 6.1: Build the Orchestrator
**Guide Copilot to build**: `src/loop/orchestrator.py`

**What to tell Copilot**:
```
"Create an orchestrator that:
- Runs the generator to produce a draft
- Runs the evaluator on the draft
- If evaluation fails, constructs a targeted correction note from the specific
  issues found, and re-runs the generator with that note included
- Repeats up to MAX_RETRIES (default 3)
- If still failing after max retries, returns the best-scoring draft along with
  the unresolved issues, rather than silently returning a bad draft
- Logs each iteration's evaluation result"
```

**This is the core loop-engineering artifact of the project** — the generate → check → retry → stop cycle, with a clear bounded stop condition. Make sure you can explain every line of this file.

---

## ✨ PHASE 7: Burstiness / Humanization Pass
**Goal**: A final rewrite pass targeting natural rhythm.

### Step 7.1: Build the Rewrite Pass
**Guide Copilot to build**: `src/generation/humanize_pass.py`
```
"Create a rewrite pass that takes a draft flagged as too uniform by the AI-tone
check, and rewrites it with explicit instructions to vary sentence length
noticeably, cut any sentence that could appear in a generic cover letter,
preserve all facts and style-profile constraints from the original draft, and
directly match the sentence-length variance, cadence, and structural rhythm
observed in the anchor_snippets captured in StyleProfile"
```
This only runs when the AI-tone check in Phase 5 flags an issue — not unconditionally on every draft.

---

## 📥 PHASE 8: Batch Mode (multiple job ads)
**Goal**: Run the full loop across many job ads, ranked by fit.

### Step 8.1: Build Batch Pipeline
**Guide Copilot to build**: `src/pipeline/batch.py`
```
"Create a pipeline that takes a list of JobAd objects, runs the full
generate-evaluate-retry loop for each concurrently (bounded concurrency, e.g.
max 3 at once to respect rate limits), and returns results ranked by a fit
score computed from facts-to-requirements overlap"
```

---

## 🚀 PHASE 9: Main Orchestrator / CLI
**Guide Copilot to build**: `src/main.py`
```
"Create a CLI entrypoint using click that:
- generate command: takes a job ad URL or text file, runs the full loop, outputs
  CV + cover letter as markdown
- batch command: takes a directory of job ad files, runs batch mode, outputs
  ranked results
- refresh-style command: re-runs style extraction on new writing samples
- Includes clear logging at each step"
```

---

## 🧪 PHASE 10: Testing Strategy

**Phase 1 (facts)**: verify schema validation catches duplicate ids and missing required fields.

**Phase 2 (style)**: verify extractor produces a StyleProfile from sample text without crashing on short/empty input.

**Phase 3 (job ads)**: mock the Deepseek API, verify parser returns correct structured fields; test batch concurrency.

**Phase 4-5 (generation/evaluation) — most important tests**:
```
"Write tests that specifically verify the evaluator correctly FLAGS a draft
containing a fabricated claim not present in the facts database — this is the
single most important test in the project"
```

**Phase 6 (loop)**: test that retry count is bounded and the loop terminates even when evaluation keeps failing.

---

## ⚠️ Common Pitfalls to Avoid
1. **Don't skip the facts database step** — without it, there's nothing for the evaluator to check hallucinations against
2. **Don't let the generator and evaluator share a prompt/call** — self-grading doesn't catch anything
3. **Don't dump raw writing samples into every prompt** — extract the style profile once
4. **Don't retry unboundedly** — always have a max retry count and a clear "give up and surface the issue" path
5. **Don't vibe-code Phase 4-6 without writing your own version of the logic first** — this is the part of the project you're actually trying to learn

---

## 📊 Validation Checklist
- [ ] Facts database validated, no placeholder/invented entries
- [ ] Style profile extracted from real writing samples, not generic tone presets
- [ ] Job ad parser produces structured, correct fields
- [ ] Generator never introduces claims outside the facts database (verify manually on first few runs)
- [ ] Evaluator catches at least one deliberately-inserted fabricated claim in testing
- [ ] Loop terminates correctly both on success and on repeated failure
- [ ] Batch mode respects rate limits (bounded concurrency)
- [ ] Final output reads distinctly less "AI-generated" than a single-shot draft — verify this against real readers, not just your own judgment

---

## 🔄 Iterative Development Approach
1. Phases 0-3: move fast, Copilot-driven, light review
2. Phases 4-6: write your own logic first, then implement with Copilot, review line by line
3. Test immediately after each phase
4. Don't build everything at once — get Phases 1-6 solid before adding batch mode and humanization
