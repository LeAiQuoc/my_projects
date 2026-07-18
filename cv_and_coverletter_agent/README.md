# CV and Cover Letter Agent

This project generates tailored jobapplication Material from structured personal facts, a saved writing-style profile, and a raw job ad. The main goal is not just text generation, but controlled generation: 
the app tries to keep claims grounded in the facts file, match the language of the job ad, and produce a separate review artifact alongside the final cover letter.

At the moment the public demo flow is built around a sanitized sample dataset. A recruiter can clone the repository, add a DeepSeek API key, and run the example job ad immediately.

## What the app does

- Generates a standalone CV in Markdown
- Generates a standalone cover letter as PDF
- Generates a separate scoring and review Markdown file for each cover letter
- Supports a legacy combined CV + cover letter output mode for debugging and iteration
- Matches the output language to the job ad when possible

## Core inputs

- Facts database: `data/facts.yaml`
- Style profile: `data/style_profile.json`
- Job ads: `job_ads/*.txt`

## Customize your data

The repository currently ships with a sanitized demo identity. To use the app for yourself, replace the demo content in these places:

### 1. Fill in your facts database

Edit `data/facts.yaml`.

This file is the main source of truth for what the app is allowed to say about you. Add or replace entries for:

- work experience
- projects
- education
- skills
- certifications

There is also a dedicated `profile-identity` entry near the top of the file. Update that block with your own:

- name
- address
- phone number
- email
- LinkedIn URL

Those fields are used for the contact section in the generated CV PDF.

If you want an example structure, look at `data/facts.example.yaml`.

### 2. Replace the writing samples

Edit or replace the text files inside `samples/`.

These files are used to capture your writing style. Good inputs include:

- older cover letters
- short bios
- freewriting notes
- informal but well-written messages

The more these files sound like your real writing, the better the style profile and final cover letters will feel.

### 3. Refresh the style profile after updating samples

After changing the sample files, rebuild the style profile:

```powershell
python -m src.main refresh-style samples
```

This updates `data/style_profile.json`.

### 4. Add or replace job ads

Put your job ads in `job_ads/` as `.txt` files.

The short commands can read a bare filename like `company_job_ad.txt` and will look for it in `job_ads/` automatically.

## First-time setup

These steps should be enough to run the example end to end on Windows PowerShell.

### 1. Install Python

Use Python 3.10.8 or newer.

### 2. Create and activate a virtual environment

From the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root with at least:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

Optional:

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 5. Run the example

Once the virtual environment is activated, the short commands are available directly in the repo root.

Generate a CV:

```powershell
generate-cv crylio_job_ad.txt
```

Generate a cover letter PDF and its review file:

```powershell
generate-cl crylio_job_ad.txt
```

The short commands look for the file in `job_ads/` when you pass only a filename.

## Expected output

The app writes files into separate output folders:

- `outputs_cv/` stores standalone CV files
- `outputs_cl/` stores standalone cover-letter PDFs and review files
- `outputs/` stores the older combined application-pack output

Example files from the included Crylio demo ad:

- `outputs_cv/cv_crylio_research.pdf`
- `outputs_cl/cover_letter_crylio_research.pdf`
- `outputs_cl/crylio_research_scoring_and_review.md`

## Short command reference

Generate a CV:

```powershell
generate-cv crylio_job_ad.txt
```

Generate a cover letter:

```powershell
generate-cl crylio_job_ad.txt
```

These wrappers call the main CLI for you, so you do not need to type `python -m src.main` for the common workflow.

## Full CLI reference

If you want to control output paths explicitly, you can still use the Python CLI.

Generate CV with explicit output:

```powershell
python -m src.main generate-cv job_ads/crylio_job_ad.txt --output outputs_cv/my_cv.pdf
```

Generate cover letter with explicit output:

```powershell
python -m src.main generate-cl job_ads/crylio_job_ad.txt --output outputs_cl/my_cover_letter.pdf
```

Generate the legacy combined application pack:

```powershell
python -m src.main generate job_ads/crylio_job_ad.txt --output outputs/crylio.md
```

Refresh the style profile from the writing samples:

```powershell
python -m src.main refresh-style samples
```

## Project structure

- `src/` contains the generators, evaluator, parser, CLI, and output rendering code
- `data/` contains the facts database and style profile
- `job_ads/` contains input job ads
- `samples/` contains writing samples used to build or refresh the style profile
- `scripts/` contains PowerShell shortcuts for the common commands

## Notes

- Cover-letter PDF titles are language-aware. Swedish job ads produce `Personligt brev`.
- Output filenames are based on the parsed company name, with a filename-prefix fallback from the job-ad filename when needed.
- Generated output folders are ignored by git in this repo.