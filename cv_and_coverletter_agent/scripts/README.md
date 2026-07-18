This folder holds small command-line helpers for common project flows.
Use `python -m src.main` for the main CLI scaffold.

Helpful facts workflow commands:
- `./scripts/facts.ps1 add` prompts you for one fact entry and appends it to `data/facts.yaml`
- `./scripts/facts.ps1 list` shows the saved entries
- `./scripts/facts.ps1 validate` checks that the YAML still loads cleanly
- `./scripts/generate-cv.ps1 crylio_job_ad.txt` writes a standalone CV into `outputs_cv/`
- `./scripts/generate-cl.ps1 crylio_job_ad.txt` writes a standalone cover-letter PDF into `outputs_cl/`
- `./scripts/generate_outputs.ps1 cv job_ads/crylio_job_ad.txt` writes a standalone CV into `outputs_cv/`
- `./scripts/generate_outputs.ps1 cl job_ads/crylio_job_ad.txt` writes a standalone cover-letter PDF into `outputs_cl/`
