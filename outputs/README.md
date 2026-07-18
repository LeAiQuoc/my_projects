# Outputs

This folder keeps the legacy combined application packs.

New separated outputs are written to:
- `outputs_cv/` for standalone CV markdown
- `outputs_cl/` for standalone cover-letter PDFs

Examples:
- `python -m src.main generate job_ads/job_ad_1_backend.txt --output outputs/job_ad_1_backend.md`
- `python -m src.main batch job_ads --output outputs/batch_results.md`