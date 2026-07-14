"""Batch helpers for parsing multiple job ads concurrently."""

from __future__ import annotations

import asyncio
from typing import Sequence

from .parser import JobAdParser
from .schema import JobAd


async def parse_job_ads_batch(parser: JobAdParser, raw_ads: Sequence[str]) -> list[JobAd]:
    """Parse many job ads concurrently and return structured models."""

    tasks = [parser.parse(raw_ad) for raw_ad in raw_ads]
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))
