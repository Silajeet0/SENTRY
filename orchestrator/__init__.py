"""
AEGIS orchestrator — the agentic layer on top of the deterministic pipeline.

Nothing in AEGIS's actual scraping/extraction code changed philosophy: the
four-tier scraper cascade and the single-LLM-call-per-paper extractor are
still fully deterministic. This package adds one new thing on top of that —
an LLM-driven tool-calling loop that decides *which* conferences to run,
*in what order*, with *what track filters*, and *when to retry failures* —
by calling the same main_driver / pipeline functions a human would call by
hand.

Submodules:
    conference_catalog  — resolve_conference_url / validate_url / detect_structure
    registry             — thread-safe in-memory state for in-flight runs
    runner               — starts main_driver.run_pipeline / pipeline.run_pipeline
                            in background threads
    tools                — the JSON-in/JSON-out functions + schemas exposed to the LLM
    agent                — the tool-calling loop itself (Orchestrator class)
"""
