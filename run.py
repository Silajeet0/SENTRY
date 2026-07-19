"""
run.py — entrypoint. Replace your existing run.py with this.

Environment variables to set before running:
    LLM_PROVIDER=nvidia          (or openai, anthropic, ollama)
    LLM_MODEL=openai/gpt-oss-20B
    LLM_API_KEY=your_key_here
    LLM_BASE_URL=https://integrate.api.nvidia.com/v1

    OPENREVIEW_USERNAME=your_openreview_email    (only needed for venue_id entries)
    OPENREVIEW_PASSWORD=your_openreview_password

No other config needed. No Agent-E. No browser agent.

Each entry in conferences_to_run is EITHER:
    - a "proceeding_url" entry  → routed through main_driver.run_pipeline,
      which fetches/scrapes the proceedings page then hands off to
      pipeline.run_pipeline with a resolved input_links_path.
    - a "venue_id" entry        → routed straight to pipeline.run_pipeline's
      OpenReview API path — no scraping, no main_driver involvement at all,
      since OpenReview needs neither HTML fetching nor a browser.
"""
from dotenv import load_dotenv
load_dotenv()

from main_driver import run_pipeline as run_scraped_pipeline
from pipeline import run_pipeline as run_openreview_pipeline

if __name__ == "__main__":
    # List of all conference pipeline configurations
    '''conferences_to_run = [
        {
            "proceeding_url": "https://ieeexplore.ieee.org/xpl/conhome/11391637/proceeding",
            "conference": "IEEE-ICDM",
            "year": "2025"
        },
        {
            "proceeding_url": "https://papers.nips.cc/paper_files/paper/2025",
            "conference": "NeurIPS",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11368763/proceeding",
            "conference": "IEEE-FOCS",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/10973274/proceeding",
            "conference": "IEEE-HRI",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11443115/proceeding",
            "conference": "IEEE-ICCV",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11127273/proceeding",
            "conference": "IEEE-ICRA",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11220258/proceeding",
            "conference": "IEEE-ISMAR",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11186120/proceeding",
            "conference": "IEEE-LICS",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11524112/proceeding",
            "conference": "IEEE-PERCOM",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/11023178/proceeding",
            "conference": "IEEE-SP",
            "year": "2025"
        },
        {
            "proceeding_url" : "https://ieeexplore.ieee.org/xpl/conhome/10937339/proceeding",
            "conference": "IEEE-CV",
            "year": "2025"
        },
        {
            "proceeding_url": "https://aclanthology.org/events/acl-2025/",
            "conference": "ACL",
            "year": "2025"
        },
        {
            "proceeding_url": "https://ieeexplore.ieee.org/xpl/conhome/11091818/proceeding",
            "conference": "IEEE-CVPR",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3770854",
            "conference": "ACM_KDD",
            "year": "2026v1"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3718958",
            "conference": "ACM_SIGCOMM",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3719027",
            "conference": "ACM_CCS",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3736252",
            "conference": "ACM_EC",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3696630",
            "conference": "ACM_FSE",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3695053",
            "conference": "ACM_ISCA",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3680207",
            "conference": "ACM_MOBICOM",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3732772",
            "conference": "ACM_PODC",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3722234",
            "conference": "ACM_PODS",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3721238",
            "conference": "ACM_SIGGRAPH",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3726302",
            "conference": "ACM_SIGIR",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3722212",
            "conference": "ACM_SIGMOD",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3731569",
            "conference": "ACM_SOSP",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3717823",
            "conference": "ACM_STOC",
            "year": "2025"
        },
        {
            "proceeding_url": "https://dl.acm.org/doi/proceedings/10.1145/3746058",
            "conference": "ACM_UIST",
            "year": "2025"
        },

        # ── OpenReview conferences — venue_id routes straight to
        #    pipeline.run_pipeline's API path, no proceeding_url needed. ──
        {
            "venue_id": "ICML.cc/2026/Conference",
            "conference": "ICML",
            "year": "2026",
            "skip_venue_keywords": ["Workshop", "Tutorial"],
        },
        {
            "venue_id": "ICLR.cc/2026/Conference",
            "conference": "ICLR",
            "year": "2026",
            "skip_venue_keywords": ["Workshop", "Tutorial"],
        },
    ]

    # Dynamically loop and execute each pipeline configuration sequentially
    for conf in conferences_to_run:
        print(f"Starting pipeline processing for: {conf['conference']} ({conf['year']})...")

        if "venue_id" in conf:
            run_openreview_pipeline(
                conference=conf["conference"],
                year=conf["year"],
                venue_id=conf["venue_id"],
                skip_venue_keywords=conf.get("skip_venue_keywords"),
                include_only_venue_keywords=conf.get("include_only_venue_keywords"),
                max_papers=None,
                resume_from=0,
                delay=10,
            )
        else:
            run_scraped_pipeline(
                proceeding_url=conf["proceeding_url"],
                conference=conf["conference"],
                year=conf["year"],
                max_papers=None,
                resume_from=0,
                delay=10
            )'''
    
    run_scraped_pipeline(
                proceeding_url="https://dl.acm.org/doi/proceedings/10.1145/3770854",
                conference="ACM_KDD",
                year="2026v1",
                max_papers=None,
                resume_from=0,
                delay=10
            )
    
