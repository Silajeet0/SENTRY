"""
run.py — entrypoint. Replace your existing run.py with this.

Environment variables to set before running:
    LLM_PROVIDER=nvidia          (or openai, anthropic, ollama)
    LLM_MODEL=openai/gpt-oss-20B
    LLM_API_KEY=your_key_here
    LLM_BASE_URL=https://integrate.api.nvidia.com/v1

No other config needed. No Agent-E. No browser agent.
"""
from dotenv import load_dotenv
load_dotenv()

from main_driver import run_pipeline

if __name__ == "__main__":
    '''
    #ieee-icdm
    run_pipeline(
        proceeding_url="https://ieeexplore.ieee.org/xpl/conhome/11391637/proceeding",
        conference="IEEE-ICDM",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )

    #neurips
    run_pipeline(
        proceeding_url="https://papers.nips.cc/paper_files/paper/2025",
        conference="NeurIPS",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )

    #acl
    run_pipeline(
        proceeding_url="https://aclanthology.org/events/acl-2025/",
        conference="ACL",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )

    #acm_kdd
    run_pipeline(
        proceeding_url="https://dl.acm.org/doi/proceedings/10.1145/3770854",
        conference="ACM_KDD",
        year="2026v1",
        max_papers=None,
        resume_from=0,
        delay=10
    )

    #IEEE-CVPR
    run_pipeline(
        proceeding_url="https://ieeexplore.ieee.org/xpl/conhome/11091818/proceeding",
        conference="IEEE-CVPR",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )
    

    #acm_sigcomm
    run_pipeline(
        proceeding_url="https://dl.acm.org/doi/proceedings/10.1145/3718958",
        conference="ACM_SIGCOMM",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )

    #acm_ccs
    run_pipeline(
        proceeding_url="https://dl.acm.org/doi/proceedings/10.1145/3719027",
        conference="ACM_CCS",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )'''

    #icml
    run_pipeline(
        proceeding_url="https://openreview.net/group?id=ICML.cc/2026/Conference#tab-accept-spotlight",
        conference="ICML",
        year="2026",
        max_papers=None,
        resume_from=0,
        delay=10
    )