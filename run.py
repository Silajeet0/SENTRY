"""
run.py — entrypoint. Replace your existing run.py with this.

Environment variables to set before running:
    LLM_PROVIDER=nvidia          (or openai, anthropic, ollama)
    LLM_MODEL=meta/llama-3.3-70b-instruct
    LLM_API_KEY=your_key_here
    LLM_BASE_URL=https://integrate.api.nvidia.com/v1

No other config needed. No Agent-E. No browser agent.
"""
from dotenv import load_dotenv
load_dotenv()

from main_driver import run_pipeline

if __name__ == "__main__":
    #neurips
    '''run_pipeline(
        proceeding_url="https://papers.nips.cc/paper_files/paper/2025",
        conference="NeurIPS",
        year="2025",
        max_papers=30,
        resume_from=0,
        delay=10
    )

    #ieee-icdm
    run_pipeline(
        proceeding_url="https://ieeexplore.ieee.org/xpl/conhome/11391637/proceeding",
        conference="IEEE-ICDM",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )'''

    run_pipeline(
        proceeding_url="https://papers.nips.cc/paper_files/paper/2025",
        conference="NeurIPS",
        year="2025",
        max_papers=None,
        resume_from=0,
        delay=10
    )
