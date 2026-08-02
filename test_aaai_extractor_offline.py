"""
Offline smoke test for workflows.link_extractors.aaai_link_extractor —
uses synthetic HTML built to mirror the structure seen in the AAAI
proceedings screenshots, so parsing logic can be validated without
network access to aaai.org / ojs.aaai.org.
"""
from bs4 import BeautifulSoup

from workflows.link_extractors.aaai_link_extractor import (
    _find_volume_links,
    _extract_tracks_from_volume,
)

# --- Synthetic Level 1 page: aaai.org/proceeding/aaai-40-2026/ ---
ROOT_HTML = """
<html><body>
<div class="content">
<p>The proceedings have been published in 3 consecutive issues:</p>
<ul>
<li><a href="https://ojs.aaai.org/index.php/AAAI/issue/view/683">Vol 40 No. 1: AAAI-26 Technical Tracks 1</a>
    <a href="https://ojs.aaai.org/index.php/AAAI/issue/view/683">AAAI Technical Track on Application Domains I</a></li>
<li><a href="https://ojs.aaai.org/index.php/AAAI/issue/view/684">Vol 40 No. 2: AAAI-26 Technical Tracks 2</a>
    <a href="https://ojs.aaai.org/index.php/AAAI/issue/view/684">AAAI Technical Track on Application Domains II</a></li>
<li><a href="https://ojs.aaai.org/index.php/AAAI/issue/view/685">Vol 40 No. 3: AAAI-26 Technical Tracks 3</a>
    <a href="https://ojs.aaai.org/index.php/AAAI/issue/view/685">AAAI Technical Track on Cognitive Modeling &amp; Cognitive Systems</a></li>
</ul>
</div>
</body></html>
"""

# --- Synthetic Level 2 page: ojs.aaai.org/index.php/AAAI/issue/view/683 ---
VOLUME_HTML = """
<html><body>
<div class="sidebar">
<h5>Information</h5>
<a href="/for/readers">For Readers</a>
<h5>For Authors</h5>
</div>
<div class="content">
<p>This issue (volume 40 no. 1) consists of 840 pages and 1 tracks:
AAAI Technical Track on Application Domains I</p>
<p>Published: 2026-03-17</p>
<hr/>
<h2 class="tocSectionTitle">AAAI Technical Track on Application Domains I</h2>
<ul>
  <li>
    <h3><a href="/index.php/AAAI/article/view/36958">Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning</a></h3>
    <div class="authors">Lejun Ai, Yulong Li, Haodong Yi, Jixuan Xie, Yue Wang, Jia Liu, Min Chen, Rui Wang</div>
    <a class="pdf" href="/index.php/AAAI/article/view/36958/39012">PDF</a>
  </li>
  <li>
    <h3><a href="/index.php/AAAI/article/view/36959">AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research</a></h3>
    <div class="authors">Alexandru-Mihai Apostu, et al.</div>
    <a class="pdf" href="/index.php/AAAI/article/view/36959/39013">PDF</a>
    <a class="video" href="https://example.com/video">Video/Poster/Slides</a>
  </li>
  <li>
    <h3><a href="/index.php/AAAI/article/view/36960">Beyond Content: A Comprehensive Speech Toxicity Dataset</a></h3>
    <div class="authors">Zhongjie Ba, et al.</div>
    <a class="pdf" href="/index.php/AAAI/article/view/36960/39014">PDF</a>
  </li>
</ul>
</div>
</body></html>
"""


def test_find_volume_links_skips_subtrack_duplicate_and_matches_vol_pattern():
    soup = BeautifulSoup(ROOT_HTML, "html.parser")
    volumes = _find_volume_links(soup, "https://aaai.org/proceeding/aaai-40-2026/")
    titles = [v["title"] for v in volumes]
    urls = [v["url"] for v in volumes]

    assert len(volumes) == 3, f"expected 3 volumes, got {len(volumes)}: {titles}"
    assert titles == [
        "Vol 40 No. 1: AAAI-26 Technical Tracks 1",
        "Vol 40 No. 2: AAAI-26 Technical Tracks 2",
        "Vol 40 No. 3: AAAI-26 Technical Tracks 3",
    ]
    assert urls == [
        "https://ojs.aaai.org/index.php/AAAI/issue/view/683",
        "https://ojs.aaai.org/index.php/AAAI/issue/view/684",
        "https://ojs.aaai.org/index.php/AAAI/issue/view/685",
    ]
    # The redundant sub-track anchor after each Vol link must NOT create a
    # duplicate/second entry pointing at the same URL.
    assert len(set(urls)) == len(urls)
    print("OK: _find_volume_links — 3 volumes found, sub-track duplicates skipped")


def test_extract_tracks_from_volume_gets_article_view_links_not_pdf():
    soup = BeautifulSoup(VOLUME_HTML, "html.parser")
    tracks = _extract_tracks_from_volume(
        soup,
        "https://ojs.aaai.org/index.php/AAAI/issue/view/683",
        "Vol 40 No. 1: AAAI-26 Technical Tracks 1",
    )

    assert len(tracks) == 1, f"expected 1 track, got {len(tracks)}: {tracks}"
    track = tracks[0]

    # Sidebar boilerplate ("Information", "For Authors") must NOT have been
    # picked up as the track title.
    assert track["track_title"] == "AAAI Technical Track on Application Domains I"

    expected_links = [
        "https://ojs.aaai.org/index.php/AAAI/article/view/36958",
        "https://ojs.aaai.org/index.php/AAAI/article/view/36959",
        "https://ojs.aaai.org/index.php/AAAI/article/view/36960",
    ]
    assert track["paper_links"] == expected_links, track["paper_links"]

    # PDF/galley links (.../article/view/36958/39012) must be excluded.
    for link in track["paper_links"]:
        assert link.count("/") == 6 or link.split("/view/")[1].isdigit(), link

    print("OK: _extract_tracks_from_volume — correct track title, article "
          "links only (no PDF/galley links, no sidebar noise)")


MULTI_TRACK_VOLUME_HTML = """
<html><body>
<div class="content">
<h2 class="tocSectionTitle">AAAI Technical Track on Computer Vision I</h2>
<ul>
  <li><h3><a href="/index.php/AAAI/article/view/1001">Paper A</a></h3></li>
  <li><h3><a href="/index.php/AAAI/article/view/1002">Paper B</a></h3></li>
</ul>
<h2 class="tocSectionTitle">AAAI Technical Track on Computer Vision II</h2>
<ul>
  <li><h3><a href="/index.php/AAAI/article/view/2001">Paper C</a></h3></li>
</ul>
</div>
</body></html>
"""

NO_HEADING_VOLUME_HTML = """
<html><body>
<div class="content">
<ul>
  <li><h3><a href="/index.php/AAAI/article/view/5001">Paper X</a></h3></li>
  <li><h3><a href="/index.php/AAAI/article/view/5002">Paper Y</a></h3></li>
</ul>
</div>
</body></html>
"""


def test_extract_tracks_from_volume_handles_multiple_tracks_on_one_issue():
    soup = BeautifulSoup(MULTI_TRACK_VOLUME_HTML, "html.parser")
    tracks = _extract_tracks_from_volume(
        soup, "https://ojs.aaai.org/index.php/AAAI/issue/view/699", "Vol 40 No. 6"
    )
    assert len(tracks) == 2, tracks
    assert tracks[0]["track_title"] == "AAAI Technical Track on Computer Vision I"
    assert tracks[0]["paper_links"] == [
        "https://ojs.aaai.org/index.php/AAAI/article/view/1001",
        "https://ojs.aaai.org/index.php/AAAI/article/view/1002",
    ]
    assert tracks[1]["track_title"] == "AAAI Technical Track on Computer Vision II"
    assert tracks[1]["paper_links"] == [
        "https://ojs.aaai.org/index.php/AAAI/article/view/2001",
    ]
    print("OK: multi-track issue page → 2 separate tracks with correct papers")


def test_extract_tracks_from_volume_falls_back_when_no_heading_found():
    soup = BeautifulSoup(NO_HEADING_VOLUME_HTML, "html.parser")
    tracks = _extract_tracks_from_volume(
        soup, "https://ojs.aaai.org/index.php/AAAI/issue/view/700", "Vol 40 No. 7"
    )
    assert len(tracks) == 1, tracks
    assert tracks[0]["track_title"] == "Vol 40 No. 7"
    assert tracks[0]["paper_links"] == [
        "https://ojs.aaai.org/index.php/AAAI/article/view/5001",
        "https://ojs.aaai.org/index.php/AAAI/article/view/5002",
    ]
    print("OK: no-heading issue page falls back to a single volume-titled bucket")


def test_end_to_end_extract_aaai_links_with_mocked_http():
    import json
    import os
    import tempfile
    from unittest.mock import patch
    import workflows.link_extractors.aaai_link_extractor as mod

    pages = {
        "https://aaai.org/proceeding/aaai-40-2026/": ROOT_HTML,
        "https://ojs.aaai.org/index.php/AAAI/issue/view/683": VOLUME_HTML,
        "https://ojs.aaai.org/index.php/AAAI/issue/view/684": MULTI_TRACK_VOLUME_HTML,
        "https://ojs.aaai.org/index.php/AAAI/issue/view/685": NO_HEADING_VOLUME_HTML,
    }

    class FakeResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(pages[url])

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            with patch.object(mod.requests, "get", side_effect=fake_get), \
                 patch.object(mod.time, "sleep", return_value=None):
                path = mod.extract_aaai_links(
                    "https://aaai.org/proceeding/aaai-40-2026/", "AAAI", "2026"
                )
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
        finally:
            os.chdir(original_cwd)

    total_links = sum(len(t["paper_links"]) for t in data)
    # 3 (issue 683) + 2 (issue 684, track I) + 1 (issue 684, track II) + 2 (issue 685, fallback) = 8
    assert total_links == 8, data
    assert len(data) == 4, f"expected 4 track buckets total, got {len(data)}"
    print("OK: end-to-end extract_aaai_links with mocked HTTP → "
          f"{total_links} paper links across {len(data)} tracks")


if __name__ == "__main__":
    test_find_volume_links_skips_subtrack_duplicate_and_matches_vol_pattern()
    test_extract_tracks_from_volume_gets_article_view_links_not_pdf()
    test_extract_tracks_from_volume_handles_multiple_tracks_on_one_issue()
    test_extract_tracks_from_volume_falls_back_when_no_heading_found()
    test_end_to_end_extract_aaai_links_with_mocked_http()
    print("\nAll offline AAAI extractor tests passed.")
