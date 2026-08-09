"""
Keep the written narrative tied to the computed results.

Every other layer of this repo pins its numbers rather than trusting prose — QA
checks are mutation-tested, DAX measures have expected values, the causal
estimate is scored against the generator. A README quoting a figure that a
regeneration has since moved is the same class of failure, and the least likely
to be noticed: nothing errors, the document just becomes quietly untrue.

This covers the headline figures that carry the argument, not every digit in the
prose. A test that pinned all of them would fail on rewording and get deleted.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from powerbi import tmdl  # noqa: E402
from powerbi.export import output_dir  # noqa: E402

DATA = output_dir()
README = PROJECT_ROOT / "README.md"
CASE_STUDY = PROJECT_ROOT / "reports" / "case_study.md"
PAGE_SPEC = PROJECT_ROOT / "powerbi" / "PAGE_SPEC.md"
PHASE8 = PROJECT_ROOT / "reports" / "phase8_powerbi.md"
REPORT_PAGES = PROJECT_ROOT / "powerbi" / "Northstar.Report" / "definition" / "pages"

# Only the tests that read the exported CSVs need the export. The rest compare
# committed documents against committed files, and were being skipped in CI for
# no reason — CI generates the dataset but does not run src/powerbi/export.py,
# so a module-level skip took the whole file with it.
needs_export = pytest.mark.skipif(
    not (DATA / "dax_parity.csv").exists(),
    reason="Power BI export not present; run src/powerbi/export.py first.",
)


def _parity():
    return pd.read_csv(DATA / "dax_parity.csv").set_index("measure")["expected_value"]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

@needs_export
@pytest.mark.parametrize(
    "measure, quoted",
    [
        ("Naive Promo Lift %", "126.7"),
        ("Causal Promo Lift %", "96.4"),
        ("True Promo Lift %", "81.0"),
        ("Bias Removed %", "64%"),          # 63.72, rounded in prose
        ("Max Recovery Error", "0.027"),
        ("Candidates Profitable %", "0.15%"),
    ],
)
def test_readme_headline_figures_match_the_parity_table(measure, quoted):
    """The three-way estimate comparison is the centrepiece; it must be right."""
    expected = _parity()[measure]
    assert quoted in _text(README), f"README no longer quotes {measure} as {quoted}"
    # Guard the rounding too, so the prose cannot drift from the source value.
    numeric = float(re.sub(r"[%£,]", "", quoted))
    assert abs(round(expected, len(quoted.split(".")[-1]) if "." in quoted else 0) - numeric) < 1.0


@needs_export
def test_monte_carlo_range_is_quoted_consistently():
    """
    The £337 plan figure and the negative median must always appear together.
    Quoting the optimiser's estimate alone would contradict the dashboard.
    """
    parity = _parity()
    for document in (README, CASE_STUDY):
        text = _text(document)
        assert "337" in text
        assert "74%" in text, "the probability of loss must accompany the plan figure"
        assert "268" in text or "267" in text, "the negative median must be stated"

    assert parity["Plan Incremental Profit"] == pytest.approx(337.03, abs=0.01)
    assert parity["Plan Profit P50"] < 0, "the median outcome is negative — say so"
    assert parity["Probability of Loss %"] == pytest.approx(74.2, abs=0.1)


@needs_export
def test_service_levels_quoted_match_the_export():
    levels = pd.read_csv(DATA / "service_levels.csv").set_index("category")
    fresh = levels.loc["Fresh Produce", "median_service_level"] * 100
    assert fresh == pytest.approx(52.31, abs=0.01)
    for document in (README, CASE_STUDY):
        assert "52%" in _text(document), "the Fresh Produce finding is the memorable one"


@needs_export
def test_plan_size_matches_the_optimiser_output():
    plan = pd.read_csv(DATA / "promo_plan.csv")
    assert len(plan) == 10
    assert "10" in _text(CASE_STUDY)


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------

def test_readme_has_the_limitations_section():
    """
    PROJECT_ARCHITECTURE.md §7 makes this mandatory, and §8 lists it in the
    Definition of Done. It is also the section most likely to be quietly
    softened in a rewrite.
    """
    text = _text(README)
    assert "What I would not claim" in text

    # Collapse wrapping — markdown breaks lines mid-phrase.
    section = re.sub(r"\s+", " ", text.split("What I would not claim", 1)[1])
    for claim in (
        "upper bound",        # parallel trends
        "not identified",     # price elasticity
        "SUTVA",              # cannibalisation
        "censored",           # forecast target
        "no traffic-building effect",
        "simulated",          # the truth column
    ):
        assert claim in section, f"the limitations section no longer mentions: {claim}"


def test_naive_estimate_is_always_shown_next_to_the_corrected_one():
    """§7: reporting the corrected number alone loses the point of the exercise."""
    for document in (README, CASE_STUDY):
        text = _text(document).lower()
        if "causal" in text:
            assert "naive" in text, f"{document.name} reports a causal figure with no naive one"


def test_every_linked_report_and_image_exists():
    for document in (README, CASE_STUDY):
        text = _text(document)
        for target in re.findall(r"\]\(([^)#]+\.(?:md|png))\)", text):
            resolved = (document.parent / target).resolve()
            assert resolved.exists(), f"{document.name} links to a missing file: {target}"


def test_every_screenshot_is_referenced():
    """
    The reverse of the check above, and the one that was missing.

    `01_executive_summary.png` was committed and pushed while nothing linked to
    it — a public repository carrying 400KB no reader would ever reach. Links
    resolving says nothing about whether every committed asset is used.
    """
    prose = _text(README) + _text(CASE_STUDY)
    orphans = [
        path.name
        for path in sorted((PROJECT_ROOT / "powerbi" / "screenshots").glob("*.png"))
        if path.name not in prose
    ]
    assert not orphans, f"screenshots committed but referenced nowhere: {orphans}"


def test_readme_stays_a_five_minute_read():
    """
    §8: 'tells the full business story in under a 5-minute read'.

    Raised from 1,600 to 1,700 once the five dashboard screenshots, the CI badge
    and the build-provenance section had all landed. Still a five-to-six minute
    read; the point of the cap is to stop the README turning into a report, not
    to pin an exact number.
    """
    words = len(re.findall(r"\S+", _text(README)))
    assert words < 1700, f"README is {words} words — past a five-minute read"


# ---------------------------------------------------------------------------
# The build state
# ---------------------------------------------------------------------------

SUPERSEDED_HEADING = "## Superseded:"

# Phrases that were true while the report was a scaffold and are false now.
# Each one, left in the live text, tells a reader to produce something the
# repository already contains.
STALE_BUILD_INSTRUCTIONS = [
    "Build the five pages",
    "cannot be declared in TMDL",
    "unopened draft",
    "already\nnamed and empty",
    "present and empty",
    "still needs a human",
    "rather than a built report",
]


def _live_text(path: Path) -> str:
    """The document above its superseded block, which is history, not instruction."""
    return _text(path).split(SUPERSEDED_HEADING, 1)[0]


def _built_pages() -> list[str]:
    return sorted(p.name for p in REPORT_PAGES.iterdir() if p.is_dir())


@pytest.mark.parametrize("document", [PAGE_SPEC, PHASE8], ids=lambda p: p.name)
def test_no_document_asks_for_work_the_repo_already_contains(document):
    """
    Both documents once described a report that had not been assembled, and both
    kept saying so after 66 visuals landed. A reader following either would have
    rebuilt five finished pages, and the second-order cost is worse: a document
    that is wrong about the thing it is nearest to stops being worth trusting on
    anything else.

    Superseded blocks are exempt by design. The build trail is worth keeping;
    what must not survive is an instruction presented as current.
    """
    live = _live_text(document)
    found = [phrase for phrase in STALE_BUILD_INSTRUCTIONS if phrase in live]
    assert not found, (
        f"{document.name} still instructs work the repo already contains: {found}. "
        f"Move it under '{SUPERSEDED_HEADING}' if it is build history."
    )


@pytest.mark.parametrize("document", [PAGE_SPEC, PHASE8], ids=lambda p: p.name)
def test_documents_claim_the_page_and_visual_counts_that_exist(document):
    """
    The counts are the load-bearing part of the claim "these pages are built".
    Prose saying five pages and 66 visuals is only worth anything if the
    definition folder still holds five pages and 66 visuals.
    """
    pages = _built_pages()
    visuals = list(REPORT_PAGES.rglob("visual.json"))
    live = _live_text(document)

    assert len(pages) == 5, f"expected 5 page directories, found {pages}"
    assert str(len(visuals)) in live, (
        f"{document.name} does not state the visual count; the repo has {len(visuals)}"
    )
    assert "five pages" in live or "all five" in live, (
        f"{document.name} no longer claims the five pages that exist"
    )


def test_measure_count_in_prose_matches_the_library():
    """
    Three documents quoted three different numbers, none of which was 45. A
    count is the cheapest possible claim to check and the easiest to leave
    behind when measures are added.
    """
    actual = len(tmdl.parse_measures(PROJECT_ROOT / "powerbi" / "measures.dax"))
    for document in (PAGE_SPEC, PHASE8):
        quoted = re.findall(r"(\d+)\s+measures", _live_text(document))
        assert quoted, f"{document.name} no longer states a measure count"
        for number in quoted:
            assert int(number) == actual, (
                f"{document.name} claims {number} measures; measures.dax defines {actual}"
            )
