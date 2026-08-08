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

from powerbi.export import output_dir  # noqa: E402

DATA = output_dir()
README = PROJECT_ROOT / "README.md"
CASE_STUDY = PROJECT_ROOT / "reports" / "case_study.md"

pytestmark = pytest.mark.skipif(
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


def test_service_levels_quoted_match_the_export():
    levels = pd.read_csv(DATA / "service_levels.csv").set_index("category")
    fresh = levels.loc["Fresh Produce", "median_service_level"] * 100
    assert fresh == pytest.approx(52.31, abs=0.01)
    for document in (README, CASE_STUDY):
        assert "52%" in _text(document), "the Fresh Produce finding is the memorable one"


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


def test_readme_stays_a_five_minute_read():
    """§8: 'tells the full business story in under a 5-minute read'."""
    words = len(re.findall(r"\S+", _text(README)))
    assert words < 1600, f"README is {words} words — past a five-minute read"
