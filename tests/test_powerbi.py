"""
Tests for the Power BI layer.

Power BI Desktop is not available here, so nothing below proves the report opens
or that the DAX evaluates. What these tests do cover is everything that can be
checked without it:

* the exported tables exist, are small enough to commit, and reconcile with the
  figures the phase reports quote;
* the measure library parses and every measure has a home table;
* the scenario measure's arithmetic identity holds, which is the one piece of
  non-trivial DAX logic and the place Phase 6's double-counting bug would recur.

The Phase 8 report states the same limitation rather than implying the model is
verified.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from powerbi import parity, tmdl  # noqa: E402
from powerbi.export import output_dir  # noqa: E402

DATA = output_dir()
POWERBI = PROJECT_ROOT / "powerbi"

pytestmark = pytest.mark.skipif(
    not (DATA / "fact_daily_category.csv").exists(),
    reason="Power BI export not present; run src/powerbi/export.py first.",
)

EXPECTED_TABLES = [
    "dim_store", "dim_product", "dim_calendar", "dim_category",
    "fact_daily_category", "reorder_policy",
    "causal_estimates", "dose_response", "spillover", "service_levels",
    "promo_plan", "promo_plan_uncertainty", "promo_plan_draws", "promo_economics",
]


# ---------------------------------------------------------------------------
# The exported data layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_exists_and_is_non_empty(table):
    path = DATA / f"{table}.csv"
    assert path.exists(), f"{table}.csv missing"
    assert len(pd.read_csv(path, nrows=5)) > 0


def test_export_stays_committable():
    """
    The whole point of aggregating was to keep the repository clonable. If this
    starts failing, the fact grain has crept back towards SKU level.
    """
    total_mb = sum(p.stat().st_size for p in DATA.glob("*.csv")) / (1024 * 1024)
    assert total_mb < 30, f"Power BI export is {total_mb:.1f} MB"


def test_fact_grain_is_unique():
    fact = pd.read_csv(DATA / "fact_daily_category.csv")
    assert not fact.duplicated(["date", "store_id", "category"]).any()


def test_latent_columns_are_not_exported():
    """
    §7 bars simulation outputs from any model. A BI surface is exactly where a
    latent-demand column would get mistaken for something actionable.
    """
    fact = pd.read_csv(DATA / "fact_daily_category.csv", nrows=5)
    for banned in ("potential_demand", "lost_sales", "anomaly"):
        assert not any(banned in c for c in fact.columns)


def test_relationship_keys_resolve():
    """Every relationship the model declares must join on values that exist."""
    for from_table, from_column, to_table, to_column in tmdl.RELATIONSHIPS:
        source = pd.read_csv(DATA / f"{from_table}.csv")
        target = pd.read_csv(DATA / f"{to_table}.csv")
        orphans = set(source[from_column].astype(str)) - set(target[to_column].astype(str))
        assert not orphans, f"{from_table}.{from_column} has orphans: {sorted(orphans)[:5]}"


def test_dimension_keys_are_unique():
    """A many-to-one relationship needs a unique key on the one side."""
    for table, key in (
        ("dim_store", "store_id"), ("dim_product", "sku_id"),
        ("dim_calendar", "date"), ("dim_category", "category"),
    ):
        frame = pd.read_csv(DATA / f"{table}.csv")
        assert frame[key].is_unique, f"{table}.{key} is not unique"


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def test_parity_table_covers_every_page():
    expected = parity.compute_expected()
    pages = set(expected["page"])
    assert pages == {
        "Executive Summary", "Promotion ROI", "Elasticity Explorer",
        "Stockout Risk", "What-If Simulator",
    }


def test_parity_values_are_finite():
    expected = parity.compute_expected()
    assert np.isfinite(expected["expected_value"]).all()


def test_parity_reconciles_with_the_phase_reports():
    """
    Spot-check the figures the written reports quote. If the export drifts, the
    reports and the dashboard would silently disagree.
    """
    expected = parity.compute_expected().set_index("measure")["expected_value"]

    assert expected["Naive Promo Lift %"] == pytest.approx(126.7, abs=0.5)
    assert expected["True Promo Lift %"] == pytest.approx(81.0, abs=0.5)
    assert expected["Promotion Rate %"] == pytest.approx(8.49, abs=0.05)
    assert expected["Stockout Rate %"] == pytest.approx(0.282, abs=0.01)
    assert expected["Cannibalisation 1 Neighbour %"] == pytest.approx(-6.1, abs=0.3)


def test_scenario_measure_reproduces_the_plan_at_multiplier_one():
    """
    The identity the What-If page depends on. At an uplift multiplier of 1 the
    scenario expression must return the optimiser's own figure; if it does not,
    the DAX is deducting the promotional give-away twice, which is precisely the
    bug Phase 6 shipped and fixed.
    """
    plan = pd.read_csv(DATA / "promo_plan.csv")
    scenario = parity._scenario_profit(plan, 1.0)
    assert scenario == pytest.approx(plan["incremental_profit"].sum(), abs=0.01)


def test_scenario_profit_rises_with_the_multiplier():
    plan = pd.read_csv(DATA / "promo_plan.csv")
    values = [parity._scenario_profit(plan, m) for m in (0.6, 0.8, 1.0, 1.2)]
    assert values == sorted(values)


def test_plan_columns_support_the_scenario_measure():
    """The what-if recomputes profit, so it needs both margins, not just the result."""
    plan = pd.read_csv(DATA / "promo_plan.csv", nrows=1)
    for column in ("baseline_units", "incremental_units", "promo_margin",
                   "full_margin", "cannibalisation_loss"):
        assert column in plan.columns


# ---------------------------------------------------------------------------
# The measure library and generated model
# ---------------------------------------------------------------------------

def test_every_measure_has_a_home_table():
    measures = tmdl.parse_measures(POWERBI / "measures.dax")
    homed = {name for names in tmdl.MEASURE_HOME.values() for name in names}
    assert set(measures) == homed


def test_measures_parse_to_non_empty_expressions():
    measures = tmdl.parse_measures(POWERBI / "measures.dax")
    assert len(measures) > 30
    for name, expression in measures.items():
        assert expression.strip(), f"{name} parsed to an empty expression"


def test_measure_home_tables_exist():
    for table in tmdl.MEASURE_HOME:
        assert (DATA / f"{table}.csv").exists(), f"{table} has measures but no data"


def test_generated_model_files_exist():
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition"
    assert (definition / "model.tmdl").exists()
    assert (definition / "relationships.tmdl").exists()
    assert (definition / "expressions.tmdl").exists()
    assert (POWERBI / f"{tmdl.PROJECT_NAME}.pbip").exists()
    for table in EXPECTED_TABLES:
        assert (definition / "tables" / f"{table}.tmdl").exists()


def test_generated_tmdl_declares_every_column():
    """
    The model is generated from the data precisely so the schema cannot drift.
    This asserts that it did not.
    """
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition" / "tables"
    for table in EXPECTED_TABLES:
        columns = list(pd.read_csv(DATA / f"{table}.csv", nrows=1).columns)
        text = (definition / f"{table}.tmdl").read_text(encoding="utf-8")
        for column in columns:
            assert f"column {column}\n" in text, f"{table}.tmdl missing column {column}"


def test_what_if_table_is_generated_not_left_to_the_user():
    """
    `Scenario Incremental Profit` reads SELECTEDVALUE('Uplift Scenario'[...], 1).
    If the table is absent that measure silently returns its default of 1 and the
    What-If page does nothing — a failure with no error message.
    """
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition"
    table = definition / "tables" / f"{tmdl.WHAT_IF_TABLE}.tmdl"
    assert table.exists(), "the what-if table was not generated"

    text = table.read_text(encoding="utf-8")
    assert "= calculated" in text
    assert "GENERATESERIES" in text
    # A ref in model.tmdl is what actually loads it.
    model = (definition / "model.tmdl").read_text(encoding="utf-8")
    assert f"ref table '{tmdl.WHAT_IF_TABLE}'" in model

    measures = tmdl.parse_measures(POWERBI / "measures.dax")
    assert f"'{tmdl.WHAT_IF_TABLE}'" in measures["Scenario Incremental Profit"]


@pytest.mark.parametrize(
    "given, expected",
    [
        (r"C:\Users\me\powerbi_data", "C:/Users/me/powerbi_data/"),
        (r"C:\Users\me\powerbi_data" + "\\", "C:/Users/me/powerbi_data/"),
        ("/Users/me/powerbi_data", "/Users/me/powerbi_data/"),
        ("/Users/me/powerbi_data/", "/Users/me/powerbi_data/"),
    ],
)
def test_data_folder_literal_normalises_separators(given, expected):
    """
    Backslash is not an escape character in Power Query text, so the path is
    emitted verbatim — doubling the separators would produce a path resolving to
    nothing. Forward slashes work on Windows too, so one form serves both and
    the literal never ends in a backslash against the closing quote.
    """
    assert tmdl.data_folder_literal(given) == expected


def test_data_folder_override_does_not_change_where_csvs_are_read_from(tmp_path):
    """
    The override targets another machine. If it leaked into the read path, type
    inference would be reading CSVs from a directory that does not exist here.

    Note `output_root` — no test may generate into `powerbi/`. That tree holds 66
    hand-authored visuals and a model Desktop has enriched, and an earlier
    version of this test called `build()` bare, which would have overwritten
    both. The visuals were untracked at the time.
    """
    result = tmdl.build(data_folder=r"C:\elsewhere\powerbi_data", output_root=tmp_path)
    assert result["data_folder"] == "C:/elsewhere/powerbi_data/"

    definition = tmp_path / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition"
    assert '"C:/elsewhere/powerbi_data/"' in (definition / "expressions.tmdl").read_text("utf-8")
    # Types still came from the real local CSVs, not the overridden path.
    generated = (definition / "tables" / "dim_store.tmdl").read_text("utf-8")
    for column in pd.read_csv(DATA / "dim_store.csv", nrows=1).columns:
        assert f"column {column}\n" in generated


def test_generating_never_touches_the_committed_report(tmp_path):
    """
    66 hand-authored visuals are not regenerable. write_report scaffolds only
    what is missing, so a routine `build()` cannot delete a day of authoring.
    """
    report = tmp_path / f"{tmdl.PROJECT_NAME}.Report"
    tmdl.build(output_root=tmp_path)

    page = report / "definition" / "pages" / tmdl.PAGES[0][0] / "page.json"
    page.write_text('{"sentinel": true}', encoding="utf-8")
    visual = page.parent / "visuals" / "v1" / "visual.json"
    visual.parent.mkdir(parents=True)
    visual.write_text('{"sentinel": true}', encoding="utf-8")

    tmdl.build(output_root=tmp_path)
    assert json.loads(page.read_text("utf-8")) == {"sentinel": True}, "page.json was overwritten"
    assert visual.exists(), "an authored visual was deleted"

    tmdl.build(output_root=tmp_path, force_report=True)
    assert "sentinel" not in page.read_text("utf-8"), "force_report should re-scaffold"


def test_boolean_columns_are_typed_as_logical_not_integer():
    """
    The defect that emptied three tables. pandas writes booleans as the text
    True/False; casting them to Int64.Type makes Power Query error on *every*
    row, so the table loads empty behind a mild "incomplete data" banner.
    dim_calendar dying took the date relationship, Revenue LY, Revenue YoY % and
    every date slicer with it, and made Service Level Insight a DAX type error.

    Nothing surfaced until Power Query ran, which is why it is pinned here.
    """
    frame = pd.DataFrame({"is_perishable": [True, False], "units": [1, 2]})
    types = tmdl.infer_types(frame)
    assert types["is_perishable"] == ("boolean", "type logical")
    assert types["units"] == ("int64", "Int64.Type")
    # A boolean renders True/False; a numeric format string on it is wrong.
    assert tmdl.format_string("is_perishable", "boolean") == ""


def test_every_boolean_column_in_the_shipped_model_is_logical():
    """The same check against the committed TMDL, not just the type mapper."""
    for path in (DEFINITION / "tables").glob("*.tmdl"):
        text = path.read_text(encoding="utf-8")
        for column in re.findall(r"^\tcolumn (is_\w+)$", text, re.M):
            block = text.split(f"\tcolumn {column}\n", 1)[1].split("\n\n", 1)[0]
            assert "dataType: boolean" in block, f"{path.name}:{column} is not boolean"
            assert f'{{"{column}", type logical}}' in text, (
                f"{path.name}:{column} is boolean in TMDL but not cast as logical in M"
            )


def test_uplift_scenario_contains_the_multiplier_parity_expects():
    """
    PAGE_SPEC once specified a 0.05 step while dax_parity.csv expects a 0.88
    multiplier — which is not a member of that series. A double-typed series
    also drifted, ending at 1.25 instead of 1.30.
    """
    expected = parity.compute_expected().set_index("measure")["expected_value"]
    assert "Scenario Profit at Multiplier 0.88" in expected.index

    text = (DEFINITION / "tables" / f"{tmdl.WHAT_IF_TABLE}.tmdl").read_text("utf-8")
    assert "dataType: decimal" in text, "double drifts through binary floating point"
    assert "ParameterMetadata" in text, "without it the slicer is a checkbox list"

    steps = round((tmdl.WHAT_IF_MAX - tmdl.WHAT_IF_MIN) / tmdl.WHAT_IF_STEP)
    members = {round(tmdl.WHAT_IF_MIN + i * tmdl.WHAT_IF_STEP, 2) for i in range(steps + 1)}
    assert 0.88 in members, "the parity multiplier is not reachable on the slider"


def test_emitted_data_folder_is_normalised():
    text = (DEFINITION / "expressions.tmdl").read_text(encoding="utf-8")
    literal = re.search(r'expression DataFolder = "([^"]*)"', text).group(1)
    assert "\\" not in literal, "backslashes would not resolve in Power Query"
    assert literal.endswith("/") and not literal.endswith("//")


def test_data_folder_is_an_editable_parameter():
    """
    Without the `meta [IsParameterQuery=...]` record this is a plain expression:
    invisible in Edit parameters, so the only way to repoint the data folder on
    another machine is to hand-edit TMDL.
    """
    text = (
        POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition" / "expressions.tmdl"
    ).read_text(encoding="utf-8")
    assert "IsParameterQuery=true" in text
    assert 'Type="Text"' in text


def test_measure_format_string_is_inside_the_measure_block():
    """
    A blank line between a measure's expression and its formatString closes the
    block, and TMDL then reads the property as the table's. This caught a real
    defect in the generator.
    """
    definition = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition" / "tables"
    text = (definition / "causal_estimates.tmdl").read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("formatString:"):
            assert lines[index - 1].strip(), "blank line precedes a formatString"


# ---------------------------------------------------------------------------
# The PBIP project files
#
# Desktop validates these against published JSON schemas and refuses to open the
# project if one does not match. Phase 8 shipped a `.pbip` carrying the report
# *item* schema where the pbip schema belongs, pointing at a report folder that
# was never generated — two blocking errors that every test above sailed past,
# because none of them opened the project files at all.
# ---------------------------------------------------------------------------

PBIP = POWERBI / f"{tmdl.PROJECT_NAME}.pbip"
REPORT = POWERBI / f"{tmdl.PROJECT_NAME}.Report"

# The pattern Desktop reported in its own error message.
PBIP_SCHEMA_PATTERN = re.compile(
    r"^https://developer\.microsoft\.com/json-schemas/fabric/pbip/"
    r"pbipProperties/1\.[0-9]+\.[0-9]+/schema\.json$"
)


def _generated_json_files():
    """
    Every JSON document in the shipped project, whatever its extension.

    `.pbi/` is excluded: it is per-user editor state, gitignored per Microsoft's
    guidance, and carries schemas this repo has no reason to mirror.
    """
    paths = (
        [PBIP, REPORT / "definition.pbir"]
        + list(POWERBI.rglob(".platform"))
        + list(REPORT.rglob("*.json"))
        + [POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition.pbism"]
    )
    return sorted(p for p in paths if ".pbi" not in p.parts)


def test_every_generated_json_document_parses():
    for path in _generated_json_files():
        assert path.exists(), f"{path.name} missing"
        json.loads(path.read_text(encoding="utf-8"))


def test_pbip_schema_is_absent_or_correct():
    """
    Desktop writes the `.pbip` with no `$schema` at all, and likewise
    `definition.pbir`. A *wrong* one is fatal — that was the first blocking
    error — so the rule is: omit it, or get it exactly right.
    """
    document = json.loads(PBIP.read_text(encoding="utf-8"))
    schema = document.get("$schema")
    if schema is not None:
        assert PBIP_SCHEMA_PATTERN.match(schema), f"Desktop will reject this $schema: {schema}"


def test_report_definition_carries_no_schema():
    """
    `definition.pbir` must not declare one. This was part of why the 1.x
    scaffold opened to a blank page.
    """
    assert "$schema" not in json.loads((REPORT / "definition.pbir").read_text("utf-8"))


def test_every_schema_url_is_a_microsoft_schema():
    """A typo'd or invented schema host fails at open time, not here, so check it here."""
    for path in _generated_json_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        if "$schema" in document:
            assert document["$schema"].startswith(
                "https://developer.microsoft.com/json-schemas/fabric/"
            ), f"{path.name} has a non-Fabric $schema"


def test_pbip_points_at_a_report_that_exists():
    """The defect that made the project unopenable: a shortcut with no target."""
    document = json.loads(PBIP.read_text(encoding="utf-8"))
    path = document["artifacts"][0]["report"]["path"]
    assert "\\" not in path, "the artifact path must use forward slashes"
    assert (POWERBI / path).is_dir(), f"{path} is referenced but does not exist"


def test_report_references_the_semantic_model_by_relative_path():
    document = json.loads((REPORT / "definition.pbir").read_text(encoding="utf-8"))
    path = document["datasetReference"]["byPath"]["path"]
    assert not Path(path).is_absolute(), "absolute dataset paths are not supported"
    assert (REPORT / path).resolve().is_dir()
    assert (REPORT / path).resolve().name == f"{tmdl.PROJECT_NAME}.SemanticModel"


def test_pbir_required_files_exist():
    definition = REPORT / "definition"
    for required in ("version.json", "report.json", "pages/pages.json"):
        assert (definition / required).exists(), f"PBIR requires {required}"


def test_every_spec_page_is_scaffolded():
    pages_dir = REPORT / "definition" / "pages"
    order = json.loads((pages_dir / "pages.json").read_text(encoding="utf-8"))["pageOrder"]
    assert order == [name for name, _ in tmdl.PAGES]

    for name, display_name in tmdl.PAGES:
        page = json.loads((pages_dir / name / "page.json").read_text(encoding="utf-8"))
        assert page["name"] == name
        assert page["displayName"] == display_name
        # width/height are required for every displayOption but DeprecatedDynamic.
        assert page["width"] > 0 and page["height"] > 0


def test_page_folder_names_follow_the_pbir_convention():
    """
    PBIR ignores a page folder whose name is not word characters or hyphens —
    silently, so the page just never appears. `&` in a title is the easy way in.
    """
    for name, _ in tmdl.PAGES:
        assert re.fullmatch(r"[\w-]+", name), f"{name} would be silently ignored"
        assert (REPORT / "definition" / "pages" / name).is_dir()


# ---------------------------------------------------------------------------
# TMDL reference resolution
#
# Desktop resolves every named reference when it deserializes the model, and
# refuses the whole database if one dangles — "refers to an object which cannot
# be found", naming the property. A `queryGroup: Parameters` that no model ever
# declared shipped in Phase 8 and cost a round trip to find, so the whole class
# is checked here instead.
# ---------------------------------------------------------------------------

DEFINITION = POWERBI / f"{tmdl.PROJECT_NAME}.SemanticModel" / "definition"


def _declared_members():
    """{table name: {columns and measures it declares}} straight from the TMDL."""
    members = {}
    for path in (DEFINITION / "tables").glob("*.tmdl"):
        text = path.read_text(encoding="utf-8")
        table = re.match(r"table '?([^'\n]+?)'?\n", text).group(1)
        members[table] = (
            set(re.findall(r"^\tcolumn '?([^'\n]+?)'?$", text, re.M))
            | set(re.findall(r"^\tmeasure '([^']+)'", text, re.M))
        )
    return members


def _tmdl_files():
    return sorted(DEFINITION.rglob("*.tmdl"))


def test_every_query_group_reference_is_declared():
    """The defect that made Desktop refuse the model on the second attempt."""
    declared = set()
    for path in _tmdl_files():
        declared |= set(
            re.findall(r"^queryGroup '?([^'\n]+?)'?$", path.read_text(encoding="utf-8"), re.M)
        )
    for path in _tmdl_files():
        used = re.findall(r"^\s*queryGroup: '?([^'\n]+?)'?$", path.read_text(encoding="utf-8"), re.M)
        for group in used:
            assert group in declared, (
                f"{path.name} references query group {group!r}, which no TMDL declares. "
                "Desktop rejects the entire database for this."
            )


def test_model_refs_and_table_files_agree():
    model = (DEFINITION / "model.tmdl").read_text(encoding="utf-8")
    refs = {name for name in re.findall(r"^ref table '?([^'\n]+?)'?$", model, re.M)}
    assert refs == set(_declared_members()), (
        "model.tmdl's ref table lines and the tables/ folder have diverged"
    )


def test_relationship_endpoints_exist_in_the_model():
    """
    test_relationship_keys_resolve checks the *data* joins. This checks that the
    model declaring those joins names columns that actually exist in the TMDL.
    """
    members = _declared_members()
    text = (DEFINITION / "relationships.tmdl").read_text(encoding="utf-8")
    endpoints = re.findall(r"(?:fromColumn|toColumn): (\S+)", text)
    assert endpoints, "no relationship endpoints found — did the format change?"
    for endpoint in endpoints:
        table, column = endpoint.rsplit(".", 1)
        assert table in members, f"relationship names unknown table {table!r}"
        assert column in members[table], f"relationship names unknown column {endpoint}"


def test_generated_dax_only_references_columns_that_exist():
    """A measure referencing a dropped column fails at load, not here — so check here."""
    members = _declared_members()
    unresolved = []
    for path in (DEFINITION / "tables").glob("*.tmdl"):
        text = path.read_text(encoding="utf-8")
        for table, column in re.findall(r"'?([A-Za-z_][\w ]*?)'?\[([^\]]+)\]", text):
            if table not in members:
                unresolved.append(f"{path.name}: unknown table {table!r}")
            # GENERATESERIES names its output column Value; it is not declared.
            elif column not in members[table] and column != "Value":
                unresolved.append(f"{path.name}: {table}[{column}]")
    assert not unresolved, f"unresolved DAX references: {sorted(set(unresolved))}"


# ---------------------------------------------------------------------------
# Schema validation
#
# The report layer is the half Desktop had not reached while the model was still
# failing. Validating it against Microsoft's published schemas checks it here
# rather than on the next launch.
# ---------------------------------------------------------------------------

SCHEMA_DIR = PROJECT_ROOT / "tests" / "fixtures" / "fabric_schemas"


def _schema_registry():
    """
    Every vendored schema, keyed on the canonical URL it declares as its `$id`.

    The schemas are draft-07 and cross-reference each other with relative paths
    (`../../semanticQuery/1.0.0/schema.json#/definitions/...`). Keying on `$id`
    is what lets those resolve against the mirror instead of the network.
    """
    from referencing import Registry, Resource

    resources = []
    for path in SCHEMA_DIR.rglob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(document)
        # Register under the URL its own path implies *and* its declared $id.
        # These disagree for the embedded schemas: the file and every $ref say
        # `schema-embedded.json`, while $id says `schema.embedded.json`. Keying
        # on $id alone leaves those refs unresolvable.
        canonical = (
            "https://developer.microsoft.com/json-schemas/"
            + str(path.relative_to(SCHEMA_DIR)).replace("\\", "/")
        )
        resources.append((canonical, resource))
        declared = document.get("$id")
        if declared and declared != canonical:
            resources.append((declared, resource))
    return Registry().with_resources(resources)


def _validate(document, registry):
    from jsonschema import Draft7Validator

    schema = json.loads(
        _vendored_path(document["$schema"]).read_text(encoding="utf-8")
    )
    Draft7Validator(schema, registry=registry).validate(document)


def _vendored_path(schema_url: str) -> Path:
    """Map a `$schema` URL onto its mirrored file."""
    marker = "/json-schemas/"
    assert marker in schema_url, f"not a Fabric schema URL: {schema_url}"
    return SCHEMA_DIR / schema_url.split(marker, 1)[1]


def test_every_generated_schema_url_is_vendored():
    """
    Bumping a schema constant in tmdl.py without mirroring the new version would
    otherwise make the validation below silently skip the document.
    """
    for name in dir(tmdl):
        if name.endswith("_SCHEMA"):
            url = getattr(tmdl, name)
            assert _vendored_path(url).exists(), (
                f"{name} points at {url}, which is not in tests/fixtures/fabric_schemas/"
            )


def test_generated_documents_validate_against_the_published_schemas():
    """
    The check that would have caught the original malformed `$schema`, and would
    catch a bad displayOption enum or a missing required property today.
    """
    registry = _schema_registry()
    checked = 0
    for path in _generated_json_files():
        document = json.loads(path.read_text(encoding="utf-8"))
        # definition.pbism and definition.pbir declare none — Desktop writes
        # definition.pbir without a $schema, and adding one gets it rejected.
        if "$schema" not in document:
            continue
        _validate(document, registry)
        checked += 1
    # 66 visuals plus the project files. A collapse in this count means the file
    # walk stopped finding the report, not that validation got easier.
    assert checked >= 70, f"only {checked} documents were schema-validated"


@pytest.mark.parametrize(
    "label, mutation",
    [
        ("bad displayOption enum", lambda d: d.update(displayOption="FitToPageish")),
        ("missing required property", lambda d: d.pop("displayName")),
        ("wrong type for width", lambda d: d.update(width="1280")),
    ],
)
def test_schema_validation_can_actually_fail(label, mutation):
    """
    A validator that cannot fail is decoration — the same rule the Phase 1 QA
    checks are mutation-tested under. Corrupt a document, require a complaint.
    """
    from jsonschema import ValidationError

    registry = _schema_registry()
    page = json.loads(
        (REPORT / "definition" / "pages" / tmdl.PAGES[0][0] / "page.json").read_text("utf-8")
    )
    _validate(page, registry)  # the unmutated document must pass

    mutation(page)
    with pytest.raises(ValidationError):
        _validate(page, registry)
