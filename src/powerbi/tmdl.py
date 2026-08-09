"""
Generate the PBIP project and its TMDL semantic model.

Why generate rather than hand-write
-----------------------------------
Every column name, data type and M transform here is derived from the exported
CSVs at build time. Hand-authoring twenty TMDL files means twenty chances to
mistype a column or guess a type wrong, and none of those mistakes are visible
until Power BI Desktop refuses to load the model. Generating removes that entire
class of error: if the export changes, the model changes with it.

Honest scope
------------
This writes the **semantic model** - tables, columns, types, relationships,
measures, formatting - the `.pbip`, and a PBIR **report shell**: five correctly
named but empty pages. It does not author the visual layer. A visual carries
position and query bindings, and authoring thirty of them blind is where this
would most likely produce something that opens to broken visuals.
`powerbi/PAGE_SPEC.md` specifies the five pages precisely enough to assemble in
Desktop instead.

The report shell is not optional decoration: a `.pbip` is only a shortcut to a
report artifact, so without it Desktop has nothing to open. Phase 8 originally
shipped the shortcut without the target, alongside a `$schema` that failed
Desktop's validation pattern - neither of which is detectable without opening
Desktop, which is why `tests/test_powerbi.py` now parses and checks every
generated JSON document offline.

None of this has been opened in Power BI Desktop, because that is not available
here. The Phase 8 report says so rather than implying the model is verified.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerbi.export import output_dir  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("northstar.powerbi.tmdl")

PROJECT_NAME = "Northstar"

# Fabric JSON schema URLs. Power BI Desktop regex-validates every one of these on
# open and refuses to load the project if one does not match, so they are exact
# strings rather than approximations. The original `.pbip` shipped with the
# *report item* schema path where the pbip path belongs, which is precisely the
# error this class of constant exists to stop recurring.
#
# PBIR has *generations*. The first scaffold emitted the 1.x set; Desktop 2.155
# reads 2.x, and on seeing definition version 1.0.0 it silently treated the
# report as PBIR-Legacy, found no legacy layout, and opened one blank page —
# then refused to save with "the report has no pages". No error named the cause.
# These values are read off a report Desktop itself wrote, not inferred: note
# that version.json keeps the 1.0.0 *schema* while its version *value* is 2.0.0,
# and that definition.pbir carries no $schema at all.
_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
PBIP_SCHEMA = f"{_SCHEMA}/pbip/pbipProperties/1.0.0/schema.json"
PLATFORM_SCHEMA = f"{_SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json"
PBIR_VERSION_SCHEMA = f"{_SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json"
PBIR_DEFINITION_VERSION = "2.0.0"
PBIR_REPORT_SCHEMA = f"{_SCHEMA}/item/report/definition/report/2.1.0/schema.json"
PBIR_PAGES_SCHEMA = f"{_SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json"
PBIR_PAGE_SCHEMA = f"{_SCHEMA}/item/report/definition/page/2.0.0/schema.json"
PBIR_VISUAL_SCHEMA = f"{_SCHEMA}/item/report/definition/visualContainer/2.1.0/schema.json"

# The five pages of powerbi/PAGE_SPEC.md, scaffolded empty.
#
# Folder and `name` must match ^[\w-]+$ — PBIR silently ignores a folder whose
# name does not, and the page simply disappears rather than erroring. The
# display name carries the punctuation instead.
PAGES: List[Tuple[str, str]] = [
    ("ExecutiveSummary", "Executive Summary"),
    ("PromotionROI", "Promotion ROI"),
    ("ElasticityExplorer", "Elasticity Explorer"),
    ("StockoutRisk", "Stockout Risk & Replenishment"),
    ("WhatIfSimulator", "What-If Promotion Simulator"),
]

# The what-if parameter behind `Scenario Incremental Profit`. Desktop's
# Modeling > New parameter builds nothing more than a calculated table, which
# TMDL expresses perfectly well, so there is no reason to leave it manual.
# The range is measures.dax's; 0.88 is the documented default because Phase 4
# found the DiD estimate overshot the simulated truth.
WHAT_IF_TABLE = "Uplift Scenario"
WHAT_IF_MIN, WHAT_IF_MAX, WHAT_IF_STEP = 0.5, 1.3, 0.01

# Tables the model loads, and the column each relates on.
DIMENSIONS = ["dim_store", "dim_product", "dim_calendar", "dim_category"]
FACTS = ["fact_daily_category", "reorder_policy"]
RESULTS = [
    "causal_estimates", "dose_response", "spillover", "service_levels",
    "promo_plan", "promo_plan_uncertainty", "promo_plan_draws", "promo_economics",
    "dax_parity",
]

RELATIONSHIPS: List[Tuple[str, str, str, str]] = [
    # (from table, from column, to table, to column)
    ("fact_daily_category", "date", "dim_calendar", "date"),
    ("fact_daily_category", "store_id", "dim_store", "store_id"),
    ("fact_daily_category", "category", "dim_category", "category"),
    ("reorder_policy", "store_id", "dim_store", "store_id"),
    ("reorder_policy", "sku_id", "dim_product", "sku_id"),
]

# Which table each measure block attaches to.
MEASURE_HOME = {
    "fact_daily_category": [
        "Total Revenue", "Total Gross Profit", "Gross Margin %", "Units Sold",
        "Promotion Rate %", "Stockout Rate %", "Promoted Revenue Share %",
        "Revenue LY", "Revenue YoY %",
    ],
    "causal_estimates": [
        "Naive Promo Lift %", "Causal Promo Lift %", "True Promo Lift %",
        "Naive Bias pp", "Causal Bias pp", "Bias Removed %", "Estimate Health",
        "Effect %",
    ],
    "dose_response": [
        "Dose Response Points", "CI Coverage %", "Max Recovery Error",
        "Mean Lift at 20% Discount", "Estimated Lift %", "True Lift %",
        "Lift Recovery Gap pp",
    ],
    "spillover": [
        "Cannibalisation 1 Neighbour %", "Cannibalisation 4+ Neighbours %",
        "Spillover Effect %",
    ],
    "service_levels": [
        "Median Service Level %", "Lowest Service Level %", "Service Level",
    ],
    "reorder_policy": [
        "Mean Safety Stock", "Mean Reorder Point", "Service Level Insight",
    ],
    "promo_plan": [
        "Promotions Selected", "Plan Spend", "Plan Incremental Profit",
        "Scenario Incremental Profit", "Scenario vs Plan", "Scenario Verdict",
    ],
    "promo_plan_uncertainty": [
        "Plan Profit P10", "Plan Profit P50", "Plan Profit P90",
        "Probability of Loss %",
    ],
    "promo_economics": ["Candidates Profitable %"],
    "promo_plan_draws": ["Draw Count"],
}

# Calculated columns the report's visuals bind to. Like the chart-support
# measures, these exist because a visual needed an axis the exported CSVs do not
# carry — not because the analysis needs them.
CALCULATED_COLUMNS: Dict[str, List[Tuple[str, str, str, str]]] = {
    # table: [(column, DAX, dataType, formatString)]
    "promo_plan_draws": [(
        "profit_bucket",
        "ROUNDDOWN ( promo_plan_draws[profit_gbp] / 100, 0 ) * 100",
        "double",
        "\\£#,0;(\\£#,0);\\£#,0",
    )],
    "dim_calendar": [(
        "month_start",
        "DATE ( YEAR ( dim_calendar[date] ), MONTH ( dim_calendar[date] ), 1 )",
        "dateTime",
        "mmm yyyy",
    )],
}


def powerbi_dir() -> Path:
    return Path(config.PROJECT_ROOT) / "powerbi"


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def infer_types(frame: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Map each column to (TMDL dataType, Power Query type).

    Deliberately conservative: anything not clearly numeric or a date becomes
    text, because a wrong numeric guess makes the model fail to refresh while a
    text column merely looks wrong.
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            mapping[column] = ("dateTime", "type date")
        elif column == "date":
            mapping[column] = ("dateTime", "type date")
        elif pd.api.types.is_bool_dtype(series):
            # NOT Int64.Type. pandas writes booleans as the text True/False, and
            # Power Query cannot convert "True" to a whole number — so every row
            # errors and the table loads *empty*, with only a mild "some tables
            # have incomplete or no data" banner to show for it. This emptied
            # dim_calendar, taking the date relationship, Revenue LY, Revenue
            # YoY % and every date slicer with it, and turned the
            # is_perishable comparison in Service Level Insight into a DAX type
            # error. Nothing surfaced until Power Query actually ran.
            mapping[column] = ("boolean", "type logical")
        elif pd.api.types.is_integer_dtype(series):
            mapping[column] = ("int64", "Int64.Type")
        elif pd.api.types.is_float_dtype(series):
            mapping[column] = ("double", "type number")
        else:
            mapping[column] = ("string", "type text")
    return mapping


def format_string(column: str, data_type: str) -> str:
    """
    Sensible display formatting, so cards do not render 14 decimal places.

    Returns "" for types that take no format string. Booleans render True/False
    without one, and a numeric format on a text column is ignored by Power BI —
    it was only ever noise in the TMDL.
    """
    lowered = column.lower()
    if data_type in ("boolean", "string"):
        return ""
    if data_type == "dateTime":
        return "yyyy-mm-dd"
    if data_type == "int64":
        return "#,0"
    if any(k in lowered for k in ("gbp", "revenue", "profit", "cost", "spend", "margin_gbp")):
        return '\\£#,0.00;(\\£#,0.00);\\£#,0.00'
    if lowered.endswith("_pct") or "percent" in lowered or lowered.endswith("_rate"):
        return "#,0.00"
    return "#,0.0000"


# ---------------------------------------------------------------------------
# Measure parsing
# ---------------------------------------------------------------------------

def parse_measures(path: Path) -> Dict[str, str]:
    """
    Read `measures.dax` into {name: expression}.

    Measures are separated by blank lines; `//` lines are comments and are not
    part of the expression.
    """
    text = path.read_text(encoding="utf-8")
    measures: Dict[str, str] = {}
    current_name: str | None = None
    current_lines: List[str] = []

    def flush() -> None:
        if current_name and current_lines:
            expression = "\n".join(current_lines).strip()
            if expression:
                measures[current_name] = expression

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("//") or not line.strip():
            if not line.strip():
                flush()
                current_name, current_lines = None, []
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9 %+#/&'\-\.]*?)\s*=\s*$", line)
        if match and current_name is None:
            current_name = match.group(1).strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    flush()
    return measures


# ---------------------------------------------------------------------------
# TMDL emission
# ---------------------------------------------------------------------------

def m_partition(table: str, types: Dict[str, Tuple[str, str]]) -> str:
    """Power Query that reads the exported CSV and applies explicit types."""
    transforms = ", ".join(
        f'{{"{column}", {pq_type}}}' for column, (_, pq_type) in types.items()
    )
    return (
        "let\n"
        f'    Source = Csv.Document(File.Contents(DataFolder & "{table}.csv"), '
        "[Delimiter=\",\", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n"
        "    Headers = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),\n"
        f"    Typed = Table.TransformColumnTypes(Headers, {{{transforms}}})\n"
        "in\n"
        "    Typed"
    )


def table_tmdl(table: str, frame: pd.DataFrame, measures: Dict[str, str]) -> str:
    types = infer_types(frame)
    lines = [f"table {table}", ""]

    for name, expression in measures.items():
        lines.append(f"\tmeasure '{name}' =")
        for expression_line in expression.splitlines():
            lines.append(f"\t\t\t{expression_line}")
        # formatString is a property of the measure block, so it must follow the
        # expression without an intervening blank line - a blank line closes the
        # block and TMDL then reads the property as belonging to the table.
        if "%" in name or "pp" in name.split():
            lines.append("\t\tformatString: #,0.00")
        elif any(k in name for k in ("Revenue", "Profit", "Spend")):
            lines.append("\t\tformatString: \\£#,0.00;(\\£#,0.00);\\£#,0.00")
        lines.append("")

    for column, (data_type, _) in types.items():
        lines.append(f"\tcolumn {column}")
        lines.append(f"\t\tdataType: {data_type}")
        lines.append(f"\t\tsourceColumn: {column}")
        fmt = format_string(column, data_type)
        if fmt:
            lines.append(f"\t\tformatString: {fmt}")
        if table == "dim_calendar" and column == "date":
            lines.append("\t\tisKey")
        lines.append("")

    for column, expression, data_type, fmt in CALCULATED_COLUMNS.get(table, []):
        lines.append(f"\tcolumn {column} = {expression}")
        lines.append(f"\t\tdataType: {data_type}")
        lines.append(f"\t\tformatString: {fmt}")
        lines.append("\t\tsummarizeBy: none")
        lines.append("")

    # Revenue LY uses DATEADD, which is only dependable against a table Power BI
    # recognises as a date table. dim_calendar qualifies: `date` is a key and the
    # range is a contiguous 731 days.
    if table == "dim_calendar":
        lines.append("\tdataCategory: Time")
        lines.append("")

    lines.append(f"\tpartition {table} = m")
    lines.append("\t\tmode: import")
    lines.append("\t\tsource =")
    for source_line in m_partition(table, types).splitlines():
        lines.append(f"\t\t\t\t{source_line}")
    lines.append("")
    return "\n".join(lines)


def relationships_tmdl() -> str:
    blocks = []
    for index, (from_table, from_column, to_table, to_column) in enumerate(RELATIONSHIPS, 1):
        blocks.append(
            f"relationship rel_{index}\n"
            f"\tfromColumn: {from_table}.{from_column}\n"
            f"\ttoColumn: {to_table}.{to_column}\n"
        )
    return "\n".join(blocks)


def what_if_tmdl() -> str:
    """
    The `Uplift Scenario` what-if table as a calculated table.

    `Scenario Incremental Profit` reads SELECTEDVALUE('Uplift Scenario'[Uplift
    Scenario], 1), so without this table that measure silently falls back to its
    default of 1 and the What-If page does nothing. GENERATESERIES names its
    column `Value`; Desktop renames it to the parameter name, which is why the
    column is `Uplift Scenario` over `sourceColumn: [Value]`.

    Deliberately carries no measure: measures.dax is the measure library and
    `tests/test_powerbi.py` asserts every measure in it has a home table with
    backing data. Nothing references a `Uplift Scenario Value` measure, so
    inventing one here would only break that invariant.
    """
    lo = int(round(WHAT_IF_MIN * 100))
    hi = int(round(WHAT_IF_MAX * 100))
    return (
        f"table '{WHAT_IF_TABLE}'\n\n"
        f"\tcolumn '{WHAT_IF_TABLE}'\n"
        # `decimal`, not `double`. Stepping through binary floating point drifts,
        # so a double series ended at 1.25 rather than 1.30 and no stored value
        # exactly equalled what the slicer computed.
        "\t\tdataType: decimal\n"
        "\t\tsourceColumn: [Value]\n"
        "\t\tformatString: #,0.00\n"
        "\t\tsummarizeBy: none\n"
        # Without ParameterMetadata the slicer renders as a checkbox list rather
        # than a what-if slider. Payload copied from the model Desktop wrote,
        # not guessed.
        "\n\t\textendedProperty ParameterMetadata =\n"
        "\t\t\t\t{\n"
        '\t\t\t\t  "version": 0\n'
        "\t\t\t\t}\n\n"
        "\t\tannotation SummarizationSetBy = User\n\n"
        f"\tpartition '{WHAT_IF_TABLE}' = calculated\n"
        "\t\tmode: import\n"
        # Built from exact integers so every step is representable. 0.01 steps,
        # not 0.05: dax_parity.csv expects a 0.88 multiplier, which is not a
        # member of a 0.05 series at all.
        f"\t\tsource = SELECTCOLUMNS ( GENERATESERIES ( {lo}, {hi}, 1 ), "
        '"Value", DIVIDE ( [Value], 100 ) )\n'
    )


# ---------------------------------------------------------------------------
# The report artifact
# ---------------------------------------------------------------------------

def _json(document: Dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def write_report(root: Path, force: bool = False) -> Path:
    """
    Write the PBIR report artifact the `.pbip` points at.

    A `.pbip` is only a shortcut to a report; without this folder Desktop has
    nothing to open no matter how valid the semantic model is.

    **This is a scaffolder, not the owner of the report.** The committed report
    carries 66 hand-authored visuals that nothing here can reproduce, so by
    default this writes only what is missing and never overwrites an existing
    file. Schema drift in a scaffold is a smaller hazard than silently deleting
    a day of visual authoring. Pass `force=True` for a clean re-scaffold, which
    is destructive by design.

    PBIR (a `definition/` folder), not PBIR-Legacy (`report.json`): the legacy
    format is documented as not supporting external editing, while every PBIR
    file has a public schema.
    """
    report_root = root / f"{PROJECT_NAME}.Report"
    definition = report_root / "definition"
    pages_dir = definition / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    def put(path: Path, text: str) -> None:
        """Write only if absent, unless forcing."""
        if force or not path.exists():
            path.write_text(text, encoding="utf-8")

    # No $schema. Desktop writes this file without one, and adding it is the
    # kind of plausible-looking extra that a strict validator rejects.
    put(report_root / "definition.pbir", _json({
        "version": "4.0",
        # Relative, forward slashes, no absolute paths — a byPath reference
        # is what makes Desktop open the semantic model in edit mode too.
        "datasetReference": {"byPath": {"path": f"../{PROJECT_NAME}.SemanticModel"}},
    }))
    put(report_root / ".platform", _json({
        "$schema": PLATFORM_SCHEMA,
        "metadata": {"type": "Report", "displayName": PROJECT_NAME},
        "config": {"version": "2.0", "logicalId": "00000000-0000-0000-0000-000000000002"},
    }))

    put(definition / "version.json", _json({
        "$schema": PBIR_VERSION_SCHEMA,
        "version": PBIR_DEFINITION_VERSION,
    }))
    put(definition / "report.json", _json({
        "$schema": PBIR_REPORT_SCHEMA,
        # Microsoft's own documented base theme pair. Inventing a name here
        # gets silently dropped rather than resolved.
        "themeCollection": {
            "baseTheme": {
                "name": "CY24SU06",
                "reportVersionAtImport": "5.55",
                "type": "SharedResources",
            }
        },
        "settings": {"useEnhancedTooltips": True},
    }))
    put(pages_dir / "pages.json", _json({
        "$schema": PBIR_PAGES_SCHEMA,
        "pageOrder": [name for name, _ in PAGES],
        "activePageName": PAGES[0][0],
    }))

    for name, display_name in PAGES:
        page_dir = pages_dir / name
        page_dir.mkdir(parents=True, exist_ok=True)
        put(page_dir / "page.json", _json({
            "$schema": PBIR_PAGE_SCHEMA,
            "name": name,
            "displayName": display_name,
            "displayOption": "FitToPage",
            # Required for every displayOption except DeprecatedDynamic.
            "width": 1280,
            "height": 720,
        }))

    return report_root


def data_folder_literal(path: str | Path) -> str:
    """
    Normalise a path for the `DataFolder` M parameter.

    Forward slashes, exactly one trailing separator. Windows file APIs accept
    `/` perfectly well, so one form works on both platforms — and it avoids a
    literal ending in a backslash immediately before the closing quote.

    Note there is deliberately no backslash escaping. Backslash is not an escape
    character in Power Query text; M escapes with `#(...)` and doubles quotes.
    Doubling separators here would emit `C:\\\\Users\\\\...`, which resolves to
    nothing. It went unnoticed for as long as every generated path was macOS.
    """
    return str(path).replace("\\", "/").rstrip("/") + "/"


def build(
    data_folder: str | None = None,
    output_root: str | Path | None = None,
    force_report: bool = False,
) -> Dict[str, object]:
    """
    Generate the semantic model and report.

    `data_folder` overrides only the path *written into* the model, for building
    against a machine other than this one. The CSVs are still read from
    `output_dir()` to infer column types — those must be local and real, so the
    two paths are deliberately not the same knob.

    `output_root` redirects everything this writes. Tests must pass it: the
    committed `powerbi/` tree holds 66 hand-authored visuals and a model Desktop
    has enriched, none of which this function can reproduce, so a test that
    called `build()` bare would quietly destroy work.

    `force_report` re-scaffolds the report from scratch. Off by default — see
    `write_report`.
    """
    data = output_dir()
    source = powerbi_dir()  # measures.dax is a hand-authored input, always read from here
    root = Path(output_root) if output_root is not None else source
    root.mkdir(parents=True, exist_ok=True)
    model_root = root / f"{PROJECT_NAME}.SemanticModel"
    definition = model_root / "definition"
    tables_dir = definition / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    measures = parse_measures(source / "measures.dax")
    LOGGER.info("Parsed %d measures from measures.dax", len(measures))

    all_tables = DIMENSIONS + FACTS + RESULTS
    written: List[str] = []
    for table in all_tables:
        source = data / f"{table}.csv"
        if not source.exists():
            LOGGER.warning("Skipping %s — %s not found", table, source)
            continue
        frame = pd.read_csv(source, nrows=5000)
        table_measures = {
            name: measures[name]
            for name in MEASURE_HOME.get(table, [])
            if name in measures
        }
        (tables_dir / f"{table}.tmdl").write_text(
            table_tmdl(table, frame, table_measures), encoding="utf-8"
        )
        written.append(table)

    # The data folder is a parameter so the model is portable: a reviewer who
    # clones the repo elsewhere repoints one value rather than every query.
    #
    # The `meta [IsParameterQuery=...]` record is what makes it an actual
    # parameter. Without it this is a plain expression, invisible to Home >
    # Transform data > Edit parameters, and the only way to move the path is to
    # hand-edit TMDL — which is exactly what this project tells people not to do.
    #
    # Deliberately no `queryGroup`. A queryGroup property must name a group
    # declared on the model, and referencing an undeclared one makes Desktop
    # refuse the entire database with "refers to an object which cannot be
    # found". It only labels a folder in the Power Query sidebar; the parameter
    # shows up in Edit parameters regardless, because that list keys off
    # IsParameterQuery. With one parameter there is nothing to organise.
    (definition / "expressions.tmdl").write_text(
        "expression DataFolder = "
        '"' + data_folder_literal(data_folder or data) + '"'
        ' meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'
        "\n\tlineageTag: data-folder-parameter\n\n"
        "\tannotation PBI_NavigationStepName = Navigation\n",
        encoding="utf-8",
    )

    (tables_dir / f"{WHAT_IF_TABLE}.tmdl").write_text(what_if_tmdl(), encoding="utf-8")

    (definition / "relationships.tmdl").write_text(relationships_tmdl(), encoding="utf-8")
    (definition / "model.tmdl").write_text(
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tdiscourageImplicitMeasures\n"
        "\tsourceQueryCulture: en-GB\n"
        "\tdataAccessOptions\n"
        "\t\tlegacyRedirects\n"
        "\t\treturnErrorValuesAsNull\n\n"
        + "".join(f"ref table {table}\n" for table in written)
        # `written` is CSV-driven; the what-if table is calculated and has no
        # export behind it, so its ref is appended separately.
        + f"ref table '{WHAT_IF_TABLE}'\n"
        + "\nref expression DataFolder\n",
        encoding="utf-8",
    )
    (definition / "database.tmdl").write_text(
        "database\n\tcompatibilityLevel: 1567\n", encoding="utf-8"
    )
    (model_root / "definition.pbism").write_text(
        '{\n  "version": "4.0",\n  "settings": {}\n}\n', encoding="utf-8"
    )
    (model_root / ".platform").write_text(
        _json({
            "$schema": PLATFORM_SCHEMA,
            "metadata": {"type": "SemanticModel", "displayName": PROJECT_NAME},
            "config": {"version": "2.0", "logicalId": "00000000-0000-0000-0000-000000000001"},
        }),
        encoding="utf-8",
    )

    report_root = write_report(root, force=force_report)

    # `artifacts[].report.path` is a relative path to the report *folder* — the
    # convention Desktop itself writes when it saves a project.
    (root / f"{PROJECT_NAME}.pbip").write_text(
        # No $schema. Desktop omits it here and in definition.pbir, and the one
        # time this file guessed one it named the wrong item's schema and
        # blocked the project from opening. Omitting is both what the consumer
        # does and the option that cannot be wrong.
        _json({
            "version": "1.0",
            "artifacts": [{"report": {"path": f"{PROJECT_NAME}.Report"}}],
            "settings": {"enableAutoRecovery": True},
        }),
        encoding="utf-8",
    )

    return {
        "tables": written + [WHAT_IF_TABLE],
        "measures": len(measures),
        "root": model_root,
        "report": report_root,
        "data_folder": data_folder_literal(data_folder or data),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the PBIP project and its TMDL semantic model.",
        epilog=(
            "Example, targeting a Windows machine: --data-folder "
            "'C:/Users/you/Desktop/power-bi-northstar/powerbi_data/'"
        ),
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        metavar="PATH",
        help=(
            "Path written into the model's DataFolder parameter, for building "
            "against a machine other than this one. The CSVs are still read "
            "locally. Defaults to this machine's export directory."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build(data_folder=args.data_folder)
    print(f"\nSemantic model written to {result['root']}")
    print(f"  tables:   {len(result['tables'])}")
    print(f"  measures: {result['measures']}")
    print(f"  data:     {result['data_folder']}")
    print(f"Report written to {result['report']}")
    print(f"  pages:    {len(PAGES)} (existing pages preserved — see powerbi/PAGE_SPEC.md)")

    # The one value a cloner must supply by hand. Printing it in the exact form
    # the parameter expects turns "edit the TMDL and hope" into copy-and-paste;
    # the committed literal is a placeholder, so this is not optional for anyone
    # who did not generate the model themselves.
    print("\nBefore refreshing in Desktop, set the DataFolder parameter:")
    print("  Home → Transform data → Edit parameters → DataFolder")
    print(f"  {data_folder_literal(output_dir())}")
    print("  (keep the trailing separator, then Apply changes and Refresh)")
