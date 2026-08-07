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
measures, formatting - and the `.pbip` project files. It does not hand-author the
report's visual layer. A `report.json` is a large, position-sensitive document,
and authoring one blind is where this would most likely produce something that
opens to broken visuals. `powerbi/PAGE_SPEC.md` specifies the five pages
precisely enough to assemble in Desktop instead.

None of this has been opened in Power BI Desktop, because that is not available
here. The Phase 8 report says so rather than implying the model is verified.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerbi.export import output_dir  # noqa: E402
from utils import config  # noqa: E402

LOGGER = logging.getLogger("promopulse.powerbi.tmdl")

PROJECT_NAME = "Northstar"

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
    ],
    "dose_response": [
        "Dose Response Points", "CI Coverage %", "Max Recovery Error",
        "Mean Lift at 20% Discount", "Estimated Lift %", "True Lift %",
        "Lift Recovery Gap pp",
    ],
    "spillover": [
        "Cannibalisation 1 Neighbour %", "Cannibalisation 4+ Neighbours %",
    ],
    "service_levels": [
        "Median Service Level %", "Lowest Service Level %",
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
            mapping[column] = ("int64", "Int64.Type")
        elif pd.api.types.is_integer_dtype(series):
            mapping[column] = ("int64", "Int64.Type")
        elif pd.api.types.is_float_dtype(series):
            mapping[column] = ("double", "type number")
        else:
            mapping[column] = ("string", "type text")
    return mapping


def format_string(column: str, data_type: str) -> str:
    """Sensible display formatting, so cards do not render 14 decimal places."""
    lowered = column.lower()
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
        lines.append(f"\t\tformatString: {format_string(column, data_type)}")
        if table == "dim_calendar" and column == "date":
            lines.append("\t\tisKey")
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


def build() -> Dict[str, object]:
    data = output_dir()
    root = powerbi_dir()
    model_root = root / f"{PROJECT_NAME}.SemanticModel"
    definition = model_root / "definition"
    tables_dir = definition / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    measures = parse_measures(root / "measures.dax")
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
    # clones the repo elsewhere edits one value rather than every query.
    (definition / "expressions.tmdl").write_text(
        "expression DataFolder = "
        '"' + str(data).replace("\\", "\\\\") + '/"'
        "\n\tlineageTag: data-folder-parameter\n"
        "\tqueryGroup: Parameters\n\n"
        "\tannotation PBI_NavigationStepName = Navigation\n",
        encoding="utf-8",
    )

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
        + "\nref expression DataFolder\n",
        encoding="utf-8",
    )
    (definition / "database.tmdl").write_text(
        f"database\n\tcompatibilityLevel: 1567\n", encoding="utf-8"
    )
    (model_root / "definition.pbism").write_text(
        '{\n  "version": "4.0",\n  "settings": {}\n}\n', encoding="utf-8"
    )
    (model_root / ".platform").write_text(
        '{\n'
        '  "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/'
        'platformProperties/2.0.0/schema.json",\n'
        '  "metadata": {\n'
        f'    "type": "SemanticModel",\n    "displayName": "{PROJECT_NAME}"\n'
        '  },\n'
        '  "config": {\n    "version": "2.0",\n    "logicalId": '
        '"00000000-0000-0000-0000-000000000001"\n  }\n}\n',
        encoding="utf-8",
    )

    (root / f"{PROJECT_NAME}.pbip").write_text(
        '{\n'
        '  "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/'
        'pbip/definitionProperties/1.0.0/schema.json",\n'
        '  "version": "1.0",\n'
        '  "artifacts": [\n'
        '    {\n'
        f'      "report": {{\n        "path": "{PROJECT_NAME}.Report"\n      }}\n'
        '    }\n'
        '  ],\n'
        '  "settings": {\n    "enableAutoRecovery": true\n  }\n}\n',
        encoding="utf-8",
    )

    return {"tables": written, "measures": len(measures), "root": model_root}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = build()
    print(f"\nSemantic model written to {result['root']}")
    print(f"  tables:   {len(result['tables'])}")
    print(f"  measures: {result['measures']}")
