from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


SECTION_PATTERN = re.compile(
    r"(?ms)^([A-Z][A-Z \-]+)\n-+\n(.*?)(?=^[A-Z][A-Z \-]+\n-+\n|\Z)"
)
NUMERIC_COLUMNS = [
    "n_samples",
    "valid_predictions",
    "invalid_predictions",
    "empty_raw_outputs",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "actual_0_pred_0",
    "actual_0_pred_1",
    "actual_1_pred_0",
    "actual_1_pred_1",
]
PAGE_WIDTH_POINTS = A4[0] - (1.5 * cm) - (1.5 * cm)


@dataclass(frozen=True)
class ParsedReport:
    model_name: str
    source_path: Path
    global_metrics: pd.DataFrame
    category_metrics: pd.DataFrame
    video_metrics: pd.DataFrame


def derive_model_name(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_metrics_report$", "", stem)
    stem = re.sub(r"[-_]+", " ", stem)
    return stem.title().strip()


def load_sections(raw_text: str) -> dict[str, str]:
    return {
        match.group(1).strip(): match.group(2).strip("\n")
        for match in SECTION_PATTERN.finditer(raw_text)
    }


def read_fixed_width_table(section_text: str) -> pd.DataFrame:
    df = pd.read_fwf(StringIO(section_text))
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_report(path: Path) -> ParsedReport:
    raw_text = path.read_text(encoding="utf-8")
    sections = load_sections(raw_text)

    required_sections = [
        "GLOBAL PERFORMANCE",
        "PER NORMALIZED QUESTION CATEGORY",
        "PER VIDEO PERFORMANCE",
    ]
    missing = [name for name in required_sections if name not in sections]
    if missing:
        raise ValueError(f"Missing sections in {path.name}: {missing}")

    return ParsedReport(
        model_name=derive_model_name(path),
        source_path=path,
        global_metrics=read_fixed_width_table(sections["GLOBAL PERFORMANCE"]),
        category_metrics=read_fixed_width_table(
            sections["PER NORMALIZED QUESTION CATEGORY"]
        ),
        video_metrics=read_fixed_width_table(sections["PER VIDEO PERFORMANCE"]),
    )


def evaluate_score(metric_name: str, value: float) -> str:
    if pd.isna(value):
        return "Not available"

    if metric_name in {"accuracy", "precision_macro", "recall_macro", "f1_macro", "coverage_rate"}:
        if value >= 0.75:
            return "Excellent"
        if value >= 0.50:
            return "Strong"
        if value >= 0.25:
            return "Moderate"
        if value >= 0.10:
            return "Weak"
        return "Critical"

    if metric_name == "invalid_rate":
        if value <= 0.05:
            return "Excellent"
        if value <= 0.15:
            return "Acceptable"
        if value <= 0.30:
            return "Concerning"
        return "Critical"

    return "Unrated"


def build_global_comparison(reports: list[ParsedReport]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for report in reports:
        row = report.global_metrics.iloc[0].to_dict()
        n_samples = float(row["n_samples"])
        valid_predictions = float(row["valid_predictions"])
        invalid_predictions = float(row["invalid_predictions"])

        coverage_rate = valid_predictions / n_samples if n_samples else math.nan
        invalid_rate = invalid_predictions / n_samples if n_samples else math.nan

        rows.append(
            {
                "model": report.model_name,
                "source_file": report.source_path.name,
                **row,
                "coverage_rate": coverage_rate,
                "invalid_rate": invalid_rate,
                "accuracy_band": evaluate_score("accuracy", float(row["accuracy"])),
                "f1_band": evaluate_score("f1_macro", float(row["f1_macro"])),
                "coverage_band": evaluate_score("coverage_rate", coverage_rate),
                "invalid_band": evaluate_score("invalid_rate", invalid_rate),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(by=["f1_macro", "accuracy"], ascending=False).reset_index(drop=True)


def build_category_comparison(reports: list[ParsedReport]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for report in reports:
        frame = report.category_metrics.copy()
        frame["model"] = report.model_name
        frame["coverage_rate"] = frame["valid_predictions"] / frame["n_samples"]
        frame["invalid_rate"] = frame["invalid_predictions"] / frame["n_samples"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    pivot = (
        combined.pivot_table(
            index="normalized_question_category",
            columns="model",
            values=["accuracy", "f1_macro", "coverage_rate", "invalid_rate"],
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns = [
        "normalized_question_category"
        if column == ("normalized_question_category", "")
        else f"{column[0]}__{column[1]}"
        for column in pivot.columns
    ]

    model_names = [report.model_name for report in reports]
    pivot["best_model_f1"] = pivot[[f"f1_macro__{m}" for m in model_names]].idxmax(axis=1)
    pivot["best_model_f1"] = pivot["best_model_f1"].str.replace("f1_macro__", "", regex=False)
    pivot["f1_spread"] = (
        pivot[[f"f1_macro__{m}" for m in model_names]].max(axis=1)
        - pivot[[f"f1_macro__{m}" for m in model_names]].min(axis=1)
    )
    return pivot.sort_values(by="f1_spread", ascending=False).reset_index(drop=True)


def build_video_comparison(reports: list[ParsedReport]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for report in reports:
        frame = report.video_metrics.copy()
        frame["model"] = report.model_name
        frame["coverage_rate"] = frame["valid_predictions"] / frame["n_samples"]
        frame["invalid_rate"] = frame["invalid_predictions"] / frame["n_samples"]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    pivot = (
        combined.pivot_table(
            index="video_id",
            columns="model",
            values=["accuracy", "f1_macro", "coverage_rate", "invalid_rate"],
        )
        .sort_index()
        .reset_index()
    )
    pivot.columns = [
        "video_id"
        if column == ("video_id", "")
        else f"{column[0]}__{column[1]}"
        for column in pivot.columns
    ]

    model_names = [report.model_name for report in reports]
    pivot["f1_spread"] = (
        pivot[[f"f1_macro__{m}" for m in model_names]].max(axis=1)
        - pivot[[f"f1_macro__{m}" for m in model_names]].min(axis=1)
    )
    return pivot.sort_values(by="f1_spread", ascending=False).reset_index(drop=True)


def select_metric_winners(global_df: pd.DataFrame) -> list[str]:
    metric_specs = [
        ("accuracy", False),
        ("precision_macro", False),
        ("recall_macro", False),
        ("f1_macro", False),
        ("coverage_rate", False),
        ("invalid_rate", True),
    ]
    lines: list[str] = []
    for metric_name, lower_is_better in metric_specs:
        ordered = global_df.sort_values(by=metric_name, ascending=lower_is_better)
        best = ordered.iloc[0]
        lines.append(
            f"{metric_name.replace('_', ' ').title()}: {best['model']} ({best[metric_name]:.4f})"
        )
    return lines


def build_threshold_reference() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Accuracy / Precision / Recall / F1 / Coverage", ">= 0.75", "Excellent"),
            ("Accuracy / Precision / Recall / F1 / Coverage", "0.50 - 0.7499", "Strong"),
            ("Accuracy / Precision / Recall / F1 / Coverage", "0.25 - 0.4999", "Moderate"),
            ("Accuracy / Precision / Recall / F1 / Coverage", "0.10 - 0.2499", "Weak"),
            ("Accuracy / Precision / Recall / F1 / Coverage", "< 0.10", "Critical"),
            ("Invalid rate", "<= 0.05", "Excellent"),
            ("Invalid rate", "0.0501 - 0.15", "Acceptable"),
            ("Invalid rate", "0.1501 - 0.30", "Concerning"),
            ("Invalid rate", "> 0.30", "Critical"),
        ],
        columns=["Metric family", "Threshold", "Interpretation"],
    )


def make_grouped_bar_chart(
    global_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str,
) -> None:
    x_positions = list(range(len(global_df)))
    width = 0.18

    plt.figure(figsize=(10.5, 5.5))
    for idx, metric in enumerate(metrics):
        offsets = [x + (idx - (len(metrics) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, global_df[metric], width=width, label=metric.replace("_", " ").title())

    plt.xticks(x_positions, global_df["model"], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_valid_invalid_chart(global_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(global_df["model"], global_df["coverage_rate"], label="Valid prediction rate")
    plt.bar(
        global_df["model"],
        global_df["invalid_rate"],
        bottom=global_df["coverage_rate"],
        label="Invalid prediction rate",
    )
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("Prediction Coverage vs Invalid Output Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_category_gap_chart(category_df: pd.DataFrame, model_names: list[str], output_path: Path) -> None:
    if len(model_names) < 2:
        return

    df = category_df.copy()
    first_model = model_names[0]
    second_model = model_names[1]
    df["gap"] = df[f"f1_macro__{second_model}"] - df[f"f1_macro__{first_model}"]
    df = df.sort_values(by="gap", ascending=True)

    plt.figure(figsize=(10.5, 6.5))
    plt.barh(df["normalized_question_category"], df["gap"])
    plt.xlabel(f"F1 difference ({second_model} - {first_model})")
    plt.ylabel("Normalized question category")
    plt.title("Category-Level F1 Gap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_video_gap_chart(video_df: pd.DataFrame, model_names: list[str], output_path: Path) -> None:
    if len(model_names) < 2:
        return

    df = video_df.copy()
    first_model = model_names[0]
    second_model = model_names[1]
    df["gap"] = df[f"f1_macro__{second_model}"] - df[f"f1_macro__{first_model}"]
    df = df.sort_values(by="gap", ascending=False).head(20).sort_values(by="gap", ascending=True)

    plt.figure(figsize=(10.5, 7))
    plt.barh(df["video_id"], df["gap"])
    plt.xlabel(f"F1 difference ({second_model} - {first_model})")
    plt.ylabel("Video ID")
    plt.title("Top 20 Video-Level F1 Gaps")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def paragraph(text: str, styles: StyleSheet1, style_name: str = "BodyText") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def sanitize_header(name: str) -> str:
    return name.replace("__", "<br/>").replace("_", " ").title()


def compute_column_widths(
    df: pd.DataFrame,
    total_width: float,
    min_first_col: float = 4.2 * cm,
) -> list[float]:
    columns = list(df.columns)
    if not columns:
        return []

    if len(columns) == 1:
        return [total_width]

    first_width = min(max(min_first_col, total_width * 0.32), total_width * 0.45)
    remaining = total_width - first_width
    other_count = len(columns) - 1
    other_width = remaining / other_count
    return [first_width] + [other_width] * other_count


def dataframe_to_long_table(
    df: pd.DataFrame,
    styles: StyleSheet1,
    max_rows: int | None = None,
    column_formats: Mapping[str, str] | None = None,
    col_widths: Sequence[float] | None = None,
    font_size: float = 8.0,
) -> LongTable:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)

    if column_formats:
        for column, fmt in column_formats.items():
            if column in view.columns:
                view[column] = view[column].map(
                    lambda x, _fmt=fmt: _fmt.format(x) if pd.notna(x) else ""
                )

    header_style = ParagraphStyle(
        name="TableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_CENTER,
        textColor=colors.white,
    )
    body_style = ParagraphStyle(
        name="TableBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
    numeric_style = ParagraphStyle(
        name="TableNumeric",
        parent=body_style,
        alignment=TA_CENTER,
    )

    data: list[list[Paragraph]] = []
    data.append([Paragraph(sanitize_header(str(column)), header_style) for column in view.columns])

    for _, row in view.iterrows():
        rendered_row: list[Paragraph] = []
        for column, value in row.items():
            text = "" if pd.isna(value) else str(value)
            style = numeric_style if isinstance(value, (int, float)) or column.endswith(("rate", "accuracy", "f1_macro")) else body_style
            rendered_row.append(Paragraph(text.replace("\n", "<br/>"), style))
        data.append(rendered_row)

    widths = list(col_widths) if col_widths is not None else compute_column_widths(view, PAGE_WIDTH_POINTS)

    table = LongTable(data, repeatRows=1, colWidths=widths, splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3B73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C7D0E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EEF3FA")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def chunked_detail_columns(
    id_column: str,
    model_names: Sequence[str],
    metrics: Sequence[str],
    models_per_chunk: int = 2,
) -> Iterator[list[str]]:
    for start in range(0, len(model_names), models_per_chunk):
        chunk_models = model_names[start : start + models_per_chunk]
        columns = [id_column]
        for model in chunk_models:
            for metric in metrics:
                columns.append(f"{metric}__{model}")
        yield columns


def build_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#16355C"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#16355C"),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#2A4C7F"),
            spaceBefore=6,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#4B5563"),
        )
    )
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 13
    return styles


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report_pdf(
    reports: list[ParsedReport],
    output_pdf: Path,
    chart_paths: Mapping[str, Path],
) -> None:
    styles = build_styles()

    global_df = build_global_comparison(reports)
    category_df = build_category_comparison(reports)
    video_df = build_video_comparison(reports)
    threshold_df = build_threshold_reference()
    model_names = [report.model_name for report in reports]

    best_f1_row = global_df.sort_values(by="f1_macro", ascending=False).iloc[0]
    best_accuracy_row = global_df.sort_values(by="accuracy", ascending=False).iloc[0]
    lowest_invalid_row = global_df.sort_values(by="invalid_rate", ascending=True).iloc[0]

    executive_summary = dedent(
        f"""
        This report compares {len(reports)} model outputs generated from structured evaluation summaries.
        The strongest model by macro F1 is <b>{best_f1_row['model']}</b> ({best_f1_row['f1_macro']:.4f}),
        while the strongest model by exact accuracy is <b>{best_accuracy_row['model']}</b>
        ({best_accuracy_row['accuracy']:.4f}).
        The lowest invalid-output rate is achieved by <b>{lowest_invalid_row['model']}</b>
        ({lowest_invalid_row['invalid_rate']:.2%} invalid predictions).
        Across the current inputs, invalid predictions are the main limiting factor for every model,
        so coverage and output parsability are as important as raw classification quality.
        """
    ).strip()

    winner_lines = "<br/>".join(select_metric_winners(global_df))
    most_discriminative_categories = category_df[["normalized_question_category", "best_model_f1", "f1_spread"]].head(8)
    top_video_gaps = video_df[["video_id", "f1_spread"]].head(12)

    elements: list[object] = []
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph("Model Comparison Report", styles["ReportTitle"]))
    elements.append(
        paragraph(
            "Automated comparative analysis of evaluation summaries, thresholds, and per-slice performance.",
            styles,
            "SubHeading",
        )
    )
    elements.append(Spacer(1, 0.15 * cm))
    elements.append(paragraph(executive_summary, styles))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(paragraph(f"<b>Metric winners</b><br/>{winner_lines}", styles))
    elements.append(Spacer(1, 0.3 * cm))

    source_rows = pd.DataFrame([{"Model": r.model_name, "Source file": r.source_path.name} for r in reports])
    elements.append(Paragraph("Input files", styles["SectionHeading"]))
    elements.append(dataframe_to_long_table(source_rows, styles, col_widths=[5.2 * cm, PAGE_WIDTH_POINTS - (5.2 * cm)]))
    elements.append(Spacer(1, 0.35 * cm))

    elements.append(Paragraph("Global comparison", styles["SectionHeading"]))
    elements.append(
        paragraph(
            "The global table compares the most important aggregate indicators. Coverage rate is the proportion of parsable predictions, while invalid rate captures outputs that could not be mapped to labels 0 or 1.",
            styles,
        )
    )
    global_table_df = global_df[[
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "coverage_rate",
        "invalid_rate",
        "accuracy_band",
        "f1_band",
        "invalid_band",
    ]]
    global_col_widths = [
        3.0 * cm,
        1.35 * cm,
        1.55 * cm,
        1.45 * cm,
        1.25 * cm,
        1.55 * cm,
        1.55 * cm,
        1.55 * cm,
        1.35 * cm,
        1.55 * cm,
    ]
    elements.append(
        dataframe_to_long_table(
            global_table_df,
            styles,
            column_formats={
                "accuracy": "{:.4f}",
                "precision_macro": "{:.4f}",
                "recall_macro": "{:.4f}",
                "f1_macro": "{:.4f}",
                "coverage_rate": "{:.2%}",
                "invalid_rate": "{:.2%}",
            },
            col_widths=global_col_widths,
            font_size=7.2,
        )
    )
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Image(str(chart_paths["global_scores"]), width=16.2 * cm, height=8.2 * cm))
    elements.append(Spacer(1, 0.15 * cm))
    elements.append(Image(str(chart_paths["coverage"]), width=16.2 * cm, height=8.2 * cm))
    elements.append(PageBreak())

    elements.append(Paragraph("Threshold reference", styles["SectionHeading"]))
    elements.append(
        paragraph(
            "These thresholds are heuristic and intended for quick operational reading. They are not benchmark-specific scientific cut-offs, but they make the report easier to interpret consistently across runs.",
            styles,
        )
    )
    elements.append(
        dataframe_to_long_table(
            threshold_df,
            styles,
            col_widths=[6.4 * cm, 3.8 * cm, PAGE_WIDTH_POINTS - (6.4 * cm + 3.8 * cm)],
        )
    )
    elements.append(Spacer(1, 0.35 * cm))

    elements.append(Paragraph("Category-level analysis", styles["SectionHeading"]))
    elements.append(
        paragraph(
            "This section highlights which normalized question categories create the largest separation between models. Large F1 spreads indicate meaningful behavioral differences rather than minor noise.",
            styles,
        )
    )
    elements.append(
        dataframe_to_long_table(
            most_discriminative_categories,
            styles,
            column_formats={"f1_spread": "{:.4f}"},
            col_widths=[7.0 * cm, 4.2 * cm, PAGE_WIDTH_POINTS - (7.0 * cm + 4.2 * cm)],
        )
    )
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Image(str(chart_paths["category_gap"]), width=16.0 * cm, height=9.2 * cm))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(
        paragraph(
            "Detailed category comparison. Wide tables are automatically split into compact blocks so every column remains fully visible.",
            styles,
        )
    )

    for index, detailed_category_cols in enumerate(
        chunked_detail_columns(
            id_column="normalized_question_category",
            model_names=model_names,
            metrics=["accuracy", "f1_macro", "invalid_rate"],
            models_per_chunk=2,
        ),
        start=1,
    ):
        category_table = category_df[detailed_category_cols].copy()
        elements.append(paragraph(f"<b>Category detail block {index}</b>", styles, "Small"))
        elements.append(
            dataframe_to_long_table(
                category_table,
                styles,
                max_rows=10,
                column_formats={
                    column: "{:.4f}"
                    for column in category_table.columns
                    if column.startswith(("accuracy__", "f1_macro__"))
                }
                | {
                    column: "{:.2%}"
                    for column in category_table.columns
                    if column.startswith("invalid_rate__")
                },
                font_size=7.0,
            )
        )
        elements.append(Spacer(1, 0.18 * cm))

    elements.append(PageBreak())
    elements.append(Paragraph("Video-level analysis", styles["SectionHeading"]))
    elements.append(
        paragraph(
            "Per-video comparisons help identify where performance differences are concentrated. A model can appear similar globally while behaving very differently on individual videos.",
            styles,
        )
    )
    elements.append(
        dataframe_to_long_table(
            top_video_gaps,
            styles,
            column_formats={"f1_spread": "{:.4f}"},
            col_widths=[5.2 * cm, PAGE_WIDTH_POINTS - (5.2 * cm)],
        )
    )
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Image(str(chart_paths["video_gap"]), width=16.0 * cm, height=10.0 * cm))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(paragraph("Detailed video comparison (top 15 rows by F1 spread).", styles))

    for index, detailed_video_cols in enumerate(
        chunked_detail_columns(
            id_column="video_id",
            model_names=model_names,
            metrics=["f1_macro", "invalid_rate"],
            models_per_chunk=3,
        ),
        start=1,
    ):
        video_table = video_df[detailed_video_cols].copy()
        elements.append(paragraph(f"<b>Video detail block {index}</b>", styles, "Small"))
        elements.append(
            dataframe_to_long_table(
                video_table,
                styles,
                max_rows=15,
                column_formats={
                    column: "{:.4f}"
                    for column in video_table.columns
                    if column.startswith("f1_macro__")
                }
                | {
                    column: "{:.2%}"
                    for column in video_table.columns
                    if column.startswith("invalid_rate__")
                },
                col_widths=compute_column_widths(video_table, PAGE_WIDTH_POINTS, min_first_col=2.7 * cm),
                font_size=7.2,
            )
        )
        elements.append(Spacer(1, 0.18 * cm))

    closing_notes = dedent(
        """
        <b>Reading guide.</b> When one model has higher accuracy but much lower coverage, the apparent gain may be misleading.
        In these inputs, invalid prediction rate is the dominant operational bottleneck, so the most useful next step is to
        improve output formatting robustness first, then revisit class-level quality. Category-level spreads can guide targeted
        prompt tuning, decoding constraints, or post-processing rules.
        """
    ).strip()
    elements.append(paragraph(closing_notes, styles))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(
        paragraph(
            "Generated automatically from plain-text evaluation summaries using pandas, matplotlib, and ReportLab.",
            styles,
            "Small",
        )
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.5 * cm,
        title="Model Comparison Report",
        author="OpenAI",
    )
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_report(input_paths: Iterable[Path], output_pdf: Path) -> None:
    reports = [parse_report(path) for path in input_paths]
    if len(reports) < 2:
        raise ValueError("At least two input reports are required for comparison.")

    asset_dir = output_pdf.with_suffix("")
    ensure_directory(asset_dir)

    global_df = build_global_comparison(reports)
    category_df = build_category_comparison(reports)
    video_df = build_video_comparison(reports)
    model_names = [report.model_name for report in reports]

    chart_paths = {
        "global_scores": asset_dir / "global_scores.png",
        "coverage": asset_dir / "coverage_vs_invalid.png",
        "category_gap": asset_dir / "category_f1_gap.png",
        "video_gap": asset_dir / "video_f1_gap.png",
    }

    make_grouped_bar_chart(
        global_df,
        metrics=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        output_path=chart_paths["global_scores"],
        title="Global Metric Comparison",
    )
    make_valid_invalid_chart(global_df, chart_paths["coverage"])
    make_category_gap_chart(category_df, model_names, chart_paths["category_gap"])
    make_video_gap_chart(video_df, model_names, chart_paths["video_gap"])

    build_report_pdf(reports, output_pdf, chart_paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more plain-text model evaluation reports and generate a PDF report with tables, thresholds, and charts."
        )
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Paths to report text files. Provide at least two files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_comparison_report.pdf"),
        help="Output PDF path.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    generate_report(args.input_files, args.output)


if __name__ == "__main__":
    main()
