from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


METRIC_COLUMN_RENAME: Dict[str, str] = {
    "n_samples": "total_examples",
    "valid_predictions": "valid_answers",
    "invalid_predictions": "invalid_answers",
    "empty_raw_outputs": "empty_answers",
    "Accuracy": "overall_accuracy",
    "Accuracy_valid_only": "valid_only_accuracy",
    "Precision_0": "precision_label_0",
    "Recall_0": "recall_label_0",
    "F1_0": "f1_label_0",
    "Precision_1": "precision_label_1",
    "Recall_1": "recall_label_1",
    "F1_1": "f1_label_1",
    "actual_0_pred_0": "true_0_pred_0",
    "actual_0_pred_1": "true_0_pred_1",
    "actual_1_pred_0": "true_1_pred_0",
    "actual_1_pred_1": "true_1_pred_1",
}


TARGET_METRIC_COLUMNS: List[str] = [
    "total_examples",
    "valid_answers",
    "invalid_answers",
    "empty_answers",
    "correct_answers",
    "overall_accuracy",
    "valid_only_accuracy",
    "precision_label_0",
    "recall_label_0",
    "f1_label_0",
    "precision_label_1",
    "recall_label_1",
    "f1_label_1",
    "true_0_pred_0",
    "true_0_pred_1",
    "true_1_pred_0",
    "true_1_pred_1",
]


def add_correct_answers(df: pd.DataFrame) -> pd.DataFrame:
    """
    In 3.py correct_answers is present.
    In 3_MAIA-like.py it is absent, but it can be reconstructed
    as the diagonal of the valid confusion matrix.
    """
    if "correct_answers" not in df.columns:
        df["correct_answers"] = df["true_0_pred_0"] + df["true_1_pred_1"]

    return df


def convert_metric_sheet(df: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """
    Converts global_metrics, category_metrics and video_metrics
    from the 3_MAIA-like schema to the 3.py schema.
    """
    df = df.copy()
    df = df.rename(columns=METRIC_COLUMN_RENAME)
    df = add_correct_answers(df)

    final_columns = [entity_column] + TARGET_METRIC_COLUMNS

    missing_columns = [col for col in final_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing columns in sheet with entity column '{entity_column}': "
            f"{missing_columns}"
        )

    return df[final_columns]


def convert_global_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts global_confusion:

    from:
        index: actual_0, actual_1
        columns: predicted_0, predicted_1

    to:
        index: true_0, true_1
        columns: pred_0, pred_1
    """
    df = df.copy()

    df.index = df.index.to_series().replace(
        {
            "actual_0": "true_0",
            "actual_1": "true_1",
        }
    )

    df = df.rename(
        columns={
            "predicted_0": "pred_0",
            "predicted_1": "pred_1",
        }
    )

    return df.loc[["true_0", "true_1"], ["pred_0", "pred_1"]]


def convert_category_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts category_confusion column names from MAIA-like names
    to the names used by 3.py.
    """
    df = df.copy()
    df = df.rename(columns=METRIC_COLUMN_RENAME)

    final_columns = [
        "normalized_question_category",
        "true_0_pred_0",
        "true_0_pred_1",
        "true_1_pred_0",
        "true_1_pred_1",
    ]

    missing_columns = [col for col in final_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in category_confusion: {missing_columns}")

    return df[final_columns]


def convert_workbook(input_xlsx: Path, output_xlsx: Path) -> None:
    sheets = pd.read_excel(input_xlsx, sheet_name=None)

    global_metrics = convert_metric_sheet(
        sheets["global_metrics"],
        entity_column="scope",
    )

    category_metrics = convert_metric_sheet(
        sheets["category_metrics"],
        entity_column="normalized_question_category",
    )

    video_metrics = convert_metric_sheet(
        sheets["video_metrics"],
        entity_column="video_id",
    )

    global_confusion_raw = pd.read_excel(
        input_xlsx,
        sheet_name="global_confusion",
        index_col=0,
    )
    global_confusion = convert_global_confusion(global_confusion_raw)

    category_confusion = convert_category_confusion(
        sheets["category_confusion"],
    )

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        global_metrics.to_excel(writer, sheet_name="global_metrics", index=False)
        category_metrics.to_excel(writer, sheet_name="category_metrics", index=False)
        video_metrics.to_excel(writer, sheet_name="video_metrics", index=False)
        global_confusion.to_excel(writer, sheet_name="global_confusion")
        category_confusion.to_excel(writer, sheet_name="category_confusion", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert 3_MAIA-like_evaluation_summary.xlsx to the same column schema "
            "used by 3_evaluation_summary.xlsx."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to 3_MAIA-like_evaluation_summary.xlsx",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path of the converted output XLSX",
    )

    args = parser.parse_args()

    convert_workbook(
        input_xlsx=args.input,
        output_xlsx=args.output,
    )

    print(f"Converted workbook saved to: {args.output}")


if __name__ == "__main__":
    main()