"""Shared data contracts for the performance figures."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from nature_style import ensure_columns


SHOT_COLUMNS = ["25-shot", "50-shot", "75-shot", "100-shot"]

GROUPS = {
    "Chemical inputs": [
        "Metal Source",
        "Organic Linkers Source",
        "Modulator Source",
        "Solvent Source",
        "Quantity of Metal",
        "Quantity of Organic Linkers",
        "Quantity of Modulator",
        "Quantity of Solvent",
    ],
    "Synthesis conditions": ["pH", "Synthesis Temperature", "Synthesis Time", "Equipment"],
    "Crystallization outcomes": ["Crystal Morphology", "Yield"],
}


def load_shot(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    ensure_columns(frame, ["model", "Cosine"], path)
    frame = frame.loc[:, ["model", "Cosine"]].copy()
    frame["shot"] = frame["model"].map(
        lambda label: int(re.search(r"\d+", str(label)).group()) if re.search(r"\d+", str(label)) else np.nan
    )
    frame["Cosine"] = pd.to_numeric(frame["Cosine"], errors="raise")
    if frame[["shot", "Cosine"]].isna().any().any():
        raise ValueError(f"{path.name} contains a model without a numeric shot count")
    return frame.sort_values("shot")


def summarize_coreference(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    if "mean" in xls.sheet_names:
        frame = pd.read_excel(xls, sheet_name="mean")
        ensure_columns(frame, ["pub", "total", "t_count", "non_t", "t_percent"], path)
        summary = frame[["pub", "total", "t_count", "non_t", "t_percent"]].rename(
            columns={"t_count": "target", "non_t": "other", "t_percent": "target_rate"}
        )
        summary = summary.copy()
        summary["pub"] = summary["pub"].astype(str).str.strip().replace({"Total": "Overall"})
        numeric = ["total", "target", "other", "target_rate"]
        summary[numeric] = summary[numeric].apply(pd.to_numeric, errors="raise")
        summary["target_rate"] /= 100
        if not np.isfinite(summary[numeric].to_numpy()).all():
            raise ValueError(f"{path.name} contains non-finite mean values")
        if summary["pub"].eq("Overall").sum() != 1:
            raise ValueError(f"{path.name} must contain exactly one Total mean row")
        return summary

    frame = pd.read_excel(path)
    ensure_columns(frame, ["Label", "pub"], path)
    frame = frame.loc[:, ["Label", "pub"]].copy()
    frame["pub"] = frame["pub"].fillna("Unspecified").astype(str)
    summary = (
        frame.groupby("pub", sort=False)["Label"]
        .agg(total="size", target=lambda values: values.astype(str).str.lower().eq("t").sum())
        .reset_index()
    )
    summary = summary[summary["pub"].str.lower() != "total"].copy()
    if summary.empty:
        raise ValueError(f"{path.name} contains no publisher groups")
    summary["other"] = summary["total"] - summary["target"]
    summary["target_rate"] = summary["target"] / summary["total"]
    overall_total = int(summary["total"].sum())
    overall_target = int(summary["target"].sum())
    overall = pd.DataFrame(
        {
            "pub": ["Overall"],
            "total": [overall_total],
            "target": [overall_target],
            "other": [overall_total - overall_target],
            "target_rate": [overall_target / overall_total],
        }
    )
    return pd.concat([summary, overall], ignore_index=True)


def load_paragraph(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frame = pd.read_excel(xls, sheet_name="mean" if "mean" in xls.sheet_names else 0)
    ensure_columns(frame, ["pub", *SHOT_COLUMNS], path)
    frame = frame.loc[:, ["pub", *SHOT_COLUMNS]].copy()
    frame[SHOT_COLUMNS] = frame[SHOT_COLUMNS].apply(pd.to_numeric, errors="raise")
    if frame["pub"].isna().any() or not np.isfinite(frame[SHOT_COLUMNS].to_numpy()).all():
        raise ValueError(f"{path.name} contains missing or non-finite plotting values")
    if frame["pub"].astype(str).str.lower().eq("total").sum() != 1:
        raise ValueError(f"{path.name} must contain exactly one Total row")
    return frame


def load_structure(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frame = pd.read_excel(xls, sheet_name="mean" if "mean" in xls.sheet_names else 0, index_col=0)
    columns = [column for group in GROUPS.values() for column in group]
    ensure_columns(frame, columns, path)
    frame = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return frame
