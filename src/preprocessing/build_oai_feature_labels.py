"""
build_oai_feature_labels.py
===========================
Match OAI baseline semi-quantitative knee readings (KXR_SQ_BU00) to local images.

Matching and validation only. Does not train, and does not modify images or folds.

Usage:
  python src/preprocessing/build_oai_feature_labels.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Confirmed against kxr_sq_bu00_Contents.pdf (OAI 0.9).
FEATURE_SRC = {
    "V00XROSFM": "ost_fem_med",
    "V00XRSCFM": "scl_fem_med",
    "V00XRCYFM": "cyst_fem_med",
    "V00XRJSM": "jsn_med",
    "V00XRCHM": "chon_med",
    "V00XROSTM": "ost_tib_med",
    "V00XRSCTM": "scl_tib_med",
    "V00XRCYTM": "cyst_tib_med",
    "V00XRATTM": "att_tib_med",
    "V00XROSFL": "ost_fem_lat",
    "V00XRSCFL": "scl_fem_lat",
    "V00XRCYFL": "cyst_fem_lat",
    "V00XRJSL": "jsn_lat",
    "V00XRCHL": "chon_lat",
    "V00XROSTL": "ost_tib_lat",
    "V00XRSCTL": "scl_tib_lat",
    "V00XRCYTL": "cyst_tib_lat",
    "V00XRATTL": "att_tib_lat",
}
OST_COLS = ["ost_fem_med", "ost_tib_med", "ost_fem_lat", "ost_tib_lat"]
FEATURE_COLS = list(FEATURE_SRC.values())
STEM_LR = re.compile(r"(\d+)([LRlr])$")
STEM_NUM = re.compile(r"(\d+)_([12])$")  # auto_test uses OAI's own side codes: 1 Right, 2 Left
IMG_EXTS = {".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val", "test", "auto_test")
AGREEMENT_STOP = 0.90

DATASETS = (
    ("Dataset", PROJECT_ROOT / "Dataset"),
    ("Dataset_combined", PROJECT_ROOT / "Dataset_combined"),
    ("kneeKL299", Path(r"D:\NN & DL\KneeXrayData\ClsKLData\kneeKL299")),
)


def parse_number(cell) -> float | None:
    """'2: 2' -> 2, '0' -> 0, blank or '.: Missing ...' -> NaN."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    text = str(cell).strip()
    if not text or text.startswith(".") or "missing" in text.lower():
        return None
    head = text.split(":", 1)[0].strip()
    try:
        return float(head)
    except ValueError:
        return None


def parse_side(cell) -> str | None:
    text = "" if cell is None else str(cell).strip()
    head = text.split(":", 1)[0].strip()
    low = text.lower()
    if head == "1" or head.upper() == "R" or "right" in low:
        return "R"
    if head == "2" or head.upper() == "L" or "left" in low:
        return "L"
    return None


def load_oai(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="|", dtype=str, encoding="latin-1", keep_default_na=False)
    expected = ["ID", "SIDE", "READPRJ", "VERSION", "V00BARCDBU", "V00XRKL", *FEATURE_SRC]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    out = pd.DataFrame({
        "ID": raw["ID"].str.strip().str.zfill(7),
        "SIDE": raw["SIDE"].map(parse_side),
        "READPRJ": raw["READPRJ"].map(parse_number),
        "KL_oai": raw["V00XRKL"].map(parse_number),
    })
    for src, name in FEATURE_SRC.items():
        out[name] = raw[src].map(parse_number)
    if out["SIDE"].isna().any() or out["READPRJ"].isna().any():
        raise SystemExit("OAI rows with unreadable SIDE or READPRJ")
    # kXR_SQ_BU_Descrip.pdf: READPRJ 42 is project 37. Tie-break uses the recoded project.
    out["READPRJ_CMP"] = out["READPRJ"].where(out["READPRJ"] != 42, 37)
    out["src_order"] = np.arange(len(out))
    return out


def parse_stem(stem: str) -> tuple[str, str] | None:
    m = STEM_LR.search(stem)
    if m:
        return m.group(1).zfill(7), m.group(2).upper()
    m = STEM_NUM.search(stem)
    if m:
        return m.group(1).zfill(7), "R" if m.group(2) == "1" else "L"
    return None


def choose_row(group: pd.DataFrame, kl_folder: int, prefer_kl: bool = True) -> tuple[pd.Series, str]:
    """Prefer folder-KL agreement, then fewest missing features, then lowest READPRJ."""
    g = group
    steps: list[str] = []
    if len(g) == 1:
        return g.iloc[0], "only_one"
    if prefer_kl:
        agree = g[g["KL_oai"] == kl_folder]
        if 0 < len(agree) < len(g):
            g = agree
            steps.append("kl_agree")
    nans = g[FEATURE_COLS].isna().sum(axis=1)
    tight = g.loc[nans == nans.min()]
    if len(tight) < len(g):
        steps.append("fewest_nan")
    g = tight
    low = g.loc[g["READPRJ_CMP"] == g["READPRJ_CMP"].min()]
    if len(low) < len(g):
        steps.append("lowest_readprj")
    g = low.sort_values("src_order")
    if len(g) > 1:
        steps.append("tie_unbroken")
    if not steps:
        steps.append("identical_rows")
    return g.iloc[0], "+".join(steps)


def _self_check() -> None:
    rows = pd.DataFrame([
        {"KL_oai": 2, "READPRJ": 37, "READPRJ_CMP": 37, "src_order": 0, **{c: 0 for c in FEATURE_COLS}},
        {"KL_oai": 0, "READPRJ": 15, "READPRJ_CMP": 15, "src_order": 1, **{c: 0 for c in FEATURE_COLS}},
        {"KL_oai": 2, "READPRJ": 42, "READPRJ_CMP": 37, "src_order": 2, **{c: 1 for c in FEATURE_COLS}},
        {"KL_oai": 2, "READPRJ": 15, "READPRJ_CMP": 15, "src_order": 3, **{c: np.nan for c in FEATURE_COLS}},
    ])
    picked, rule = choose_row(rows.iloc[[0, 1]], 0)
    assert rule == "kl_agree" and picked["READPRJ"] == 15
    picked, rule = choose_row(rows.iloc[[0, 3]], 2)
    assert rule == "fewest_nan" and int(picked["src_order"]) == 0
    picked, rule = choose_row(rows.iloc[[2, 3]], 2)
    assert rule == "fewest_nan" and picked["READPRJ"] == 42
    both_full = rows.iloc[[0, 1]].copy()
    both_full["KL_oai"] = 2
    picked, rule = choose_row(both_full, 2)
    assert rule == "lowest_readprj" and picked["READPRJ"] == 15
    assert parse_number("2: 2") == 2 and parse_number(".: Missing Form") is None
    assert parse_side("1: Right") == "R" and parse_side("2: Left") == "L"
    assert parse_stem("9001695L") == ("9001695", "L")
    assert parse_stem("9003175_1") == ("9003175", "R") and parse_stem("9003175_2") == ("9003175", "L")
    assert parse_stem("gp_E1_MildG2 (1)") is None


def walk_images() -> pd.DataFrame:
    rows = []
    for dataset, root in DATASETS:
        if not root.exists():
            raise SystemExit(f"image folder not found: {root}")
        for split in SPLITS:
            for grade in range(5):
                folder = root / split / str(grade)
                if not folder.exists():
                    continue
                for path in sorted(folder.iterdir()):
                    if path.suffix.lower() not in IMG_EXTS:
                        continue
                    parsed = parse_stem(path.stem)
                    rows.append({
                        "path": str(path),
                        "rel": f"{split}/{grade}/{path.name}",
                        "filename": path.name,
                        "dataset": dataset,
                        "split": split,
                        "KL_folder": grade,
                        "ID": parsed[0] if parsed else None,
                        "SIDE": parsed[1] if parsed else None,
                    })
    return pd.DataFrame(rows)


def attach_folds(images: pd.DataFrame, folds_csv: Path) -> pd.DataFrame:
    folds = pd.read_csv(folds_csv, dtype={"path": str, "fold": int})
    fold_of = dict(zip(folds["path"], folds["fold"]))
    images = images.copy()
    images["fold"] = images["rel"].map(fold_of).astype("Int64")
    return images


def match_images(images: pd.DataFrame, oai: pd.DataFrame) -> pd.DataFrame:
    groups = {key: g for key, g in oai.groupby(["ID", "SIDE"], sort=False)}
    cache: dict[tuple, tuple] = {}
    chosen_rows = []
    for rec in images.itertuples(index=False):
        key = (rec.ID, rec.SIDE, rec.KL_folder)
        if rec.ID is None or rec.SIDE is None:
            chosen_rows.append((None, "bad_filename", 0))
            continue
        group = groups.get((rec.ID, rec.SIDE))
        if group is None:
            chosen_rows.append((None, "unmatched", 0))
            continue
        if key not in cache:
            cache[key] = (*choose_row(group, rec.KL_folder), len(group))
        row, rule, n = cache[key]
        chosen_rows.append((row, rule, n))

    base = images.drop(columns=["rel"]).reset_index(drop=True)
    extra = []
    for row, rule, n in chosen_rows:
        if row is None:
            extra.append({"KL_oai": np.nan, "readprj_used": np.nan, "n_candidate_rows": n, "dedup_rule": rule,
                          **{c: np.nan for c in FEATURE_COLS}})
        else:
            extra.append({"KL_oai": row["KL_oai"], "readprj_used": row["READPRJ"], "n_candidate_rows": n,
                          "dedup_rule": rule, **{c: row[c] for c in FEATURE_COLS}})
    out = pd.concat([base, pd.DataFrame(extra)], axis=1)
    both = out["KL_oai"].notna()
    out["kl_agree"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[both, "kl_agree"] = out.loc[both, "KL_oai"] == out.loc[both, "KL_folder"]
    cols = ["path", "dataset", "split", "fold", "KL_folder", "KL_oai", "kl_agree", *FEATURE_COLS,
            "readprj_used", "n_candidate_rows", "dedup_rule", "ID", "SIDE", "filename"]
    return out[cols]


def _rates(df: pd.DataFrame, by: list[str]) -> list[dict]:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        matched = int((~g["dedup_rule"].isin(["unmatched", "bad_filename"])).sum())
        rec = dict(zip(by, key))
        rec.update({"n": int(len(g)), "matched": matched, "match_rate": matched / len(g) if len(g) else 0.0})
        rows.append(rec)
    return rows


def _value_counts(series: pd.Series) -> dict:
    clean = series.dropna()
    counts = {str(int(k) if float(k).is_integer() else k): int(v) for k, v in clean.value_counts().sort_index().items()}
    counts["nan"] = int(series.isna().sum())
    return counts


def kl_without_preferring_folder(images: pd.DataFrame, oai: pd.DataFrame) -> dict:
    """Agreement of the row we would keep if folder KL were not allowed to break the tie.

    The saved KL_oai prefers a reading that matches the folder, so that agreement is not an
    independent check. This one is.
    """
    groups = {key: g for key, g in oai.groupby(["ID", "SIDE"], sort=False)}
    seen = set()
    blind_ok = blind_n = conflict_knees = 0
    conflict_examples = []
    for rec in images.itertuples(index=False):
        if rec.dedup_rule in ("unmatched", "bad_filename"):
            continue
        key = (rec.ID, rec.SIDE, int(rec.KL_folder))
        if key in seen:
            continue
        seen.add(key)
        group = groups[(rec.ID, rec.SIDE)]
        other = sorted({int(v) for v in group["KL_oai"].dropna() if int(v) != key[2]})
        if other:
            conflict_knees += 1
            if len(conflict_examples) < 20:
                conflict_examples.append({"ID": rec.ID, "SIDE": rec.SIDE, "KL_folder": key[2], "other_kl": other})
        picked, _ = choose_row(group, key[2], prefer_kl=False)
        blind_n += 1
        if pd.notna(picked["KL_oai"]) and int(picked["KL_oai"]) == key[2]:
            blind_ok += 1
    return {
        "unique_knees": blind_n,
        "agree_without_kl_preference": blind_ok,
        "agreement_without_kl_preference": blind_ok / blind_n if blind_n else 0.0,
        "knees_with_a_conflicting_oai_kl": conflict_knees,
        "conflict_examples": conflict_examples,
    }


def build_report(images: pd.DataFrame, oai: pd.DataFrame) -> dict:
    matched = images[~images["dedup_rule"].isin(["unmatched", "bad_filename"])].copy()
    both = matched[matched["KL_oai"].notna()]
    agree_n = int(both["kl_agree"].sum()) if len(both) else 0
    agreement = agree_n / len(both) if len(both) else 0.0

    kl_oai = both["KL_oai"].round().astype(int)
    ct = pd.crosstab(both["KL_folder"], kl_oai, dropna=False)
    columns = list(range(5)) + [c for c in ct.columns if c not in range(5)]
    ct = ct.reindex(index=range(5), columns=columns, fill_value=0)

    knees = matched.drop_duplicates(["ID", "SIDE"])
    knees = knees.copy()
    knees["ost_max"] = knees[OST_COLS].max(axis=1, skipna=True)
    dist = {}
    for grade, g in knees.groupby("KL_folder"):
        dist[str(int(grade))] = {
            "n_knees": int(len(g)),
            "ost_max": _value_counts(g["ost_max"]),
            "jsn_med": _value_counts(g["jsn_med"]),
            "jsn_lat": _value_counts(g["jsn_lat"]),
        }

    # auto_test is a second copy of the test knees (filenames <ID>_1/_2). Do not treat that copy as a different split.
    lr_bad = []
    held_in = images[(images["ID"].notna()) & (images["split"] != "auto_test")]
    for (dataset, patient), g in held_in.groupby(["dataset", "ID"]):
        if set(g["SIDE"]) < {"L", "R"}:
            continue
        splits = sorted(set(g["split"]))
        folds = sorted({int(f) for f in g["fold"].dropna().unique()})
        if len(splits) > 1 or len(folds) > 1:
            lr_bad.append({"dataset": dataset, "ID": patient, "splits": splits, "folds": folds,
                           "files": sorted(g["filename"].unique())})
    at = images[(images["split"] == "auto_test") & images["ID"].notna()]
    te = images[(images["split"] == "test") & images["ID"].notna()][["dataset", "ID", "SIDE", "KL_folder"]]
    both_copies = at.merge(te, on=["dataset", "ID", "SIDE"], suffixes=("_at", "_te"))
    grade_clash = both_copies[both_copies["KL_folder_at"] != both_copies["KL_folder_te"]]

    dup_keys = oai.groupby(["ID", "SIDE"]).size()
    dup_knees = int((dup_keys > 1).sum())
    rule_counts = matched.drop_duplicates(["ID", "SIDE", "KL_folder"])["dedup_rule"].value_counts().to_dict()

    rng = np.random.default_rng(0)
    take = matched.iloc[rng.choice(len(matched), size=min(10, len(matched)), replace=False)]
    spot = []
    for rec in take.itertuples(index=False):
        spot.append({
            "filename": rec.filename,
            "dataset": rec.dataset,
            "split": rec.split,
            "KL_folder": int(rec.KL_folder),
            "KL_oai": None if pd.isna(rec.KL_oai) else int(rec.KL_oai),
            "kl_agree": bool(rec.kl_agree) if pd.notna(rec.kl_agree) else None,
            "readprj_used": None if pd.isna(rec.readprj_used) else int(rec.readprj_used),
            "dedup_rule": rec.dedup_rule,
            "n_candidate_rows": int(rec.n_candidate_rows),
            "features": {c: (None if pd.isna(getattr(rec, c)) else float(getattr(rec, c))) for c in FEATURE_COLS},
        })

    return {
        "n_oai_rows": int(len(oai)),
        "n_oai_knees": int(dup_keys.shape[0]),
        "n_oai_knees_with_duplicate_rows": dup_knees,
        "readprj_42_rows_treated_as_37": int((oai["READPRJ"] == 42).sum()),
        "n_images": int(len(images)),
        "n_matched": int(len(matched)),
        "n_unmatched": int((images["dedup_rule"] == "unmatched").sum()),
        "n_bad_filename": int((images["dedup_rule"] == "bad_filename").sum()),
        "match_rate": len(matched) / len(images) if len(images) else 0.0,
        "match_by_split": _rates(images, ["split"]),
        "match_by_kl": _rates(images, ["KL_folder"]),
        "match_by_dataset": _rates(images, ["dataset"]),
        "unmatched_filenames": sorted(images.loc[images["dedup_rule"].isin(["unmatched", "bad_filename"]), "path"]),
        "kl_agreement_n": agree_n,
        "kl_agreement_denom": int(len(both)),
        "kl_agreement": agreement,
        "kl_check_independent_of_dedup": kl_without_preferring_folder(images, oai),
        "kl_oai_missing_among_matched": int(matched["KL_oai"].isna().sum()),
        "crosstab_columns": [int(c) for c in ct.columns],
        "crosstab_kl_folder_rows_kl_oai_cols": ct.values.tolist(),
        "missing_per_feature_matched_images": {c: int(matched[c].isna().sum()) for c in FEATURE_COLS},
        "feature_distribution_unique_knees": dist,
        "n_bad_filename_gp": int(images.loc[images["dedup_rule"] == "bad_filename", "filename"].str.startswith("gp_").sum()),
        "lr_split_or_fold_conflicts": lr_bad,
        "auto_test_same_knee_as_test": int(len(both_copies)),
        "auto_test_images": int(len(at)),
        "auto_test_folder_kl_disagrees_with_test": int(grade_clash.drop_duplicates(["ID", "SIDE"]).shape[0]),
        "dedup_rule_counts_unique_knee_and_folder_kl": {str(k): int(v) for k, v in rule_counts.items()},
        "spot_check": spot,
        "note": "Test and auto_test are labeled here only. They are not used to choose a model.",
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# OAI feature-label match",
        "",
        "Source: `KXR_SQ_BU00.txt` (baseline, Boston University reading). "
        "Column names confirmed against `kxr_sq_bu00_Contents.pdf`. "
        "`READPRJ` 42 is treated as project 37 when breaking ties (`kXR_SQ_BU_Descrip.pdf`).",
        "",
        f"- Images: {report['n_images']}",
        f"- Matched: {report['n_matched']} ({report['match_rate']:.1%})",
        f"- Unmatched: {report['n_unmatched']}",
        f"- Filename has no OAI id: {report['n_bad_filename']}",
        f"- KL agreement of the saved row (KL_oai == folder KL): {report['kl_agreement']:.1%} "
        f"({report['kl_agreement_n']} / {report['kl_agreement_denom']})",
        f"- KL agreement if a matching grade is not allowed to win the tie: "
        f"{report['kl_check_independent_of_dedup']['agreement_without_kl_preference']:.1%} "
        f"({report['kl_check_independent_of_dedup']['agree_without_kl_preference']} / "
        f"{report['kl_check_independent_of_dedup']['unique_knees']} knees)",
        f"- Knees that also have an OAI reading with a different KL: "
        f"{report['kl_check_independent_of_dedup']['knees_with_a_conflicting_oai_kl']}",
        f"- OAI rows: {report['n_oai_rows']}; knees with more than one row: "
        f"{report['n_oai_knees_with_duplicate_rows']}",
        f"- Rows coded READPRJ 42 (compared as 37): {report['readprj_42_rows_treated_as_37']}",
        "",
        "## Match rate",
        "",
        "| Group | n | matched | rate |",
        "|---|---:|---:|---:|",
    ]
    for block in ("match_by_dataset", "match_by_split", "match_by_kl"):
        for rec in report[block]:
            label = ", ".join(f"{k}={rec[k]}" for k in rec if k not in ("n", "matched", "match_rate"))
            lines.append(f"| {label} | {rec['n']} | {rec['matched']} | {rec['match_rate']:.1%} |")
    cols = report["crosstab_columns"]
    lines += [
        "",
        "## KL agreement",
        "",
        "The saved `KL_oai` keeps an OAI reading that matches the folder grade when one exists, so the crosstab can look perfect. "
        "The independent check above ignores that preference. "
        "`auto_test` files are named `<ID>_1` (right) and `<ID>_2` (left), the same side codes as the OAI file. "
        "Osteophyte, sclerosis, cyst, and attrition grades are often missing when the OAI KL is 0 or 1; the data dictionary says those features were not read on knees without definite osteoarthritis. Joint-space grades are filled in.",
        "",
        "| KL_folder | " + " | ".join(str(c) for c in cols) + " |",
        "|---|" + "|".join("---:" for _ in cols) + "|",
    ]
    for grade, row in enumerate(report["crosstab_kl_folder_rows_kl_oai_cols"]):
        lines.append(f"| {grade} | " + " | ".join(str(x) for x in row) + " |")
    conflicts = report["kl_check_independent_of_dedup"]["conflict_examples"]
    if conflicts:
        lines += ["", "Knees whose OAI file also contains a different KL (up to 20):", ""]
        for rec in conflicts:
            lines.append(f"- {rec['ID']}{rec['SIDE']}: folder {rec['KL_folder']}, other OAI grades {rec['other_kl']}")
    lines += ["", "## Missing features among matched images", "", "| Feature | missing |", "|---|---:|"]
    for name, n in report["missing_per_feature_matched_images"].items():
        lines.append(f"| {name} | {n} |")
    lines += ["", "## Feature distribution, one row per knee (ID, SIDE)", ""]
    for grade, block in report["feature_distribution_unique_knees"].items():
        lines.append(f"### KL {grade} (n={block['n_knees']})")
        lines.append("")
        for name in ("ost_max", "jsn_med", "jsn_lat"):
            lines.append(f"- {name}: `{json.dumps(block[name])}`")
        lines.append("")
    lines += ["## Same patient, left and right", ""]
    conflicts = report["lr_split_or_fold_conflicts"]
    if conflicts:
        lines.append(f"{len(conflicts)} patient/dataset pairs have L and R in different splits or folds, ignoring the auto_test copy.")
        lines.append("")
        for rec in conflicts[:20]:
            lines.append(f"- {rec['dataset']} {rec['ID']}: splits {rec['splits']}, folds {rec['folds']}")
    else:
        lines.append("Ignoring the auto_test copy: every patient with both knees keeps L and R in the same split, and in the same Pipeline fold when a fold is assigned. Train/val fold conflicts: 0.")
    lines += [
        "",
        f"`auto_test` is the same knees as `test` ({report['auto_test_same_knee_as_test']} of {report['auto_test_images']} images, per dataset copy included). "
        f"The folder grade differs between those two copies for {report['auto_test_folder_kl_disagrees_with_test']} knees. "
        f"The {report['n_bad_filename']} files with no OAI id are the `gp_` images ({report['n_bad_filename_gp']}).",
        "",
    ]
    lines += ["", "## Which row was kept when a knee had several OAI readings", "",
              "| Rule | knees |", "|---|---:|"]
    for rule, n in sorted(report["dedup_rule_counts_unique_knee_and_folder_kl"].items()):
        lines.append(f"| {rule} | {n} |")
    lines += ["", "## Spot check (seed 0)", ""]
    for rec in report["spot_check"]:
        feats = ", ".join(f"{k}={v}" for k, v in rec["features"].items() if v not in (None, 0))
        lines.append(
            f"- `{rec['filename']}` ({rec['dataset']}/{rec['split']}) folder KL {rec['KL_folder']}, "
            f"OAI KL {rec['KL_oai']}, readprj {rec['readprj_used']}, rule {rec['dedup_rule']}, "
            f"candidates {rec['n_candidate_rows']}"
            + (f", nonzero features: {feats}" if feats else ", all listed features 0 or missing")
        )
    lines += ["", "## Unmatched files", ""]
    names = report["unmatched_filenames"]
    lines.append(f"{len(names)} files. Full list is in `match_report.json`.")
    lines.append("")
    for name in names[:50]:
        lines.append(f"- `{name}`")
    if len(names) > 50:
        lines.append(f"- … {len(names) - 50} more in the json")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _self_check()
    ap = argparse.ArgumentParser(description="Match OAI KXR_SQ_BU00 features onto knee images.")
    ap.add_argument("--oai-txt", type=Path, default=PROJECT_ROOT / "data" / "oai_labels" / "KXR_SQ_BU00.txt")
    ap.add_argument("--folds-csv", type=Path, default=PROJECT_ROOT / "reports" / "pipeline" / "folds.csv")
    ap.add_argument("--out-csv", type=Path, default=PROJECT_ROOT / "data" / "oai_labels" / "oai_features_matched.csv")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "oai_match")
    args = ap.parse_args()

    oai = load_oai(args.oai_txt)
    if len(oai) != 16666:
        raise SystemExit(f"expected 16666 OAI rows, parsed {len(oai)}")
    images = attach_folds(walk_images(), args.folds_csv)
    matched = match_images(images, oai)
    if len(matched) != len(images):
        raise SystemExit("row count changed during the match; refusing to write a partial file")

    for col in ["KL_oai", "readprj_used", *FEATURE_COLS]:
        values = matched[col].dropna()
        if len(values) and (values % 1 == 0).all():
            matched[col] = matched[col].astype("Int64")
    report = build_report(matched, oai)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_cols = ["path", "dataset", "split", "fold", "KL_folder", "KL_oai", "kl_agree", *FEATURE_COLS,
                 "readprj_used", "n_candidate_rows", "dedup_rule"]
    matched[save_cols].to_csv(args.out_csv, index=False)
    (args.out_dir / "match_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, args.out_dir / "match_report.md")

    print(f"images {report['n_images']}")
    print(f"matched {report['n_matched']} ({report['match_rate']:.1%})")
    independent = report["kl_check_independent_of_dedup"]["agreement_without_kl_preference"]
    print(f"KL agreement {report['kl_agreement']:.1%} ({report['kl_agreement_n']} / {report['kl_agreement_denom']})")
    print(f"KL agreement without preferring a matching grade {independent:.1%}")
    print(f"csv {args.out_csv}")
    print(f"report {args.out_dir / 'match_report.md'}")
    for rec in report["spot_check"]:
        print(f"  spot {rec['filename']} folder={rec['KL_folder']} oai={rec['KL_oai']} rule={rec['dedup_rule']}")
    if report["kl_agreement"] < AGREEMENT_STOP or independent < AGREEMENT_STOP:
        print(f"STOP: KL agreement {report['kl_agreement']:.1%} is below {AGREEMENT_STOP:.0%}. Files were written; no rows were dropped.")
        sys.exit(2)


if __name__ == "__main__":
    main()
