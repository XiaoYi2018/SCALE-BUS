"""Derive logit shifts and propagated/observed variance ratios from saved results.

Inputs are read only. Outputs are created in a new directory. The uncertainty
CSV must contain means and standard uncertainties of the logit draws; taking a
logit of a mean probability is a different calculation and is not used here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from metrology_paths import OUTPUT_ROOT


PROBABILITY_COLUMN = "probability_advanced76_ensemble"
CLIP = 1e-6
PRIMARY_CONTOUR = 0.025
CONTOUR_LEVELS = (0.01, 0.025, 0.05)
GROUPS = ("all", "0", "1")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(OUTPUT_ROOT)
    parser.add_argument("--score-uncertainty", type=Path, default=root / "v07_uncertainty_propagation" / "per_image_score_uncertainty.csv")
    parser.add_argument("--busbra-predictions", type=Path, default=root / "v07_positive_control" / "busbra_predictions.csv")
    parser.add_argument("--breast-predictions", type=Path, default=root / "v07_positive_control" / "breast_predictions.csv")
    parser.add_argument("--lesion-pairs", type=Path, default=root / "v07_interview_reproducibility" / "included_lesion_pairs.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; an existing directory is rejected.")
    return parser.parse_args()


def require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def numeric(frame, columns, name):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: nonfinite values in {column}")


def targets(frame, name):
    numeric(frame, ["target"], name)
    if not frame["target"].isin([0, 1]).all():
        raise ValueError(f"{name}: target must be 0 or 1")
    frame["target"] = frame["target"].astype(int)


def read_predictions(path, dataset):
    frame = pd.read_csv(path, dtype={"image": str})
    require_columns(frame, ["image", "target", PROBABILITY_COLUMN], dataset)
    targets(frame, dataset)
    numeric(frame, [PROBABILITY_COLUMN], dataset)
    if frame["image"].isna().any() or frame["image"].duplicated().any():
        raise ValueError(f"{dataset}: image identifiers must be present and unique")
    if not frame[PROBABILITY_COLUMN].between(0, 1).all():
        raise ValueError(f"{dataset}: probabilities outside [0, 1]")
    result = frame[["image", "target"]].copy()
    result["dataset"] = dataset
    result["clean_logit"] = logit(np.clip(frame[PROBABILITY_COLUMN].to_numpy(), CLIP, 1 - CLIP))
    return result


def read_pairs(path, clean):
    frame = pd.read_csv(path, dtype={"first_image": str, "second_image": str})
    require_columns(frame, ["first_image", "second_image", "target"], "lesion pairs")
    targets(frame, "lesion pairs")
    images = pd.concat([frame["first_image"], frame["second_image"]], ignore_index=True)
    if images.isna().any() or images.duplicated().any():
        raise ValueError("Lesion pairs must have distinct, nonmissing images, each used once")
    result = frame[["first_image", "second_image", "target"]].copy()
    lookup = clean.loc[clean["dataset"] == "busbra"].set_index("image")
    for side in ("first", "second"):
        result[f"{side}_logit"] = result[f"{side}_image"].map(lookup["clean_logit"])
        found_target = result[f"{side}_image"].map(lookup["target"])
        wrong_target = found_target.notna() & (found_target != result["target"])
        if wrong_target.any():
            raise ValueError(f"Lesion-pair target disagrees with {side}-view prediction")
    matched = result.dropna(subset=["first_logit", "second_logit"]).copy()
    if matched.empty:
        raise ValueError("No complete lesion pair matches clean BUS-BRA predictions")
    matched["squared_difference_over_two"] = (matched["first_logit"] - matched["second_logit"]) ** 2 / 2
    return matched, len(frame)


def read_uncertainty(path, clean):
    frame = pd.read_csv(path, dtype={"dataset": str, "image": str})
    require_columns(frame, ["dataset", "image", "target", "source", "contour_sigma", "output", "u", "mean"], "score uncertainty")
    frame = frame.loc[frame["output"] == "logit"].copy()
    if frame.empty:
        raise ValueError("No logit rows in score uncertainty")
    targets(frame, "score uncertainty")
    numeric(frame, ["contour_sigma", "u", "mean"], "score uncertainty")
    if (frame["u"] < 0).any():
        raise ValueError("Standard uncertainties must be nonnegative")
    if frame[["dataset", "image", "source"]].isna().any().any():
        raise ValueError("Missing uncertainty identifiers")
    keys = ["dataset", "image", "source", "contour_sigma"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate logit uncertainty rows")
    if not frame["dataset"].isin(["busbra", "breast"]).all():
        raise ValueError("Expected dataset identifiers busbra and breast")
    result = frame.merge(clean.rename(columns={"target": "clean_target"}), on=["dataset", "image"], how="left", validate="many_to_one")
    if result["clean_logit"].isna().any():
        raise ValueError("Uncertainty images lack clean ensemble predictions")
    if (result["target"] != result["clean_target"]).any():
        raise ValueError("Uncertainty targets disagree with clean predictions")
    result["logit_shift"] = result["mean"] - result["clean_logit"]
    result["variance"] = result["u"] ** 2
    return result


def subgroup(frame, target):
    return frame if target == "all" else frame.loc[frame["target"] == int(target)]


def source_label(source):
    return "gain" if source == "gamma" else source


def shift_summary(uncertainty):
    rows = []
    for (dataset, source, level), frame in uncertainty.groupby(["dataset", "source", "contour_sigma"], sort=True):
        for target in GROUPS:
            group = subgroup(frame, target)
            if group.empty:
                continue
            rows.append({"dataset": dataset, "source": source, "source_label": source_label(source), "contour_sigma": float(level), "target": target, "n_images": len(group), "median_logit_shift": float(group["logit_shift"].median())})
    return pd.DataFrame(rows)


def observed_summary(pairs, included_pairs):
    rows = []
    for target in GROUPS:
        group = subgroup(pairs, target)
        if group.empty:
            continue
        rows.append({"dataset": "busbra", "target": target, "included_pairs_total": included_pairs, "n_pairs": len(group), "n_images": 2 * len(group), "observed_within_lesion_logit_variance": float(group["squared_difference_over_two"].mean())})
    return pd.DataFrame(rows)


def variance_summary(uncertainty, pairs, included_pairs):
    rows = []
    specs = [(source, level, level) for level in CONTOUR_LEVELS for source in ("joint", "contour")]
    specs += [(source, 0.0, PRIMARY_CONTOUR) for source in ("speckle", "resolution", "gamma")]
    busbra = uncertainty.loc[uncertainty["dataset"] == "busbra"]
    for source, input_level, comparison_level in specs:
        available = busbra.loc[(busbra["source"] == source) & np.isclose(busbra["contour_sigma"], input_level, rtol=0, atol=1e-12)]
        image_variance = available.set_index("image")["variance"]
        mask = pairs["first_image"].isin(image_variance.index) & pairs["second_image"].isin(image_variance.index)
        complete = pairs.loc[mask].copy()
        complete["first_variance"] = complete["first_image"].map(image_variance)
        complete["second_variance"] = complete["second_image"].map(image_variance)
        for target in GROUPS:
            eligible = subgroup(pairs, target)
            found = subgroup(available, target)
            group = subgroup(complete, target)
            eligible_images = set(eligible["first_image"]) | set(eligible["second_image"])
            available_pair_images = int(found["image"].isin(eligible_images).sum())
            observed = float(group["squared_difference_over_two"].mean()) if len(group) else None
            propagated = float(group[["first_variance", "second_variance"]].to_numpy().mean()) if len(group) else None
            ratio = propagated / observed if observed is not None and observed > 0 else None
            rows.append({"dataset": "busbra", "source": source, "source_label": source_label(source), "contour_sigma": comparison_level, "source_contour_sigma": input_level, "target": target, "included_pairs_total": included_pairs, "clean_matched_pairs": len(eligible), "available_uncertainty_images": len(found), "available_pair_images": available_pair_images, "matched_pairs": len(group), "matched_images": 2 * len(group), "unpaired_available_images": available_pair_images - 2 * len(group), "mean_squared_standard_uncertainty": propagated, "observed_within_lesion_logit_variance": observed, "propagated_observed_variance_ratio": ratio, "propagated_observed_variance_percent": 100 * ratio if ratio is not None else None})
    return pd.DataFrame(rows)


def records(frame):
    # pandas serializes missing numeric results as JSON null rather than NaN.
    return json.loads(frame.to_json(orient="records", double_precision=15))


def main():
    args = arguments()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    clean = pd.concat([read_predictions(args.busbra_predictions, "busbra"), read_predictions(args.breast_predictions, "breast")], ignore_index=True)
    pairs, included_pairs = read_pairs(args.lesion_pairs, clean)
    uncertainty = read_uncertainty(args.score_uncertainty, clean)
    shifts = shift_summary(uncertainty)
    observed = observed_summary(pairs, included_pairs)
    comparison = variance_summary(uncertainty, pairs, included_pairs)
    summary = {
        "definitions": {
            "clean_logit": "logit(clip(clean probability_advanced76_ensemble, 1e-6, 1-1e-6))",
            "logit_shift": "saved mean of perturbed logits minus clean logit; median across images within dataset/source/contour_sigma/target",
            "observed_variance": "mean((first clean logit - second clean logit)**2) / 2 across complete included lesion pairs",
            "propagated_variance": "mean(saved logit standard uncertainty u**2) over the two images of the same complete pairs",
            "variance_ratio": "propagated variance / observed variance, recomputed on matching pairs for each row",
            "single_source_levels": "speckle, resolution and gamma use source_contour_sigma=0 and are compared at the primary contour_sigma=0.025",
            "gamma": "gamma is the gain perturbation",
            "target": "0 benign; 1 malignant; all combines both classes",
            "missing_pairs": "Only complete pairs enter variance comparisons; missing pairs and unmatched single views are reported in the CSV counts. No missing observations are imputed.",
        },
        "input_counts": {"busbra_clean_images": int((clean["dataset"] == "busbra").sum()), "breast_clean_images": int((clean["dataset"] == "breast").sum()), "included_pairs": included_pairs, "clean_matched_pairs": len(pairs), "logit_uncertainty_rows": len(uncertainty)},
        "logit_shifts": records(shifts),
        "observed_interview_variance": records(observed),
        "propagated_observed_variance": records(comparison),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [("median_logit_shifts.csv", shifts), ("observed_interview_variance.csv", observed), ("propagated_observed_variance.csv", comparison)]:
        with (output / name).open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
    with (output / "derived_quantities.json").open("x", encoding="utf-8", newline="") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Median logit shifts (target 0=benign, 1=malignant):")
    print(shifts.to_string(index=False))
    print("\nObserved within-lesion logit variance:")
    print(observed.to_string(index=False))
    print("\nPropagated variance relative to observed variance on matching pairs:")
    print(comparison.to_string(index=False))
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
