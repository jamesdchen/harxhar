"""
Unified experiment submission script.

Consolidates all submission modes into one entry point:

    python scripts/submit.py model_comparison [--models ridge xgboost ...]
    python scripts/submit.py feature_transforms [--features har pca ae] [--subgroup all_features]
    python scripts/submit.py individual_features [--subgroup moments]
    python scripts/submit.py subgroup_analysis [--models all --features all --subgroups all]
    python scripts/submit.py naive
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.cli.backends import get_backend
from src.cli.experiment_config import load_experiment_config
from src.cli.submit import (
    ExperimentSpec,
    add_common_submit_args,
    build_extra_args,
    submit_experiment_batch,
)
from src.core.config import DEFAULT_RESULTS_DIR
from src.features.feature_groups import ALL_MODELS, FEATURE_TYPES, SUBGROUPS

# ---------------------------------------------------------------------------
# Mode: model_comparison
# ---------------------------------------------------------------------------


def _add_model_comparison_args(sub):
    sub.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help=f"Models to compare. Default: {ALL_MODELS}.",
    )
    sub.set_defaults(result_dir="results/model_comparison")


def _run_model_comparison(args):
    feature_type = "har"
    extra_args = build_extra_args(feature_type, args)

    specs = [
        ExperimentSpec(
            exp_id=i + 1,
            exp_name="baseline",
            model_type=model,
            feature_type=feature_type,
            variables=[],
            extra_args=extra_args,
        )
        for i, model in enumerate(args.models)
    ]
    return specs


# ---------------------------------------------------------------------------
# Mode: feature_transforms
# ---------------------------------------------------------------------------


def _add_feature_transforms_args(sub):
    sub.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model to use for the comparison.",
    )
    sub.add_argument(
        "--subgroup",
        type=str,
        default="all_features",
        help=f"Feature subgroup to use. Choices: {list(SUBGROUPS.keys())}.",
    )
    sub.add_argument(
        "--features",
        nargs="+",
        default=FEATURE_TYPES,
        help=f"Feature types to compare. Default: {FEATURE_TYPES}.",
    )
    sub.set_defaults(result_dir="results/feature_transforms")


def _run_feature_transforms(args):
    variables = SUBGROUPS[args.subgroup]

    specs = []
    for i, feature_type in enumerate(args.features):
        extra_args = build_extra_args(feature_type, args)
        specs.append(
            ExperimentSpec(
                exp_id=i + 1,
                exp_name=f"{args.subgroup}_{feature_type}",
                model_type=args.model,
                feature_type=feature_type,
                variables=variables,
                extra_args=extra_args,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Mode: individual_features
# ---------------------------------------------------------------------------


def _add_individual_features_args(sub):
    sub.add_argument(
        "--model",
        type=str,
        default="ridge",
        help="Model to use.",
    )
    sub.add_argument(
        "--subgroup",
        type=str,
        default="moments",
        help=f"Subgroup whose features to test individually. Choices: {list(SUBGROUPS.keys())}.",
    )


def _run_individual_features(args):
    if args.result_dir == DEFAULT_RESULTS_DIR:
        args.result_dir = f"results/individual_{args.subgroup}"

    features = SUBGROUPS[args.subgroup]
    if not features:
        print(f"Subgroup '{args.subgroup}' has no features to test individually.")
        return []

    feature_type = "har"
    extra_args = build_extra_args(feature_type, args)

    return [
        ExperimentSpec(
            exp_id=i + 1,
            exp_name=f"{args.subgroup}_{feature}",
            model_type=args.model,
            feature_type=feature_type,
            variables=[feature],
            extra_args=extra_args,
        )
        for i, feature in enumerate(features)
    ]


# ---------------------------------------------------------------------------
# Mode: subgroup_analysis
# ---------------------------------------------------------------------------


def _resolve_list(arg, full_list):
    if len(arg) == 1 and arg[0] == "all":
        return full_list
    return arg


def _resolve_subgroups(arg):
    if len(arg) == 1 and arg[0] == "all":
        return SUBGROUPS
    return {k: SUBGROUPS[k] for k in arg if k in SUBGROUPS}


def _add_subgroup_analysis_args(sub):
    sub.add_argument(
        "--models",
        nargs="+",
        default=["ridge"],
        help=f"Models to run. Use 'all' for: {ALL_MODELS}.",
    )
    sub.add_argument(
        "--features",
        nargs="+",
        default=["har"],
        help=f"Feature types to run. Use 'all' for: {FEATURE_TYPES}.",
    )
    sub.add_argument(
        "--subgroups",
        nargs="+",
        default=["all"],
        help=f"Subgroups to run. Use 'all' for: {list(SUBGROUPS.keys())}.",
    )
    sub.set_defaults(result_dir="results/subgroup_analysis")


def _run_subgroup_analysis(args):
    models_to_run = _resolve_list(args.models, ALL_MODELS)
    features_to_run = _resolve_list(args.features, FEATURE_TYPES)
    subgroups_to_run = _resolve_subgroups(args.subgroups)

    print(
        f"Generating experiments for {len(subgroups_to_run)} subgroups "
        f"x {len(models_to_run)} models x {len(features_to_run)} feature types"
        + (" + Naive baseline" if not args.no_naive else "")
        + "..."
    )

    specs = []
    exp_id = 1
    for feature_type in features_to_run:
        extra_args = build_extra_args(feature_type, args)
        for model_type in models_to_run:
            for exp_name, variables in subgroups_to_run.items():
                specs.append(
                    ExperimentSpec(
                        exp_id=exp_id,
                        exp_name=exp_name,
                        model_type=model_type,
                        feature_type=feature_type,
                        variables=variables,
                        extra_args=extra_args,
                    )
                )
                exp_id += 1
    return specs


# ---------------------------------------------------------------------------
# Mode: naive
# ---------------------------------------------------------------------------


def _add_naive_args(sub):
    sub.set_defaults(result_dir="results/naive")


def _run_naive(_args):
    return []


# ---------------------------------------------------------------------------
# Mode: from-config (YAML experiment config)
# ---------------------------------------------------------------------------


def _add_from_config_args(sub):
    sub.add_argument("config_file", type=str, help="Path to YAML/JSON experiment config file.")


def _run_from_config(args):
    """Load experiment specs from a YAML config file."""
    cfg = load_experiment_config(args.config_file)

    # Override args from config
    if cfg.result_dir:
        args.result_dir = cfg.result_dir
    else:
        args.result_dir = f"results/{cfg.name}"
    args.total_chunks = cfg.total_chunks
    args.backend = cfg.backend
    args.no_naive = cfg.no_naive
    args.train_window = cfg.train_window
    args.n_components = cfg.n_components
    args.ae_alpha = cfg.ae_alpha
    args.ae_epochs = cfg.ae_epochs
    args.ae_hidden = cfg.ae_hidden
    args.ae_weights_path = cfg.ae_weights_path
    args.horizon = cfg.horizon

    # Delegate to the appropriate mode runner
    inner_mode = cfg.mode
    if inner_mode == "model_comparison":
        args.models = cfg.models
        return _run_model_comparison(args)
    elif inner_mode == "feature_transforms":
        args.model = cfg.models[0] if cfg.models else "ridge"
        args.features = cfg.features
        args.subgroup = cfg.subgroups[0] if cfg.subgroups else "all_features"
        return _run_feature_transforms(args)
    elif inner_mode == "individual_features":
        args.model = cfg.models[0] if cfg.models else "ridge"
        args.subgroup = cfg.subgroups[0] if cfg.subgroups else "moments"
        return _run_individual_features(args)
    elif inner_mode == "subgroup_analysis":
        args.models = cfg.models
        args.features = cfg.features
        args.subgroups = cfg.subgroups
        return _run_subgroup_analysis(args)
    elif inner_mode == "naive":
        return _run_naive(args)
    else:
        print(f"Unknown mode '{inner_mode}' in config file.")
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES = {
    "model_comparison": (_add_model_comparison_args, _run_model_comparison),
    "feature_transforms": (_add_feature_transforms_args, _run_feature_transforms),
    "individual_features": (_add_individual_features_args, _run_individual_features),
    "subgroup_analysis": (_add_subgroup_analysis_args, _run_subgroup_analysis),
    "naive": (_add_naive_args, _run_naive),
    "from-config": (_add_from_config_args, _run_from_config),
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Submission mode.")

    for mode_name, (add_args_fn, _) in MODES.items():
        sub = subparsers.add_parser(
            mode_name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        # from-config doesn't need common submit args (they come from the config file)
        if mode_name != "from-config":
            add_common_submit_args(sub)
        add_args_fn(sub)

    args = parser.parse_args()
    _, run_fn = MODES[args.mode]

    specs = run_fn(args)

    # For naive mode, force include_naive=True
    include_naive = not getattr(args, "no_naive", False) if args.mode != "naive" else True

    backend = get_backend(getattr(args, "backend", "slurm"))
    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=getattr(args, "total_chunks", 100),
        include_naive=include_naive,
        backend=backend,
    )


if __name__ == "__main__":
    main()
