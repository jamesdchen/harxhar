"""
Submit full subgroup analysis: models × feature types × subgroups.

Paper result: Table showing which feature subgroups improve forecasts.
This is the "run everything" script — use the focused scripts for targeted runs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.feature_groups import ALL_MODELS, FEATURE_TYPES, SUBGROUPS
from src.submit import (
    ExperimentSpec, add_common_submit_args, build_extra_args,
    submit_experiment_batch,
)


def resolve_list(arg, full_list):
    if len(arg) == 1 and arg[0] == "all":
        return full_list
    return arg


def resolve_subgroups(arg):
    if len(arg) == 1 and arg[0] == "all":
        return SUBGROUPS
    return {k: SUBGROUPS[k] for k in arg if k in SUBGROUPS}


def main():
    parser = argparse.ArgumentParser(
        description="Submit subgroup analysis experiments to Slurm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        help=f"Models to run. Use 'all' for: {ALL_MODELS}.",
    )
    parser.add_argument(
        "--features", nargs="+", default=["raw"],
        help=f"Feature types to run. Use 'all' for: {FEATURE_TYPES}.",
    )
    parser.add_argument(
        "--subgroups", nargs="+", default=["all"],
        help=f"Subgroups to run. Use 'all' for: {list(SUBGROUPS.keys())}.",
    )
    parser.set_defaults(result_dir="results_subgroup_analysis")
    args = parser.parse_args()

    models_to_run = resolve_list(args.models, ALL_MODELS)
    features_to_run = resolve_list(args.features, FEATURE_TYPES)
    subgroups_to_run = resolve_subgroups(args.subgroups)

    total = len(subgroups_to_run) * len(models_to_run) * len(features_to_run)
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
                specs.append(ExperimentSpec(
                    exp_id=exp_id,
                    exp_name=exp_name,
                    model_type=model_type,
                    feature_type=feature_type,
                    variables=variables,
                    extra_args=extra_args,
                ))
                exp_id += 1

    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=not args.no_naive,
    )


if __name__ == "__main__":
    main()
