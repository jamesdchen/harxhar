"""
Submit feature transform comparison: raw vs HAR vs PCA vs AE.

Paper result: Table comparing feature transformation methods.
"""
import argparse
from src.feature_groups import FEATURE_TYPES, SUBGROUPS
from src.submit import (
    ExperimentSpec, add_common_submit_args, build_extra_args,
    submit_experiment_batch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Submit feature transform comparison experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.add_argument(
        "--model", type=str, default="ridge",
        help="Model to use for the comparison.",
    )
    parser.add_argument(
        "--subgroup", type=str, default="all_features",
        help=f"Feature subgroup to use. Choices: {list(SUBGROUPS.keys())}.",
    )
    parser.add_argument(
        "--features", nargs="+", default=FEATURE_TYPES,
        help=f"Feature types to compare. Default: {FEATURE_TYPES}.",
    )
    parser.set_defaults(result_dir="results_feature_transforms")
    args = parser.parse_args()

    variables = SUBGROUPS[args.subgroup]

    specs = []
    for i, feature_type in enumerate(args.features):
        extra_args = build_extra_args(feature_type, args)
        specs.append(ExperimentSpec(
            exp_id=i + 1,
            exp_name=f"{args.subgroup}_{feature_type}",
            model_type=args.model,
            feature_type=feature_type,
            variables=variables,
            extra_args=extra_args,
        ))

    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=not args.no_naive,
    )


if __name__ == "__main__":
    main()
