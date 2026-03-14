"""
Submit per-feature marginal value experiments within a subgroup.

Paper result: Appendix table showing each feature's individual contribution.
Replaces the old submit_moments.py.
"""
import argparse
from src.feature_groups import SUBGROUPS
from src.submit import (
    ExperimentSpec, add_common_submit_args, build_extra_args,
    submit_experiment_batch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Submit individual feature experiments within a subgroup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.add_argument(
        "--model", type=str, default="ridge",
        help="Model to use.",
    )
    parser.add_argument(
        "--subgroup", type=str, default="moments",
        help=f"Subgroup whose features to test individually. Choices: {list(SUBGROUPS.keys())}.",
    )
    parser.set_defaults(result_dir=None)
    args = parser.parse_args()

    if args.result_dir is None:
        args.result_dir = f"results_individual_{args.subgroup}"

    features = SUBGROUPS[args.subgroup]
    if not features:
        print(f"Subgroup '{args.subgroup}' has no features to test individually.")
        return

    feature_type = "raw"
    extra_args = build_extra_args(feature_type, args)

    specs = [
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

    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=not args.no_naive,
    )


if __name__ == "__main__":
    main()
