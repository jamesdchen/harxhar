"""
Submit model horse race: all models with baseline features (HAR target lags only).

Paper result: Table comparing model architectures on forecasting ability.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.cli.backends import get_backend
from src.cli.submit import (
    ExperimentSpec,
    add_common_submit_args,
    build_extra_args,
    submit_experiment_batch,
)
from src.feature_groups import ALL_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Submit model comparison experiments (all models, baseline features).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_submit_args(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help=f"Models to compare. Default: {ALL_MODELS}.",
    )
    parser.set_defaults(result_dir="results/model_comparison")
    args = parser.parse_args()

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

    backend = get_backend(args.backend)
    submit_experiment_batch(
        specs=specs,
        base_dir=args.result_dir,
        total_chunks=args.total_chunks,
        include_naive=not args.no_naive,
        backend=backend,
    )


if __name__ == "__main__":
    main()
