from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineOptions, run_pipeline
from .paths import DEFAULT_FACE_TEMPLATE_PATH, DEFAULT_SKULL_TEMPLATE_PATH


STAGES = (
    "standardize",
    "render",
    "fit3dmm",
    "pose",
    "texture",
    "face_nicp",
    "skull_controls",
    "skull_nicp",
)


def build_parser():
    parser = argparse.ArgumentParser(prog="preprocess_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one or more preprocessing stages")
    run.add_argument("--data-root", required=True)
    run.add_argument("--output-root", help="write each subject to OUTPUT_ROOT/SUBJECT instead of INPUT/SUBJECT/output")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--deterministic", action="store_true", help="fixed-order CUDA backward and strict reproducibility controls")
    run.add_argument("--seed", type=int, default=0, help="seed used by --deterministic (default: 0)")
    run.add_argument("--stage", action="append", choices=(*STAGES, "all"), default=None)
    run.add_argument("--include", action="append", default=[])
    run.add_argument("--skull-template-dir")
    run.add_argument("--nicp-config")
    run.add_argument("--first-rf-iters", type=int, default=1000)
    run.add_argument("--first-nrf-iters", type=int, default=500)
    run.add_argument("--rf-lr", type=float, default=1e-2)
    run.add_argument("--nrf-lr", type=float, default=1e-2)

    tmpl = sub.add_parser("make-template", help="export skull template assets for --skull-template-dir")
    tmpl.add_argument("--output-dir", required=True)
    tmpl.add_argument("--face-template", default=str(DEFAULT_FACE_TEMPLATE_PATH))
    tmpl.add_argument("--skull-template", default=str(DEFAULT_SKULL_TEMPLATE_PATH))

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "make-template":
        from .topology import export_template_controls

        export_template_controls(
            Path(args.output_dir),
            face_template=args.face_template,
            skull_template=args.skull_template,
        )
        return
    stages = tuple(args.stage) if args.stage else None
    fit = None
    stage_probe = set(stages or ())
    if stages is None or "all" in stage_probe or "fit3dmm" in stage_probe:
        from .bfm_fit import FitOptions

        fit = FitOptions(
            device=args.device,
            first_rf_iters=args.first_rf_iters,
            first_nrf_iters=args.first_nrf_iters,
            rf_lr=args.rf_lr,
            nrf_lr=args.nrf_lr,
        )
    options = PipelineOptions(
        data_root=args.data_root,
        device=args.device,
        include=tuple(args.include),
        skull_template_dir=args.skull_template_dir,
        nicp_config=args.nicp_config,
        fit=fit,
        deterministic=args.deterministic,
        seed=args.seed,
        output_root=args.output_root,
    )
    run_pipeline(options, stages=stages)


if __name__ == "__main__":
    main()
