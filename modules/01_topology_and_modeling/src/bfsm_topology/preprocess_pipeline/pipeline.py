from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import SamplePaths, iter_samples, check_registration_output


@dataclass
class PipelineOptions:
    data_root: str | Path
    device: str = "cuda:0"
    include: tuple[str, ...] = ()
    skull_template_dir: str | Path | None = None
    nicp_config: str | Path | None = None
    fit: object | None = None
    deterministic: bool = False
    seed: int = 0
    output_root: str | Path | None = None
    output_dir: str | Path | None = None


DEFAULT_STAGES = (
    "standardize",
    "render",
    "fit3dmm",
    "pose",
    "texture",
    "face_nicp",
    "skull_controls",
    "skull_nicp",
)


def run_pipeline(options: PipelineOptions, stages: Iterable[str] | None = None) -> None:
    requested = tuple(stages or DEFAULT_STAGES)
    if "all" in requested:
        requested = DEFAULT_STAGES
    stage_set = set(requested)
    unknown = stage_set - set(DEFAULT_STAGES)
    if unknown:
        raise ValueError("unknown stages: " + ", ".join(sorted(unknown)))
    samples = iter_samples(options.data_root, include=options.include)
    if options.output_root is not None and options.output_dir is not None:
        raise ValueError("Choose output_root or single-subject output_dir, not both")
    if options.output_dir is not None:
        if len(samples) != 1:
            raise ValueError("output_dir requires exactly one subject")
        samples = [SamplePaths(samples[0].root, Path(options.output_dir).expanduser().resolve())]
    elif options.output_root is not None:
        output_root = Path(options.output_root).expanduser().resolve()
        samples = [SamplePaths(sample.root, output_root / sample.root.name) for sample in samples]
    for sample in samples:
        check_registration_output(sample.root, sample.output, nicp_config=options.nicp_config,
                                  skull_template_dir=options.skull_template_dir)
    if options.deterministic:
        from .reproducibility import configure
        configure(options.device, options.seed)

    def wants(name: str) -> bool:
        return name in stage_set

    if wants("standardize"):
        from .mesh_ops import standardize_sample

        for sample in samples:
            standardize_sample(sample)

    if wants("render"):
        from .rendering import render_face_scan

        for sample in samples:
            render_face_scan(sample.standardized, device=options.device)

    if wants("fit3dmm"):
        from .bfm_fit import FitOptions, fit_render_image, make_fit_models

        fit_options = options.fit or FitOptions(device=options.device)
        fit_options.device = options.device
        fa, recon_model = make_fit_models(fit_options)
        for sample in samples:
            # Fresh initialization per subject; retain the original first-subject state.
            recon_model.init_coeff_tensors()
            fit_render_image(sample.standardized / "render.png", sample.standardized, fa, recon_model, fit_options)

    if wants("pose"):
        from .pose_normalize import normalize_pose

        for sample in samples:
            normalize_pose(sample.standardized, sample.posed, device="cpu")

    if wants("texture"):
        from .texture_transfer import transfer_render_texture

        for sample in samples:
            transfer_render_texture(
                sample.posed,
                source_color_stage_dir=sample.standardized,
                geometry_mode="nearest",
                texture_mode="ray",
            )

    if wants("face_nicp"):
        from .topology import run_face_nicp

        for sample in samples:
            run_face_nicp(
                sample.posed,
                source_color_stage_dir=sample.standardized,
                config_path=options.nicp_config,
                device=options.device,
            )

    if wants("skull_controls"):
        from .topology import make_skull_control_points

        for sample in samples:
            make_skull_control_points(sample.posed, template=False)

    if wants("skull_nicp"):
        from .topology import run_skull_nicp

        for sample in samples:
            run_skull_nicp(
                sample.posed,
                template_dir=options.skull_template_dir,
                config_path=options.nicp_config,
                device=options.device,
            )
