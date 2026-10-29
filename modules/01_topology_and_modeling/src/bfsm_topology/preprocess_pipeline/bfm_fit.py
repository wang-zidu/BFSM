from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from numbers import Real

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import face_alignment
except ImportError:
    face_alignment = None

from . import rendering
from .core import get_recon_model
from .core import losses, utils


@dataclass
class FitOptions:
    tar_size: int = 224
    recon_model: str = "bfm09"
    first_rf_iters: int = 1000
    first_nrf_iters: int = 500
    rf_lr: float = 1e-2
    nrf_lr: float = 1e-2
    lm_loss_w: float = 100
    rgb_loss_w: float = 1.6
    id_reg_w: float = 1e-3
    exp_reg_w: float = 0.8e-3
    tex_reg_w: float = 1.7e-6
    tex_w: float = 1
    device: str = "cuda:0"



def _validate_fit_options(opt: FitOptions) -> None:
    for name, minimum in (("tar_size", 1), ("first_rf_iters", 0), ("first_nrf_iters", 0)):
        value = getattr(opt, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("rf_lr", "nrf_lr", "lm_loss_w", "rgb_loss_w", "id_reg_w", "exp_reg_w", "tex_reg_w", "tex_w"):
        value = getattr(opt, name)
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

def _weighted_landmarks(device: str, base_weights, blend: bool):
    w = torch.zeros(68).to(device)
    w[36:43] = 100
    w[48:60] = 100
    w[31] = 50
    w[33] = 50
    w[35] = 50
    w = w / w.sum()
    if blend:
        w = (w + base_weights) / 2
    return w


def _nonrigid_loss(recon_model, pred_dict, img_tensor, lms, lm_weights, opt: FitOptions):
    rendered_img = pred_dict["rendered_img"]
    mask = rendered_img[:, :, :, 3].detach()
    photo = losses.photo_loss(rendered_img[:, :, :, :3], img_tensor, mask > 0)
    lm = losses.lm_loss(pred_dict["lms_proj"], lms, lm_weights, img_size=opt.tar_size)
    id_reg = losses.get_l2(recon_model.get_id_tensor())
    exp_reg = losses.get_l2(recon_model.get_exp_tensor())
    tex_reg = losses.get_l2(recon_model.get_tex_tensor())
    tex_loss = losses.reflectance_loss(pred_dict["face_texture"], recon_model.get_skinmask())
    base = (
        lm * opt.lm_loss_w
        + id_reg * opt.id_reg_w
        + exp_reg * opt.exp_reg_w
        + tex_reg * opt.tex_reg_w
        + tex_loss * opt.tex_w
        + photo * opt.rgb_loss_w
    )
    return base, lm, photo, tex_loss, id_reg, exp_reg, tex_reg


def _run_nonrigid_depth_stage(
    recon_model,
    img_tensor,
    lms,
    lm_weights,
    depth_gt,
    depth_gt_mask,
    opt: FitOptions,
    iters: int,
    depth_divisor: float,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(
        [
            recon_model.get_id_tensor(),
            recon_model.get_exp_tensor(),
            recon_model.get_gamma_tensor(),
            recon_model.get_tex_tensor(),
            recon_model.get_rot_tensor(),
            recon_model.get_trans_tensor(),
        ],
        lr=lr,
    )
    for _ in tqdm(range(iters)):
        optimizer.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=True)
        base, *_ = _nonrigid_loss(recon_model, pred, img_tensor, lms, lm_weights, opt)
        pred_depth, pred_mask = rendering.get_depth(pred["mesh"], device=opt.device)
        depth_loss = (((depth_gt - pred_depth) * pred_mask * depth_gt_mask) ** 2).mean()
        loss = base + opt.lm_loss_w / depth_divisor * depth_loss
        loss.backward()
        optimizer.step()


def fit_render_image(image_path: str | Path, res_folder: str | Path, fa, recon_model, opt: FitOptions) -> None:
    _validate_fit_options(opt)
    image_path = Path(image_path)
    res_folder = Path(res_folder)
    img_arr = cv2.imread(str(image_path))[:, :, ::-1]
    depth_gt = torch.tensor(np.load(str(image_path)[:-4] + "_depth.npy"), dtype=torch.float32, device=opt.device)
    depth_gt_mask = torch.tensor(np.load(str(image_path)[:-4] + "_depth_mask.npy"), dtype=torch.float32, device=opt.device)
    lms = fa.get_landmarks_from_image(img_arr)[0]
    lms = torch.tensor(lms[:, :2][None, ...], dtype=torch.float32, device=opt.device)
    img_tensor = torch.tensor(img_arr[None, ...].copy(), dtype=torch.float32, device=opt.device)
    lm_weights = utils.get_lm_weights(opt.device)

    rigid_optimizer = torch.optim.Adam(
        [recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
        lr=opt.rf_lr,
    )
    for _ in tqdm(range(opt.first_rf_iters)):
        rigid_optimizer.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=False)
        lm = losses.lm_loss(pred["lms_proj"], lms, lm_weights, img_size=opt.tar_size)
        total_loss = opt.lm_loss_w * lm
        total_loss.backward()
        rigid_optimizer.step()

    nonrigid_optimizer = torch.optim.Adam(
        [
            recon_model.get_id_tensor(),
            recon_model.get_exp_tensor(),
            recon_model.get_gamma_tensor(),
            recon_model.get_tex_tensor(),
            recon_model.get_rot_tensor(),
            recon_model.get_trans_tensor(),
        ],
        lr=opt.nrf_lr,
    )
    for _ in tqdm(range(opt.first_nrf_iters)):
        nonrigid_optimizer.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=True)
        loss, *_ = _nonrigid_loss(recon_model, pred, img_tensor, lms, lm_weights, opt)
        loss.backward()
        nonrigid_optimizer.step()

    rigid_optimizer = torch.optim.Adam(
        [recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
        lr=opt.rf_lr,
    )
    for _ in tqdm(range(500)):
        rigid_optimizer.zero_grad()
        pred = recon_model(recon_model.get_packed_tensors(), render=True)
        lm = losses.lm_loss(pred["lms_proj"], lms, lm_weights, img_size=opt.tar_size)
        pred_depth, pred_mask = rendering.get_depth(pred["mesh"], device=opt.device)
        depth_loss = (((depth_gt - pred_depth) * pred_mask * depth_gt_mask) ** 2).mean()
        total_loss = opt.lm_loss_w * lm + opt.lm_loss_w / 10 * depth_loss
        total_loss.backward()
        rigid_optimizer.step()

    _run_nonrigid_depth_stage(recon_model, img_tensor, lms, lm_weights, depth_gt, depth_gt_mask, opt, 1000, 1, opt.nrf_lr)
    lm_w = _weighted_landmarks(opt.device, lm_weights, blend=True)
    _run_nonrigid_depth_stage(recon_model, img_tensor, lms, lm_w, depth_gt, depth_gt_mask, opt, 1000, 0.1, opt.nrf_lr)
    lm_w = _weighted_landmarks(opt.device, lm_weights, blend=False)
    _run_nonrigid_depth_stage(recon_model, img_tensor, lms, lm_w, depth_gt, depth_gt_mask, opt, 1000, 0.05, opt.nrf_lr)
    _run_nonrigid_depth_stage(recon_model, img_tensor, lms, lm_w, depth_gt, depth_gt_mask, opt, 1000, 0.03, opt.nrf_lr * 0.1)

    with torch.no_grad():
        coeffs = recon_model.get_packed_tensors()
        pred = recon_model(coeffs, render=True)
        rendered_img = pred["rendered_img"].cpu().numpy().squeeze()
        out_img = rendered_img[:, :, :3].astype(np.uint8)
        out_mask = (rendered_img[:, :, 3] > 0).astype(np.uint8)[..., None]
        composed_img = img_arr * (1 - out_mask) + out_img * out_mask
        utils.mymkdirs(res_folder)
        basename = image_path.stem
        cv2.imwrite(str(res_folder / f"{basename}_composed_img.jpg"), composed_img[:, :, ::-1])
        np.save(res_folder / f"{basename}_coeffs.npy", coeffs.detach().cpu().numpy().squeeze())
        vs = pred["vs"].cpu().numpy().squeeze()
        tri = pred["tri"].cpu().numpy().squeeze()
        color = pred["color"].cpu().numpy().squeeze()
        utils.save_obj(res_folder / f"{basename}_mesh.obj", vs, tri + 1, np.clip(color, 0, 255))


def make_fit_models(opt: FitOptions):
    _validate_fit_options(opt)
    if face_alignment is None:
        raise RuntimeError(
            "fit3dmm requires the face_alignment package. "
            "The configured environment does not currently provide it."
        )
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device=opt.device)
    if torch.are_deterministic_algorithms_enabled():
        # FaceAlignment enables autotuning in its constructor; retain the caller's strict mode.
        torch.backends.cudnn.benchmark = False
    recon_model = get_recon_model(
        model=opt.recon_model,
        device=opt.device,
        batch_size=1,
        img_size=opt.tar_size,
    )
    return fa, recon_model
