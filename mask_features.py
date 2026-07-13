"""
Feature Extraction Module.
Implements Hounsfield Unit clamping, intensity quantization, luma-based mask binarization,
SimpleITK geometry resampling, and texture descriptor computations (GLCM, GLDS, GLRLM, LBP).

This file contains a curated selection of core pipeline fragments and algorithmic logic 
from the original production code. End-to-end file I/O operations and local dataset loops 
are omitted for data privacy and repository cleanliness.
"""

from typing import Dict, List
import numpy as np
import SimpleITK as sitk
import pyfeats

# === PREPROCESSING PARAMETERS ===
HU_CLAMP = (-1350.0, 150.0)   # HU windowing before quantization
NG_LEVELS = 64                # Number of gray levels for matrices
GLCM_IGNORE_ZEROS = True      # 0 = background (outside mask), ignored in GLCM
GLDS_DX, GLDS_DY = [0, 1, 1, 1], [1, 1, 0, -1]
LBP_P, LBP_R = [8, 16, 24], [1, 2, 3]

MASK_WHITE_ONLY = True        # True = treat only white/bright pixels as ROI
MASK_GRAY_THRESH = 127        # Luminance threshold for gray/RGB masks


def clamp_quantize(arr2d: np.ndarray) -> np.ndarray:
    """Clamps intensities into HU_CLAMP and quantizes to [0..NG_LEVELS-1]"""
    lo, hi = HU_CLAMP
    a = np.clip(arr2d, lo, hi)
    scale = (NG_LEVELS - 1) / (hi - lo)
    q = ((a - lo) * scale).round().astype(np.uint8)
    return q


def _rgb_to_gray_u8(rgb_arr: np.ndarray) -> np.ndarray:
    """
    Converts (y,x,3|4) RGB/RGBA array to 8-bit grayscale using standard luma weights.
    Ignores the alpha channel if present.
    """
    if rgb_arr.shape[-1] == 4:
        rgb_arr = rgb_arr[..., :3]
    r = rgb_arr[..., 0].astype(np.float32)
    g = rgb_arr[..., 1].astype(np.float32)
    b = rgb_arr[..., 2].astype(np.float32)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def _binarize_mask_from_any(arr: np.ndarray) -> np.ndarray:
    """
    Transforms arbitrary 2D/3D/4D input arrays into a clean boolean 2D mask ROI.
    Handles multichannel images via luma thresholding.
    """
    a = arr
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]

    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            gray = _rgb_to_gray_u8(a)
            mask = gray >= MASK_GRAY_THRESH if MASK_WHITE_ONLY else gray > 0
            return mask.astype(bool)
        if a.shape[0] == 1:
            a = a[0]

    if a.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {a.shape}")
        
    if MASK_WHITE_ONLY:
        mask = a.astype(np.float32) >= float(MASK_GRAY_THRESH)
    else:
        mask = a.astype(np.float32) > 0.0
    return mask.astype(bool)


def _resample_mask_to_image(mask_bool: np.ndarray, img_fp: str) -> np.ndarray:
    """
    Resamples a binary mask to match the geometry (spacing, origin, direction) 
    of a reference DICOM slice using SimpleITK NearestNeighbor interpolation.
    """
    ref_img = sitk.ReadImage(img_fp)
    mask_img = sitk.GetImageFromArray(mask_bool.astype(np.uint8))
    
    mask_img.SetSpacing(ref_img.GetSpacing())
    mask_img.SetOrigin(ref_img.GetOrigin())
    mask_img.SetDirection(ref_img.GetDirection())

    res = sitk.Resample(
        mask_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8
    )
    out = sitk.GetArrayFromImage(res)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    return out > 0


def glcm_image_from_roi(img_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sets background to 0 and shifts ROI to [1..255] for strict background handling."""
    out = np.zeros_like(img_u8, dtype=np.uint8)
    if mask.any():
        roi = img_u8[mask].astype(np.uint16) + 1
        roi = np.clip(roi, 0, 255).astype(np.uint8)
        out[mask] = roi
    return out


def compute_pyfeats_for_slice_masked(img_u8: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Computes texture descriptors (GLCM, GLDS, GLRLM, LBP) inside the boolean mask."""
    feats: Dict[str, float] = {}

    if img_u8.min() == img_u8.max():
        img_u8 = img_u8.copy()
        img_u8[0, 0] = np.uint8((int(img_u8[0, 0]) + 1) % NG_LEVELS)

    # ---- GLCM
    try:
        glcm_img = glcm_image_from_roi(img_u8, mask)
        glcm_mean, _, glcm_labels_mean, _ = pyfeats.glcm_features(
            np.ascontiguousarray(glcm_img), ignore_zeros=GLCM_IGNORE_ZEROS
        )
        for n, v in zip(glcm_labels_mean, glcm_mean.flatten()):
            feats[f"GLCM_{n}"] = float(v)
    except Exception:
        feats["__warn_glcm__"] = 1.0

    # ---- GLDS
    try:
        glds_vals, glds_names = pyfeats.glds_features(
            np.ascontiguousarray(img_u8), mask.astype(bool), Dx=GLDS_DX, Dy=GLDS_DY
        )
        for n, v in zip(glds_names, glds_vals.flatten()):
            feats[f"GLDS_{n}"] = float(v)
    except Exception:
        feats["__warn_glds__"] = 1.0

    # ---- GLRLM
    try:
        glrlm_vals, glrlm_names = pyfeats.glrlm_features(
            np.ascontiguousarray(img_u8), mask.astype(bool), Ng=NG_LEVELS
        )
        for n, v in zip(glrlm_names, glrlm_vals.flatten()):
            feats[f"GLRLM_{n}"] = float(v)
    except Exception:
        feats["__warn_glrlm__"] = 1.0

    # ---- LBP
    try:
        lbp_vals, lbp_names = pyfeats.lbp_features(
            np.ascontiguousarray(img_u8), mask.astype(bool), P=LBP_P, R=LBP_R
        )
        for n, v in zip(lbp_names, lbp_vals.flatten()):
            feats[f"LBP_{n}"] = float(v)
    except Exception:
        feats["__warn_lbp__"] = 1.0

    return feats
