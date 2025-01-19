import os
import logging
import numpy as np
import functools
import fitsio

CAT_FIELDS = [
    "fpfs_e1",
    "fpfs_de1_dg1",
    "fpfs_e2",
    "fpfs_de2_dg2",
    "fpfs_q1",
    "fpfs_dq1_dg1",
    "fpfs_q2",
    "fpfs_dq2_dg2",
    "fpfs_w",
    "fpfs_dw_dg1",
    "fpfs_dw_dg2",
    "fpfs_m00",
    "fpfs_dm00_dg1",
    "fpfs_dm00_dg2",
    "fpfs_m20",
    "fpfs_dm20_dg1",
    "fpfs_dm20_dg2",
]


def setup_custom_logger(verbose=False):
    logger = logging.getLogger(__name__)
    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S --- ",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    return logger


def force_detection(ngrid=64, nstamp=1):
    indx = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
    indy = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
    ns = len(indx) * len(indy)
    inds = np.meshgrid(indy, indx, indexing="ij")
    yx = np.vstack([np.ravel(_) for _ in inds])
    dtype = np.dtype(
        [
            ("y", np.int32),
            ("x", np.int32),
            ("is_peak", np.int32),
            ("mask_value", np.int32),
        ]
    )
    detection = np.empty(ns, dtype=dtype)
    detection["y"] = yx[0]
    detection["x"] = yx[1]
    detection["is_peak"] = np.ones(ns)
    detection["mask_value"] = np.zeros(ns)
    return detection


def apply_sel(*, acat, sel, sel_min):
    if sel is None:
        cat = acat
    else:
        if isinstance(sel, str):
            sel = [sel]
            sel_min = [sel_min]
        mask = np.ones_like(acat, dtype=bool)
        for (s, s_min) in zip(sel, sel_min):
            mask &= acat[s] > s_min
        cat = acat[mask]
    return cat


def estimate_m_and_c(*, res, true_shear=0.02):
    res = np.asarray(res)
    nsim = res.shape[0]
    res_avg = np.average(res, axis=0)
    res_std = np.std(res, axis=0)
    m = res_avg[0] / res_avg[2] / true_shear - 1
    merr = res_std[0] / res_avg[2] / true_shear / np.sqrt(nsim)
    c = res_avg[1] / res_avg[2]
    cerr = res_std[1] / res_avg[2] / np.sqrt(nsim)
    return m, merr, c, cerr


@functools.lru_cache()
def cached_descwl_catalog_read():
    fname = os.path.join(os.environ["CATSIM_DIR"], "OneDegSq.fits")
    cat = fitsio.read(fname)
    cut = cat["r_ab"] < 26.0
    cat = cat[cut]
    return cat
