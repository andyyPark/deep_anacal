import logging
import numpy as np

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


def get_e_w(*, acat, component=1, force_detection=False):
    ename = f"fpfs_e{component}"
    wname = "fpfs_w"
    if isinstance(acat, list):
        return [
            (cat[ename], cat[wname] if not force_detection else 1.0) for cat in acat
        ]
    elif isinstance(acat, np.ndarray):
        return acat[ename], acat[wname] if not force_detection else 1.0
    else:
        raise ValueError(
            f"acal_res in 'get_e_w' must be of type 'list' or 'np.ndarray', not {type(acat)}"
            )


# TODO - Add selection weights
def get_R(*, acat, component=1, force_detection=False):
    egname = f"fpfs_de{component}_dg{component}"
    wgname = f"fpfs_dw_dg{component}"
    e_w = get_e_w(acat=acat, component=component, force_detection=force_detection)
    if isinstance(e_w, list):
        return [cat[wgname] * e + cat[egname] * w for cat, (e, w) in zip(acat, e_w)]
    else:
        e, w = e_w
        R = acat[wgname] * e + acat[egname] * w
        return R


# TODO - Compute bias from multiple realizations
def compute_m_and_c(
    *,
    acal_res,
    component=1,
    true_shear=0.02,
    force_detection=False,
):
    if isinstance(acal_res, dict):
        cat1 = acal_res["wide"]
        cat2 = acal_res["deep"]
    elif isinstance(acal_res, list):
        cat1 = acal_res
        cat2 = acal_res
    else:
        raise ValueError("acal_res must be of type 'dict' or 'list'")

    (e_plus, w_plus), (e_minus, w_minus) = get_e_w(
        acat=cat1, component=component, force_detection=force_detection
    )
    (R_plus, R_minus) = get_R(
        acat=cat2, component=component, force_detection=force_detection
    )

    if e_plus.sum() < e_minus.sum():
        e_plus, e_minus = e_minus, e_plus
        w_plus, w_minus = w_minus, w_plus
        R_plus, R_minus = R_minus, R_plus

    num1 = np.mean(w_plus * e_plus) - np.mean(w_minus * e_minus)
    num2 = np.mean(w_plus * e_plus) + np.mean(w_minus * e_minus)
    denom = np.mean(R_plus) + np.mean(R_minus)
    mbias = num1 / denom / true_shear - 1
    cbias = num2 / denom

    return mbias, cbias
