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


# TODO - Add checking for multiple selections
def get_e_w(*, acat, sel=None, sel_min=None, component=1, force_detection=False):
    if not isinstance(acat, np.ndarray):
        raise ValueError(
            f"acal_res in 'get_e_w' must be of type'np.ndarray', not {type(acat)}"
        )
    ename = f"fpfs_e{component}"
    wname = "fpfs_w"
    cat = apply_sel(acat=acat, sel=sel, sel_min=sel_min)
    return cat[ename], cat[wname] if not force_detection else 1.0


# TODO - Add selection weights
def get_R(
    *,
    acat,
    sel=None,
    sel_min=None,
    component=1,
    force_detection=False,
    correct_selection_bias=False,
):
    egname = f"fpfs_de{component}_dg{component}"
    wgname = f"fpfs_dw_dg{component}"
    cat = apply_sel(acat=acat, sel=sel, sel_min=sel_min)
    e_w = get_e_w(
        acat=cat,
        component=component,
        force_detection=force_detection,
    )
    e, w = e_w
    R = cat[wgname] * e + cat[egname] * w
    if correct_selection_bias:
        # TODO - include selection from M20 and fix hard-coding dg
        selg = f"fpfs_dm00_dg{component}"
        dg = 0.02
        tmp = acat[(acat[sel] + dg * acat[selg]) > sel_min]
        e_plus, w_plus = get_e_w(
            acat=tmp,
            component=component,
            force_detection=force_detection
        )
        ellip_plus = np.sum(e_plus * w_plus)
        del tmp
        tmp = acat[(acat[sel] - dg * acat[selg]) > sel_min]
        e_minus, w_minus = get_e_w(
            acat=tmp,
            component=component,
            force_detection=force_detection
        )
        ellip_minus = np.sum(e_minus * w_minus)
        del tmp
        R_sel = (ellip_plus - ellip_minus) / 2.0 / dg
    else:
        R_sel = 0.0
    return R, R_sel


# TODO - Compute bias from multiple realizations
def compute_m_and_c(
    *,
    wacal_res,
    dacal_res,
    sel=None,
    component=1,
    true_shear=0.02,
    force_detection=False,
    correct_selection_bias=False,
):
    if len(wacal_res) != 2 or len(dacal_res) != 2:
        raise ValueError("len of 'wacal_res' and 'dacal_res' must be 2")

    e_plus, w_plus = get_e_w(
        acat=wacal_res[0], sel=sel, component=component, force_detection=force_detection
    )
    e_minus, w_minus = get_e_w(
        acat=wacal_res[1], sel=sel, component=component, force_detection=force_detection
    )
    R_plus, R_sel_plus = get_R(
        acat=dacal_res[0],
        sel=sel,
        component=component,
        force_detection=force_detection,
        correct_selection_bias=correct_selection_bias,
    )
    R_minus, R_sel_minus = get_R(
        acat=dacal_res[0],
        sel=sel,
        component=component,
        force_detection=force_detection,
        correct_selection_bias=correct_selection_bias,
    )

    if e_plus.sum() < e_minus.sum():
        e_plus, e_minus = e_minus, e_plus
        w_plus, w_minus = w_minus, w_plus
        R_plus, R_minus = R_minus, R_plus
        R_sel_plus, R_sel_minus = R_sel_minus, R_sel_plus

    num1 = np.mean(w_plus * e_plus) - np.mean(w_minus * e_minus)
    num2 = np.mean(w_plus * e_plus) + np.mean(w_minus * e_minus)
    denom = np.mean(R_plus) + np.mean(R_sel_plus) + np.mean(R_minus) + np.mean(R_sel_minus)
    mbias = num1 / denom / true_shear - 1
    cbias = num2 / denom

    return mbias, cbias
