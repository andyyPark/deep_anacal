import logging
import numpy as np


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

def get_e_w(*, acal_res, wname, wgname, ename, egname):
    shear_res = dict()
    for i, sign in enumerate(["plus", "minus"]):
        shear_res[sign] = dict()
        for quant in [wname, wgname, ename, egname]:
            shear_res[sign][quant] = acal_res[i][quant]
    return shear_res

# TODO - Compute bias from multiple realizations
def compute_m_and_c(
        *,
        acal_res,
        component=1,
        true_shear=0.02,
        force_detection=False,
):
    wname = "fpfs_w"
    wgname = f"fpfs_dw_dg{component}"
    ename = f"fpfs_e{component}"
    egname = f"fpfs_de{component}_dg{component}"
    shear_res = get_e_w(acal_res=acal_res,
                        wname=wname, wgname=wgname,
                        ename=ename, egname=egname)
    if force_detection:
        for sign in ["plus", "minus"]:
            shear_res[sign][wname] = 1.0
            shear_res[sign][wgname] = 0.0
    
    def _get_stuff(shear_res, sign, key):
        return shear_res[sign][key]
    
    w_plus = _get_stuff(shear_res, "plus", wname)
    e_plus = _get_stuff(shear_res, "plus", ename)
    wg_plus = _get_stuff(shear_res, "plus", wgname)
    eg_plus = _get_stuff(shear_res, "plus", egname)
    R_plus = wg_plus * e_plus + w_plus * eg_plus

    w_minus = _get_stuff(shear_res, "minus", wname)
    e_minus = _get_stuff(shear_res, "minus", ename)
    wg_minus = _get_stuff(shear_res, "minus", wgname)
    eg_minus = _get_stuff(shear_res, "minus", egname)
    R_minus = wg_minus * e_minus + w_minus * eg_minus
    
    num1 = np.sum(w_plus * e_plus) - np.sum(w_minus * e_minus)
    num2 = np.sum(w_plus * e_plus) + np.sum(w_minus * e_minus)
    denom = np.sum(R_plus) + np.sum(R_minus)
    mbias = num1 / denom / true_shear - 1
    cbias = num2 / denom
    return mbias, cbias