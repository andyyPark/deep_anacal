import numpy as np
import numpy.lib.recfunctions as rfn
import anacal

nx = 64
ny = 64
norder = 6
klim = 2.650718801466388/0.2

def prepare_fpfs_bases(*, scale, fpfs_config):
    """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
    bfunc = []
    colnames = []
    sfunc, snames = anacal.fpfs.base.shapelets2d(
        norder=norder,
        npix=fpfs_config.npix,
        sigma=scale / fpfs_config.sigma_arcsec,
        kmax=klim*scale,
    )
    bfunc.append(sfunc)
    colnames = colnames + snames
    dfunc, dnames = anacal.fpfs.base.detlets2d(
        npix=fpfs_config.npix,
        sigma=scale / fpfs_config.sigma_arcsec,
        kmax=klim*scale,
    )
    bfunc.append(dfunc)
    colnames = colnames + dnames
    bfunc = np.vstack(bfunc)
    bfunc_use = np.transpose(bfunc, (1, 2, 0))
    dtype = [(name, "f8") for name in colnames]
    return bfunc_use, dtype


def get_max_smooth_scale(*, fwhm_w, fwhm_d, fpfs_config):
    fac = 2.0 * np.sqrt(2.0 * np.log(2))
    if isinstance(fpfs_config, anacal.fpfs.FpfsConfig):
        fpfs_smooth_scale = fac * fpfs_config.sigma_arcsec
    else:
        fpfs_smooth_scale = fac * fpfs_config
    return max(fwhm_w, fwhm_d, fpfs_smooth_scale) / fac


def get_e_and_R(*, wide_cat, deep_cat, component=1, force_detection=True):
    ename = f"fpfs_e{component}"
    egname = f"fpfs_de{component}_dg{component}"
    qname = f"fpfs_q{component}"
    qgname = f"fpfs_dq{component}_dg{component}"
    wname = "fpfs_w"
    wgname = f"fpfs_dw_dg{component}"
    if force_detection:
        return {
            "e": wide_cat[ename],
            "q": wide_cat[qname],
            "R": deep_cat[egname],
            "Rq": deep_cat[qgname],
        }
    else:
        return {
            "e": wide_cat[wname] * wide_cat[ename],
            "q": wide_cat[wname] * wide_cat[qname],
            "R": deep_cat[ename] * deep_cat[wgname]
            + deep_cat[wname] * deep_cat[egname],
            "Rq": deep_cat[qname] * deep_cat[wgname]
            + deep_cat[wname] * deep_cat[qgname],
        }


def create_fpfs_task(fpfs_config, scale, noise_var, psf_array, do_detection=True):
    return anacal.fpfs.FpfsTask(
        npix=fpfs_config.npix,
        pixel_scale=scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        noise_variance=noise_var,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
        do_detection=do_detection,
        bound=fpfs_config.bound,
    )


def pure_noise(rng, noise_var, shape, rotate=False):
    noise = rng.normal(scale=np.sqrt(noise_var), size=shape)
    return np.rot90(noise, k=-1) if rotate else noise


def match_noise(
    *,
    scale,
    fpfs_config,
    gal_array_w,
    gal_array_d,
    psf_array_w,
    psf_array_d,
    noise_array_w,
    noise_array_d,
    noise_array_d90,
    detection_w,
    detection_d,
):
    mtask = anacal.fpfs.FpfsDeepWideImage(
        nx=nx, ny=ny, scale=scale,
        sigma_arcsec=fpfs_config.sigma_arcsec, klim=klim,
        use_estimate=True
    )
    bfunc_use, dtype = prepare_fpfs_bases(scale=scale, fpfs_config=fpfs_config)

    nu_wide = mtask.measure_source(
        gal_array=gal_array_w,
        filter_image=bfunc_use,
        psf_array=psf_array_w,
        det=detection_w,
        do_rotate=False
    )
    dnu_deep_w = mtask.measure_source(
        gal_array=noise_array_d + noise_array_d90,
        filter_image=bfunc_use,
        psf_array=psf_array_d,
        det=detection_w,
        do_rotate=False
    )
    meas_wide = nu_wide + dnu_deep_w
    meas_wide = rfn.unstructured_to_structured(meas_wide, dtype=dtype)
    src_wide = {"data": meas_wide, "noise": None}

    nu_deep = mtask.measure_source(
        gal_array=gal_array_d,
        filter_image=bfunc_use,
        psf_array=psf_array_d,
        det=detection_d,
        do_rotate=False
        )
    dnu_deep_d = mtask.measure_source(
        gal_array=noise_array_d90,
        filter_image=bfunc_use,
        psf_array=psf_array_d,
        det=detection_d,
        do_rotate=False
    )
    dnu_wide_d = mtask.measure_source(
        gal_array=noise_array_w,
        filter_image=bfunc_use,
        psf_array=psf_array_w,
        det=detection_d,
        do_rotate=False
    )
    meas_deep = nu_deep + dnu_deep_d + dnu_wide_d
    meas_deep = rfn.unstructured_to_structured(meas_deep, dtype=dtype)
    meas_deep_n = rfn.unstructured_to_structured(dnu_deep_d + 0.5*dnu_wide_d, dtype=dtype)
    src_deep = {"data": meas_deep, "noise": meas_deep_n}
    return src_wide, src_deep


def run_wide_deep_meas(*, fpfs_config, std_m00, src_wide, src_deep):
    # Wide source measurement
    wide_meas = anacal.fpfs.measure_fpfs(
        C0=fpfs_config.c0,
        v_min=fpfs_config.v_min,
        omega_v=fpfs_config.omega_v,
        pthres=fpfs_config.pthres,
        std_m00=std_m00,
        m00_min=std_m00 * fpfs_config.snr_min,
        r2_min=fpfs_config.r2_min,
        omega_r2=fpfs_config.omega_r2,
        x_array=src_wide["data"],
        y_array=src_wide["noise"],
    )
    map_dict = {name: "fpfs_" + name for name in wide_meas.dtype.names}
    wide_fpfs = rfn.rename_fields(wide_meas, map_dict)
    # Deep source measurement
    deep_meas = anacal.fpfs.measure_fpfs(
        C0=fpfs_config.c0,
        v_min=fpfs_config.v_min,
        omega_v=fpfs_config.omega_v,
        pthres=fpfs_config.pthres,
        std_m00=std_m00,
        m00_min=std_m00 * fpfs_config.snr_min,
        r2_min=fpfs_config.r2_min,
        omega_r2=fpfs_config.omega_r2,
        x_array=src_deep["data"],
        y_array=src_deep["noise"],
    )
    map_dict = {name: "fpfs_" + name for name in deep_meas.dtype.names}
    deep_fpfs = rfn.rename_fields(deep_meas, map_dict)
    return wide_fpfs, deep_fpfs


def run_deep_anacal(
    *,
    scale,
    fpfs_config,
    gal_array_w,
    gal_array_d,
    psf_array_w,
    psf_array_d,
    noise_var_w,
    noise_var_d,
    noise_array_w,
    noise_array_d,
    noise_array_d90,
    detection_w,
    detection_d,
):
    ftask_w = create_fpfs_task(fpfs_config, scale, 0.5 * noise_var_w, psf_array_w)
    ftask_d = create_fpfs_task(fpfs_config, scale, noise_var_d, psf_array_d)
    std_m00 = np.sqrt(ftask_w.std_m00 ** 2 + ftask_d.std_m00 ** 2)
    src_wide, src_deep = match_noise(
        scale=scale,
        fpfs_config=fpfs_config,
        gal_array_w=gal_array_w,
        gal_array_d=gal_array_d,
        psf_array_w=psf_array_w,
        psf_array_d=psf_array_d,
        noise_array_w=noise_array_w,
        noise_array_d=noise_array_d,
        noise_array_d90=noise_array_d90,
        detection_w=detection_w,
        detection_d=detection_d,
    )
    wide_fpfs, deep_fpfs = run_wide_deep_meas(
        fpfs_config=fpfs_config,
        std_m00=std_m00,
        src_wide=src_wide,
        src_deep=src_deep,
    )
    return wide_fpfs, deep_fpfs
