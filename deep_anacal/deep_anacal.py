import numpy as np
import numpy.lib.recfunctions as rfn
import anacal


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
    wname = "fpfs_w"
    wgname = f"fpfs_dw_dg{component}"
    if force_detection:
        return {
            "e": wide_cat[ename],
            "R": deep_cat[egname],
        }
    else:
        return {
            "e": wide_cat[wname] * wide_cat[ename],
            "R": deep_cat[ename] * deep_cat[wgname]
            + deep_cat[wname] * deep_cat[egname],
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
    seed,
    ftask_w,
    ftask_d,
    gal_array_w,
    gal_array_d,
    psf_array_w,
    psf_array_d,
    noise_var_w,
    noise_var_d,
    detection,
):
    rng = np.random.RandomState(seed=seed)
    # Wide + Deep for response:
    pure_noise_d = pure_noise(rng, noise_var_d, gal_array_d.shape)
    pure_noise_d_rot = pure_noise(rng, noise_var_d, gal_array_d.shape, rotate=True)
    z2d = np.zeros_like(gal_array_d)
    deep_data, deep_noise = ftask_d.run_psf_array(
        gal_array=gal_array_d,
        psf_array=psf_array_d,
        det=detection,
        noise_array=pure_noise_d_rot,
    )
    pure_noise_w1 = pure_noise(rng, noise_var_w / 2, gal_array_w.shape)
    pure_noise_w2 = pure_noise(rng, noise_var_w / 2, gal_array_w.shape)
    _, pure_noise_w1 = ftask_w.run_psf_array(
        gal_array=z2d,
        psf_array=psf_array_w,
        det=detection,
        noise_array=pure_noise_w1,
    )
    _, pure_noise_w2 = ftask_w.run_psf_array(
        gal_array=z2d,
        psf_array=psf_array_w,
        det=detection,
        noise_array=pure_noise_w2,
    )
    deep_data = deep_data + pure_noise_w1 + pure_noise_w2
    deep_noise = deep_noise + pure_noise_w1

    deep_data = rfn.unstructured_to_structured(arr=deep_data, dtype=ftask_d.dtype)
    deep_noise = rfn.unstructured_to_structured(arr=deep_noise, dtype=ftask_d.dtype)
    src_deep = {"data": deep_data, "noise": deep_noise}
    # Wide + Deep for ellipticity
    wide_data, _ = ftask_w.run_psf_array(
        gal_array=gal_array_w, psf_array=psf_array_w, det=detection, noise_array=None
    )
    _, pure_noise_d1 = ftask_d.run_psf_array(
        gal_array=z2d,
        psf_array=psf_array_d,
        det=detection,
        noise_array=pure_noise_d_rot,
    )
    _, pure_noise_d2 = ftask_d.run_psf_array(
        gal_array=z2d, psf_array=psf_array_d, det=detection, noise_array=pure_noise_d
    )
    wide_data = wide_data + pure_noise_d1 + pure_noise_d2
    wide_data = rfn.unstructured_to_structured(arr=wide_data, dtype=ftask_w.dtype)
    src_wide = {"data": wide_data, "noise": None}
    return src_wide, src_deep


def run_wide_deep_meas(*, fpfs_config, ftask_w, ftask_d, src_wide, src_deep):
    # Wide source measurement
    wide_meas = anacal.fpfs.measure_fpfs(
        C0=fpfs_config.c0,
        v_min=fpfs_config.v_min,
        omega_v=fpfs_config.omega_v,
        pthres=fpfs_config.pthres,
        std_m00=ftask_w.std_m00,
        m00_min=ftask_w.std_m00 * fpfs_config.snr_min,
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
        std_m00=ftask_d.std_m00,
        m00_min=ftask_d.std_m00 * fpfs_config.snr_min,
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
    seed,
    scale,
    fpfs_config,
    gal_array_w,
    gal_array_d,
    psf_array_w,
    psf_array_d,
    noise_var_w,
    noise_var_d,
    detection,
):
    # After matching noise they have one wide + 2 deep
    noise_var_match = 0.5 * (noise_var_w + 2 * noise_var_d)
    ftask_w = create_fpfs_task(fpfs_config, scale, noise_var_match, psf_array_w)
    ftask_d = create_fpfs_task(fpfs_config, scale, noise_var_match, psf_array_d)
    src_wide, src_deep = match_noise(
        seed=seed,
        ftask_w=ftask_w,
        ftask_d=ftask_d,
        gal_array_w=gal_array_w,
        gal_array_d=gal_array_d,
        psf_array_w=psf_array_w,
        psf_array_d=psf_array_d,
        noise_var_w=noise_var_w,
        noise_var_d=noise_var_d,
        detection=detection,
    )
    wide_fpfs, deep_fpfs = run_wide_deep_meas(
        fpfs_config=fpfs_config,
        ftask_w=ftask_w,
        ftask_d=ftask_d,
        src_wide=src_wide,
        src_deep=src_deep,
    )
    return wide_fpfs, deep_fpfs
