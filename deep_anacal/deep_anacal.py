import numpy as np
import anacal

def get_max_smooth_scale(*, fwhm_w, fwhm_d, fpfs_config):
    fac = 2.0 * np.sqrt(2.0 * np.log(2))
    if isinstance(fpfs_config, anacal.fpfs.FpfsConfig):
        fpfs_smooth_scale = fac * fpfs_config.sigma_arcsec
    else:
        fpfs_smooth_scale = fac * fpfs_config
    return max(fwhm_w, fwhm_d, fpfs_smooth_scale) / fac

def measure_eR(*, gal_array, psf_array, noise_array, noise_variance, fpfs_config, scale, detection, 
               component=1, force_detection=True):
    acal_res = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        pixel_scale=scale,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        noise_array=noise_array,
        noise_variance=noise_variance,
        detection=detection,
    )
    ename = f"fpfs_e{component}"
    egname = f"fpfs_de{component}_dg{component}"
    wname = "fpfs_w"
    wgname = f"fpfs_dw_dg{component}"
    if force_detection:
        return {
            "e": acal_res[ename],
            "R": acal_res[egname],
        }
    else:
        return {
            "e": acal_res[wname] * acal_res[ename],
            "R": acal_res[ename] * acal_res[wgname] + acal_res[wname] * acal_res[egname]
        }

def match_noise(*, deep_noise_fac, noise_std_wide, noise_array_wide, do_rotate=False):
    if deep_noise_fac > 1.0:
        raise ValueError(
            f"{deep_noise_fac} must be less than 1.0"
        )
    # NOTE - Current implementation of anacal assumes
    #        2 * noise_std_wide**2 after adding pure noise
    if deep_noise_fac <= 0.5:
        noise_variance = noise_std_wide**2 / 2
        renoise_array_wide = None
        renoise_array_deep = (1 - deep_noise_fac) * noise_array_wide
    else:
         noise_variance = (2*deep_noise_fac - 1) * noise_std_wide**2 / 2
         renoise_array_wide = (2*deep_noise_fac-1) * noise_array_wide
         renoise_array_deep = deep_noise_fac * noise_array_wide

    if do_rotate:
        renoise_array_wide = np.rot90(renoise_array_wide, k=1)
        renoise_array_deep = np.rot90(renoise_array_deep, k=1)

    return noise_variance, renoise_array_wide, renoise_array_deep