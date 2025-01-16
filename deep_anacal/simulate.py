import galsim
import numpy as np
from .utils import setup_custom_logger

logger = setup_custom_logger()

# TODO - Add unit tests

def build_fixed_psf(
        *,
        fwhm,
        psf_name="gaussian",
):
    if psf_name not in ["gaussian", "moffat"]:
        raise ValueError(f"{psf_name} is not supported")
    logger.info(
        f"Creating {psf_name} PSF with fixed size of {fwhm}"
    )
    if psf_name == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm)
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0)
    return psf
    

def simulate_exponential(
        *,
        ngrid,
        scale,
        do_shift=False,
        flux=1,
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        fwhm=0.6,
        psf_name="gaussian",
        fix_psf=True,
):
    logger = setup_custom_logger()
    logger.info(
        f"Simulating exponential galaxies with g1={g1} and g2={g2} and {psf_name} psf with fwhm={fwhm}"
        )
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal = galsim.Exponential(half_light_radius=hlr).withFlux(flux).shear(g1=g1, g2=g2)
    # TODO - Build variable psf
    if fix_psf:
        psf = build_fixed_psf(fwhm=fwhm, psf_name=psf_name)
    gal = galsim.Convolve([gal, psf], gsparams=gsparams)
    if do_shift:
        shift_x = rng.uniform(low=-0.5, high=0.5) * scale
        shift_y = rng.uniform(low=-0.5, high=0.5) * scale
        gal.shift(shift_x, shift_y)
    gal = gal.shift(0.5 * scale, 0.5 * scale)
    gal_array = gal.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    psf_array = psf.shift(
        0.5 * scale, 0.5 * scale
        ).drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    return gal_array, psf_array

def simulate_pure_noise(*, seed, image, s2n, deep_noise_frac=1.0, renoise=True):
    # Make sure pure noise for renoising has different seed
    if renoise:
        seed = seed + 1
    rng = np.random.RandomState(seed)
    noise_std = np.sqrt(np.sum(image**2)) / s2n
    img_noise = rng.normal(scale=noise_std * deep_noise_frac, size=(image.shape))
    return img_noise, noise_std


def sim_wide_deep(
        *,
        seed,
        ngrid,
        scale,
        do_shift=False,
        flux=1,
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        psf_name="gaussian",
        fix_psf=True,
        gal_type='exp',
        fwhm_w=0.9,
        fwhm_d=0.7,
        s2n=1e8,
        deep_noise_frac=1.0):
    assert fwhm_w >= fwhm_d, "deep field usually has lower fwhm than wide field"
    # TODO - Add WLDeblend galaxy
    if gal_type == 'exp':
        make_sim = simulate_exponential
    gal_array_w, psf_array_w = make_sim(
        ngrid=ngrid,
        scale=scale,
        do_shift=do_shift,
        flux=flux,
        g1=g1,
        g2=g2,
        hlr=hlr,
        psf_name=psf_name,
        fix_psf=fix_psf,
        fwhm=fwhm_w,
    )
    gal_array_d, psf_array_d = make_sim(
        ngrid=ngrid,
        scale=scale,
        do_shift=do_shift,
        flux=flux,
        g1=g1,
        g2=g2,
        hlr=hlr,
        psf_name=psf_name,
        fix_psf=fix_psf,
        fwhm=fwhm_d,
    )

    image_noise_w, noise_std_w = simulate_pure_noise(
        seed=seed, image=gal_array_w, s2n=s2n, deep_noise_frac=1.0
    )
    image_noise_d, noise_std_d = simulate_pure_noise(
        seed=seed, image=gal_array_d, s2n=s2n, deep_noise_frac=deep_noise_frac
    )

    return {
        "wide": gal_array_w + image_noise_w,
        "wide_psf": psf_array_w,
        "wide_noise_std": noise_std_w,
        "deep": gal_array_d + image_noise_d,
        "deep_psf": psf_array_d,
        "deep_noise_std": noise_std_d
    }




