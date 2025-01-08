import galsim
import numpy as np
from .utils import setup_custom_logger

# TODO - Add unit tests

def build_fixed_psf(
        *,
        fwhm,
        psf_name="gaussian",
        logger=None,
):
    if psf_name not in ["gaussian", "moffat"]:
        raise ValueError(f"{psf_name} is not supported")
    if logger is None:
        logger = setup_custom_logger()
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
        seed,
        ngrid,
        ny,
        nx,
        scale,
        do_shift=False,
        flux=1,
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        fwhm=0.6,
        psf_name="gaussian",
        fix_psf=True,
        s2n=10,
        noise_fac=1.0,
):
    logger = setup_custom_logger()
    if nx % ngrid != 0:
        raise ValueError("nx is not divisible by ngrid")
    if ny % ngrid != 0:
        raise ValueError("ny is not divisible by ngrid")
    ngalx = int(nx // ngrid)
    ngaly = int(ny // ngrid)
    logger.info(
        f"Simulating exponential galaxies with g1={g1:.2f} and g2={g2:.2f}"
        )
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal = galsim.Exponential(half_light_radius=hlr).withFlux(flux).shear(g1=g1, g2=g2)
    rng = np.random.RandomState(seed=seed)
    # TODO - Build variable psf
    if fix_psf:
        psf = build_fixed_psf(fwhm=fwhm, psf_name=psf_name, logger=logger)
    gal = galsim.Convolve([gal, psf], gsparams=gsparams)
    if do_shift:
        shift_x = rng.uniform(low=-0.5, high=0.5) * scale
        shift_y = rng.uniform(low=-0.5, high=0.5) * scale
        gal.shift(shift_x, shift_y)
    gal = gal.shift(0.5 * scale, 0.5 * scale)
    gal_image = gal.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    noise_std = np.sqrt(np.sum(gal_image**2)) / s2n
    noise_std *= noise_fac
    gal_image = np.tile(gal_image, (ngaly, ngalx))
    psf_image = psf.shift(
        0.5 * scale, 0.5 * scale
        ).drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    seed_renoise = seed + 212 # Make sure the second noise has different seed
    rng_renoise = np.random.RandomState(seed_renoise)
    image_noise = rng.normal(scale=noise_std, size=gal_image.shape)
    noise_array = rng_renoise.normal(scale=noise_std, size=gal_image.shape)
    return gal_image, psf_image, noise_std, image_noise, noise_array