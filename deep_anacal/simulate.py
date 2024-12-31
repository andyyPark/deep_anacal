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
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        fwhm=0.6,
        psf_name="gaussian",
        fix_psf=True,
        return_noise=True,
        noise_std=0,
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
    gal = galsim.Exponential(half_light_radius=hlr).shear(g1=g1, g2=g2)
    # TODO - Build variable psf
    if fix_psf:
        psf = build_fixed_psf(fwhm=fwhm, psf_name=psf_name, logger=logger)
    psf_image = psf.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    gal = galsim.Convolve([gal, psf], gsparams=gsparams)
    # TODO - Allow shift in the center of the pixel
    gal = gal.shift(0.5 * scale, 0.5 * scale)
    gal_image = gal.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    gal_image = np.tile(gal_image, (ngaly, ngalx))
    if return_noise:
        image_noise, renoise_image = simulate_noise(
            seed=seed,
            image_shape=gal_image.shape,
            noise_std=noise_std,
        )
        return gal_image, psf_image, image_noise, renoise_image
    else:
        gal_image, psf_image


def simulate_noise(*, seed, image_shape, noise_std):
    rng = np.random.RandomState(seed)
    image_noise = rng.normal(size=image_shape, scale=noise_std)
    renoise_noise = rng.normal(size=image_shape, scale=noise_std)
    return image_noise, renoise_noise