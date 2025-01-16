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
        seed,
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
        f"Simulating exponential galaxies with g1={g1:.2f} and g2={g2:.2f}"
        )
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal = galsim.Exponential(half_light_radius=hlr).withFlux(flux).shear(g1=g1, g2=g2)
    rng = np.random.RandomState(seed=seed)
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