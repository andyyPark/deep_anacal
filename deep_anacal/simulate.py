import galsim
import numpy as np
from .utils import setup_custom_logger

logger = setup_custom_logger()

# TODO - Add unit tests

def build_fixed_psf(
        *,
        field="wide",
        psf_name="gaussian",
):
    check_psf_params(field, psf_name)
    fwhm = 0.9 if field == "wide" else 0.7
    if psf_name == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm)
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0)
    return psf

def build_variable_psf(
        *,
        seed,
        field="wide",
        psf_name="gaussian",
):
    check_psf_params(field, psf_name)
    rng = np.random.RandomState(seed)
    if field == "wide":
        fwhm = rng.uniform(low=0.8, high=1.0)
    else:
        fwhm = rng.uniform(low=0.6, high=0.8)
    psf_g1 = rng.uniform(low=-0.02, high=0.02)
    psf_g2 = rng.uniform(low=-0.02, high=0.02)
    if psf_name == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm).shear(
            g1=psf_g1,
            g2=psf_g2
        )
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0).shear(
            g1=psf_g1,
            g2=psf_g2
        )
    return psf

def check_psf_params(field, psf_name):
    if field not in ["wide", "deep"]:
        raise ValueError(f"{field} must be wide or deep")
    if psf_name not in ["gaussian", "moffat"]:
        raise ValueError(f"{psf_name} is not supported")
    

def simulate_exponential(
        *,
        seed,
        ngrid,
        scale,
        flux=1,
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        field="wide",
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
        psf = build_fixed_psf(field=field, psf_name=psf_name)
    else:
        psf = build_variable_psf(seed, field=field, psf_name=psf_name)
    gal = galsim.Convolve([gal, psf], gsparams=gsparams)
    gal = gal.shift(0.5 * scale, 0.5 * scale)
    gal_array = gal.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    psf_array = psf.shift(
        0.5 * scale, 0.5 * scale
        ).drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    return gal_array, psf_array


def sim_wide_deep(
        *,
        seed,
        ngrid,
        nstamp,
        scale,
        flux=1,
        g1=0.0,
        g2=0.0,
        hlr=0.5,
        gal_type='exp',
        psf_name="gaussian",
        fwhm_w=0.9,
        fwhm_d=0.7,
        fix_psf=True,
        s2n=1e8,
        deep_noise_frac=1.0,
        fix_noise=True):
    assert fwhm_w >= fwhm_d, "deep field usually has lower fwhm than wide field"
    # TODO - Add WLDeblend galaxy
    if gal_type == 'exp':
        make_sim = simulate_exponential
    gal_array_w, psf_array_w = make_sim(
        seed=seed,
        ngrid=ngrid,
        scale=scale,
        flux=flux,
        g1=g1,
        g2=g2,
        hlr=hlr,
        field="wide",
        psf_name=psf_name,
        fix_psf=fix_psf,
        fwhm=fwhm_w,
    )
    gal_array_d, psf_array_d = make_sim(
        seed=seed,
        ngrid=ngrid,
        scale=scale,
        flux=flux,
        g1=g1,
        g2=g2,
        hlr=hlr,
        field="deep",
        psf_name=psf_name,
        fix_psf=fix_psf,
        fwhm=fwhm_d,
    )
    noise_std_w = np.sqrt(np.sum(gal_array_w**2)) / s2n
    noise_std_d = noise_std_w * deep_noise_frac
    gal_array_w = np.tile(gal_array_w, (nstamp, nstamp))
    gal_array_d = np.tile(gal_array_d, (nstamp, nstamp))
    rng = np.random.RandomState(seed=seed)
    if not fix_noise:
        scale_wide = rng.uniform(low=0.9, high=1.1)
        scale_deep = rng.uniform(low=0.9, high=1.1)
    else:
        scale_wide = 1.0
        scale_deep = 1.0
    noise_std_w *= scale_wide
    noise_std_d *= scale_deep
    image_noise_w = rng.normal(
        scale=noise_std_w,
        size=(gal_array_w.shape)
        )
    image_noise_d = rng.normal(
        scale=noise_std_d,
        size=(gal_array_d.shape)
        )

    return {
        "gal_w": gal_array_w,
        "img_noise_w": image_noise_w,
        "psf_w": psf_array_w,
        "noise_std_w": noise_std_w,
        "gal_d": gal_array_d,
        "img_noise_d": image_noise_d,
        "psf_d": psf_array_d,
        "noise_std_d": noise_std_d,
    }




