import galsim
import numpy as np
from .utils import setup_custom_logger, cached_descwl_catalog_read

import descwl

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
        psf = galsim.Gaussian(fwhm=fwhm).shear(g1=psf_g1, g2=psf_g2)
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0).shear(
            g1=psf_g1, g2=psf_g2
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
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal = galsim.Exponential(half_light_radius=hlr).withFlux(flux).shear(g1=g1, g2=g2)
    if fix_psf:
        psf = build_fixed_psf(field=field, psf_name=psf_name)
    else:
        psf = build_variable_psf(seed=seed, field=field, psf_name=psf_name)
    gal = galsim.Convolve([gal, psf], gsparams=gsparams)
    gal = gal.shift(0.5 * scale, 0.5 * scale)
    gal_array = gal.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
    psf_array = (
        psf.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    return gal_array, psf_array


def get_survey():
    pars = descwl.survey.Survey.get_defaults(survey_name="LSST", filter_band="r")
    pars["survey_name"] = "LSST"
    pars["filter_band"] = "r"
    pars["pixel_scale"] = 0.2
    pars["image_width"] = 64
    pars["image_height"] = 64
    survey = descwl.survey.Survey(**pars)
    return survey


def simulate_descwl(
    *,
    seed,
    ngrid,
    nstamp,
    scale,
    g1=0.0,
    g2=0.0,
    field="wide",
    psf_name="gaussian",
    fix_psf=True,
):
    ngal = nstamp * nstamp
    rng = np.random.RandomState(seed=seed)
    if fix_psf:
        psf = build_fixed_psf(field=field, psf_name=psf_name)
    else:
        psf = build_variable_psf(seed=seed, field=field, psf_name=psf_name)
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    survey = get_survey()
    builder = descwl.model.GalaxyBuilder(
        survey=survey, no_disk=False, no_bulge=False, no_agn=False, verbose_model=False
    )
    cat = cached_descwl_catalog_read()
    cat["pa_disk"] = rng.uniform(low=0.0, high=360.0, size=cat.size)
    cat["pa_bulge"] = cat["pa_disk"]
    gal_image = galsim.ImageF(ngrid * nstamp, ngrid * nstamp, scale=scale)
    gal_image.setOrigin(0, 0)
    gal0 = None
    for i in range(ngal):
        ix = i % nstamp
        iy = i // nstamp
        irot = i % 2
        igal = i // 2
        if irot == 0:
            del gal0
            gal0 = builder.from_catalog(
                cat[igal], 0, 0, survey.filter_band
            ).model.rotate(rng.uniform() * 360 * galsim.degrees)
        else:
            assert gal0 is not None
            ang = np.pi / 2 * galsim.radians
            gal0 = gal0.rotate(ang)

        gal = gal0.shear(g1=g1, g2=g2)
        gal = galsim.Convolve([psf, gal], gsparams=gsparams)
        b = galsim.BoundsI(
            ix * ngrid,
            (ix + 1) * ngrid - 1,
            iy * ngrid,
            (iy + 1) * ngrid - 1,
        )
        sub_img = gal_image[b]
        gal = gal.shift(0.5 * scale, 0.5 * scale)
        gal.drawImage(sub_img, add_to_image=True)
    psf_array = (
        psf.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    return gal_image.array, psf_array


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
    gal_type="exp",
    psf_name="gaussian",
    fix_psf=True,
):
    if gal_type not in ["exp", "descwl"]:
        raise ValueError(f"gal_type of {gal_type} not supported")
    if gal_type == "exp":
        gal_array_w, psf_array_w = simulate_exponential(
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
        )
        gal_array_w = np.tile(gal_array_w, (nstamp, nstamp))
        gal_array_d, psf_array_d = simulate_exponential(
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
        )
        gal_array_d = np.tile(gal_array_d, (nstamp, nstamp))
    else:
        gal_array_w, psf_array_w = simulate_descwl(
            seed=seed,
            ngrid=ngrid,
            nstamp=nstamp,
            scale=scale,
            g1=g1,
            g2=g2,
            field="wide",
            psf_name=psf_name,
            fix_psf=fix_psf,
        )
        gal_array_d, psf_array_d = simulate_descwl(
            seed=seed,
            ngrid=ngrid,
            nstamp=nstamp,
            scale=scale,
            g1=g1,
            g2=g2,
            field="deep",
            psf_name=psf_name,
            fix_psf=fix_psf,
        )
    return {
        "gal_w": gal_array_w,
        "psf_w": psf_array_w,
        "gal_d": gal_array_d,
        "psf_d": psf_array_d,
    }


def simulate_noise(
    *,
    seed,
    shape,
    gal_type="exp",
    s2n=1e8,
    deep_noise_frac=1.0,
    fix_noise=True,
    signal=1e8,
):
    # Noise properties
    if gal_type == "exp":
        noise_std_w = signal / s2n
        noise_std_d = noise_std_w * deep_noise_frac
    else:
        survey = get_survey()
        noise_std_w = np.sqrt(survey.mean_sky_level)
        noise_std_d = noise_std_w * deep_noise_frac
    rng = np.random.RandomState(seed=seed)
    if not fix_noise:
        scale_wide = rng.uniform(low=0.9, high=1.1)
        scale_deep = rng.uniform(low=0.9, high=1.1)
    else:
        scale_wide = 1.0
        scale_deep = 1.0
    noise_std_w *= scale_wide
    noise_std_d *= scale_deep
    image_noise_w = rng.normal(scale=noise_std_w, size=shape)
    image_noise_d = rng.normal(scale=noise_std_d, size=shape)

    return {
        "img_noise_w": image_noise_w,
        "noise_std_w": noise_std_w,
        "img_noise_d": image_noise_d,
        "noise_std_d": noise_std_d,
    }
