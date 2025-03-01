import galsim
import numpy as np
from .utils import setup_custom_logger, cached_descwl_catalog_read
import anacal

import descwl

logger = setup_custom_logger()

# TODO - Add unit tests


def build_fixed_psf(
    *,
    fwhm,
    psf_name="gaussian",
):
    if psf_name not in ["gaussian", "moffat"]:
        raise ValueError(f"{psf_name} is not supported")
    if psf_name == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm)
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0)
    return psf


def build_variable_psf(
    *,
    seed,
    fwhm,
    psf_name="gaussian",
):
    if psf_name not in ["gaussian", "moffat"]:
        raise ValueError(f"{psf_name} is not supported")
    rng = np.random.RandomState(seed)
    fwhm = rng.uniform(low=fwhm - 0.1, high=fwhm + 0.1)
    psf_g1 = rng.uniform(low=-0.02, high=0.02)
    psf_g2 = rng.uniform(low=-0.02, high=0.02)
    if psf_name == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm).shear(g1=psf_g1, g2=psf_g2)
    else:
        psf = galsim.Moffat(beta=2.5, fwhm=fwhm, trunc=0.6 * 4.0).shear(
            g1=psf_g1, g2=psf_g2
        )
    return psf


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
    fwhm=0.9,
    psf_name="gaussian",
    fix_psf=True,
):
    gsparams = galsim.GSParams(maximum_fft_size=10240)
    gal = galsim.Exponential(half_light_radius=hlr).withFlux(flux).shear(g1=g1, g2=g2)
    if fix_psf:
        psf = build_fixed_psf(fwhm=fwhm, field=field, psf_name=psf_name)
    else:
        psf = build_variable_psf(seed=seed, fwhm=fwhm, field=field, psf_name=psf_name)
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
    fwhm=0.9,
    psf_name="gaussian",
    fix_psf=True,
):
    ngal = nstamp * nstamp
    rng = np.random.RandomState(seed=seed)
    if fix_psf:
        psf = build_fixed_psf(fwhm=fwhm, field=field, psf_name=psf_name)
    else:
        psf = build_variable_psf(seed=seed, fwhm=fwhm, field=field, psf_name=psf_name)
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


def simulate_wide_deep(
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
    fwhm_w=0.9,
    fwhm_d=0.7,
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
            fwhm=fwhm_w,
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
            fwhm=fwhm_d,
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
            fwhm=fwhm_w,
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
            fwhm=fwhm_d,
            psf_name=psf_name,
            fix_psf=fix_psf,
        )
    return {
        "gal_w": gal_array_w,
        "psf_w": psf_array_w,
        "gal_d": gal_array_d,
        "psf_d": psf_array_d,
    }

def simulate_wide_deep_isolated(
    seed,
    ngrid=64,
    nstamp=100,
    scale=0.2,
    gal_type="mixed",
    fwhm_w=0.9,
    fwhm_d=0.7,
    fix_psf=True,
):
    if fix_psf:
        psf_w = galsim.Moffat(beta=2.5, fwhm=fwhm_w, trunc=0.6 * 4.0)
        psf_d = galsim.Moffat(beta=2.5, fwhm=fwhm_d, trunc=0.6 * 4.0)
    else:
        rng = np.random.RandomState(seed=seed)
        fwhm_w = rng.uniform(low=fwhm_w - 0.1, high=fwhm_w + 0.1)
        fwhm_d = rng.uniform(low=fwhm_d - 0.1, high=fwhm_d + 0.1)
        psf_g1 = rng.uniform(low=-0.02, high=0.02)
        psf_g2 = rng.uniform(low=-0.02, high=0.02)
        psf_w = galsim.Moffat(beta=2.5, fwhm=fwhm_w, trunc=0.6 * 4.0).shear(g1=psf_g1, g2=psf_g2)
        psf_d = galsim.Moffat(beta=2.5, fwhm=fwhm_d, trunc=0.6 * 4.0).shear(g1=psf_g1, g2=psf_g2)
    gal_wp = anacal.simulation.make_isolated_sim(
        seed=seed,
        scale=scale,
        psf_obj=psf_w,
        ngrid=ngrid,
        ny=nstamp*ngrid,
        nx=nstamp*ngrid,
        gname="g1-1",
        gal_type=gal_type,
        sim_method="fft",
        buff=0,
        do_shift=False,
        mag_zero=30.0
    )[0]
    gal_wm = anacal.simulation.make_isolated_sim(
        seed=seed,
        scale=scale,
        psf_obj=psf_w,
        ngrid=ngrid,
        ny=nstamp*ngrid,
        nx=nstamp*ngrid,
        gname="g1-0",
        gal_type=gal_type,
        sim_method="fft",
        buff=0,
        do_shift=False,
        mag_zero=30.0
    )[0]
    if fwhm_w == fwhm_d:
        gal_dp = gal_wp
        gal_dm = gal_wm
    else:
        gal_dp = anacal.simulation.make_isolated_sim(
            seed=seed,
            scale=scale,
            psf_obj=psf_d,
            ngrid=ngrid,
            ny=nstamp*ngrid,
            nx=nstamp*ngrid,
            gname="g1-1",
            gal_type=gal_type,
            sim_method="fft",
            buff=0,
            do_shift=False,
            mag_zero=30.0
        )[0]
        gal_dm = anacal.simulation.make_isolated_sim(
            seed=seed,
            scale=scale,
            psf_obj=psf_d,
            ngrid=ngrid,
            ny=nstamp*ngrid,
            nx=nstamp*ngrid,
            gname="g1-0",
            gal_type=gal_type,
            sim_method="fft",
            buff=0,
            do_shift=False,
            mag_zero=30.0
        )[0]
    return {
        "gal_wp": gal_wp,
        "gal_wm": gal_wm,
        "psf_w": psf_w,
        "gal_dp": gal_dp,
        "gal_dm": gal_dm,
        "psf_d": psf_d,
    }


def simulate_noise(
    *,
    seed,
    shape,
    gal_type="exp",
    deep_noise_frac=1.0,
    fix_noise=True,
    **kwargs,
):
    # Noise properties
    if gal_type == "exp":
        if "noise_std_w" in kwargs:
            noise_std_w = kwargs["noise_std_w"]
        else:
            signal = kwargs.get("signal", 1e8)
            s2n = kwargs.get("s2n", 1e8)
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
