import numpy as np
import galsim
import anacal
from deep_anacal import utils


def simulate_gal_psf(ngrid, scale):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )

    gal_obj = galsim.Gaussian(fwhm=0.6).shear(g1=0.02, g2=0.0)
    gal_array = (
        gal_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )

    # force detection at center
    coords = np.array(
        [(ngrid / 2.0, ngrid / 2.0, True, 0)],
        dtype=[
            ("y", "f8"),
            ("x", "f8"),
            ("is_peak", "i4"),
            ("mask_value", "i4"),
        ],
    )
    return gal_array, psf_array, coords


def test_weight_force():
    ngrid = 64
    scale = 0.2
    gal_array, psf_array, coords = simulate_gal_psf(
        ngrid,
        scale,
    )

    fpfs_config = anacal.fpfs.FpfsConfig(sigma_arcsec=0.52)
    cat = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=scale,
        noise_variance=1e-3,
        noise_array=None,
        detection=coords,
    )

    e, w = utils.get_e_w(acat=cat, component=1, force_detection=True)
    assert w == 1.0

    return
