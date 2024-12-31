import numpy as np
import ngmix
import deep_metacal

def get_psf_obs(*, psf, dim, scale):
    cen = deep_metacal.metacal.cen_from_dim(dim)
    jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=cen,
        col=cen,
    )
    psf_obs = ngmix.Observation(
        image=psf,
        jacobian=jac,
        weight=np.ones_like(psf)
    )
    return psf_obs

def make_ngmix_obs(*, img, psf, dim, scale, nse_img, nse_level):
    cen = deep_metacal.metacal.cen_from_dim(dim)
    jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=cen,
        col=cen,
    )
    psf_obs = ngmix.Observation(
        image=psf,
        jacobian=jac,
        weight=np.ones_like(img),
    )
    obs = ngmix.Observation(
        image=img,
        jacobian=jac,
        weight=np.ones_like(img) / nse_level**2,
        noise=nse_img,
        psf=psf_obs,
    )
    return obs