#!/usr/bin/env python
import numpy as np
import joblib
import multiprocessing

import anacal
from deep_anacal import deep_anacal, simulate, utils

ngrid = 64
indx = [ngrid//2]
indy = [ngrid//2]
ns = len(indx) * len(indy)
inds = np.meshgrid(indy, indx, indexing="ij")
yx = np.vstack([np.ravel(_) for _ in inds])
dtype = np.dtype(
    [
        ("y", np.int32),
        ("x", np.int32),
        ("is_peak", np.int32),
        ("mask_value", np.int32),
    ]
)
detection = np.empty(ns, dtype=dtype)
detection["y"] = yx[0]
detection["x"] = yx[1]
detection["is_peak"] = np.ones(ns)
detection["mask_value"] = np.zeros(ns)

def run_sim_pair(seed, s2n, deep_noise_frac):
    scale = 0.2
    fpfs_config = anacal.fpfs.FpfsConfig(sigma_arcsec=0.52)
    sim_p = simulate.sim_wide_deep(
        seed=seed,
        ngrid=ngrid,
        scale=scale,
        g1=0.02,
        fwhm_w=0.9,
        fwhm_d=0.7,
        s2n=s2n,
        deep_noise_frac=deep_noise_frac,
    )
    ep_wide = deep_anacal.measure_eR(
        gal_array=sim_p["gal_w"]+sim_p["img_noise_w"], psf_array=sim_p["psf_w"],
        noise_array=sim_p["img_noise_d"]+sim_p["noise_array_d"],
        noise_variance=0.5*(sim_p["noise_std_w"]**2+2*sim_p["noise_std_d"]**2),
        fpfs_config=fpfs_config, scale=scale, force_detection=True, detection=detection, component=1
    )["e"][0]
    Rp_deep = deep_anacal.measure_eR(
        gal_array=sim_p["gal_d"]+sim_p["img_noise_d"], psf_array=sim_p["psf_d"],
        noise_array=sim_p["noise_array_d"]+sim_p["img_noise_w"],
        noise_variance=0.5*(sim_p["noise_std_w"]**2+2*sim_p["noise_std_d"]**2),
        fpfs_config=fpfs_config, scale=scale, force_detection=True, detection=detection, component=1
    )["R"][0]
    sim_m = simulate.sim_wide_deep(
        seed=seed,
        ngrid=ngrid,
        scale=scale,
        g1=-0.02,
        fwhm_w=0.9,
        fwhm_d=0.7,
        s2n=s2n,
        deep_noise_frac=deep_noise_frac,
    )
    em_wide = deep_anacal.measure_eR(
        gal_array=sim_m["gal_w"]+sim_m["img_noise_w"], psf_array=sim_m["psf_w"],
        noise_array=sim_m["img_noise_d"]+sim_m["noise_array_d"],
        noise_variance=0.5*(sim_m["noise_std_w"]**2+2*sim_m["noise_std_d"]**2),
        fpfs_config=fpfs_config, scale=scale, force_detection=True, detection=detection, component=1
    )["e"][0]
    Rm_deep = deep_anacal.measure_eR(
        gal_array=sim_m["gal_d"]+sim_m["img_noise_d"], psf_array=sim_m["psf_d"],
        noise_array=sim_m["noise_array_d"]+sim_m["img_noise_w"],
        noise_variance=0.5*(sim_m["noise_std_w"]**2+2*sim_m["noise_std_d"]**2),
        fpfs_config=fpfs_config, scale=scale, force_detection=True, detection=detection, component=1
    )["R"][0]
    return (ep_wide, em_wide, Rp_deep, Rm_deep)

def main():
    nsims = 1000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size + 1
    s2n = 1e8
    deep_noise_frac = 1 / np.sqrt(10) # deep image variance is 10x smaller
    nsims = nchunks * chunk_size
    rng = np.random.RandomState(seed=12)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    num1 = []
    num2 = []
    denom = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc:loc + chunk_size]
        jobs = [
            joblib.delayed(run_sim_pair)(seed, s2n, deep_noise_frac)
            for seed in _seeds
        ]
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
        for res in outputs:
            num1.append(res[0] - res[1])
            num2.append(res[0] + res[1])
            denom.append(res[2] + res[3])
        res = np.vstack([num1, num2, denom]).T
        m, merr, c, cerr = utils.estimate_m_and_c(res=res, true_shear=0.02)
        print("# of sims:", len(num1), flush=True)
        print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
        print("c: %f +/- %f [1e-4, 3-sigma]" % (c/1e-4, 3*cerr/1e-4), flush=True)

if __name__ == "__main__":
    main()