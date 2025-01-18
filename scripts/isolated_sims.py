#!/usr/bin/env python
import numpy as np
import joblib
import multiprocessing

import anacal
from deep_anacal import deep_anacal, simulate, utils

ngrid = 64
def run_sim_pair(seed, nstamp, s2n, deep_noise_frac):
    scale = 0.2
    fpfs_config = anacal.fpfs.FpfsConfig(sigma_arcsec=0.52)
    detection = utils.force_detection(ngrid=ngrid, nstamp=nstamp)
    sim_p = simulate.sim_wide_deep(
        seed=seed,
        ngrid=ngrid,
        nstamp=nstamp,
        scale=scale,
        g1=0.02,
        fwhm_w=0.9,
        fwhm_d=0.7,
        s2n=s2n,
        deep_noise_frac=deep_noise_frac,
    )
    wide_cat_p, deep_cat_p = deep_anacal.match_noise(
        seed=seed,
        scale=scale,
        fpfs_config=fpfs_config,
        gal_array_w=sim_p["gal_w"]+sim_p["img_noise_w"],
        gal_array_d=sim_p["gal_d"]+sim_p["img_noise_d"],
        psf_array_w=sim_p["psf_w"],
        psf_array_d=sim_p["psf_d"],
        noise_var_w=sim_p["noise_std_w"]**2,
        noise_var_d=sim_p["noise_std_d"]**2,
        detection=detection
    )
    res_p = deep_anacal.get_e_and_R(wide_cat=wide_cat_p, deep_cat=deep_cat_p, force_detection=True)
    ep_wide = res_p["e"]
    Rp_deep = res_p["R"]

    sim_m = simulate.sim_wide_deep(
        seed=seed,
        ngrid=ngrid,
        nstamp=nstamp,
        scale=scale,
        g1=-0.02,
        fwhm_w=0.9,
        fwhm_d=0.7,
        s2n=s2n,
        deep_noise_frac=deep_noise_frac,
    )
    wide_cat_m, deep_cat_m = deep_anacal.match_noise(
        seed=seed,
        scale=scale,
        fpfs_config=fpfs_config,
        gal_array_w=sim_m["gal_w"]+sim_m["img_noise_w"],
        gal_array_d=sim_m["gal_d"]+sim_m["img_noise_d"],
        psf_array_w=sim_m["psf_w"],
        psf_array_d=sim_m["psf_d"],
        noise_var_w=sim_m["noise_std_w"]**2,
        noise_var_d=sim_m["noise_std_d"]**2,
        detection=detection
    )
    res_m = deep_anacal.get_e_and_R(wide_cat=wide_cat_m, deep_cat=deep_cat_m, force_detection=True)
    em_wide = res_m["e"]
    Rm_deep = res_m["R"]
    return (ep_wide, em_wide, Rp_deep, Rm_deep)

def main():
    nsims = 5000
    chunk_size = multiprocessing.cpu_count()
    nchunks = nsims // chunk_size + 1
    s2n = 19
    nstamp = 100
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
            joblib.delayed(run_sim_pair)(seed, nstamp, s2n, deep_noise_frac)
            for seed in _seeds
        ]
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
        for res in outputs:
            num1.append(res[0].sum() - res[1].sum())
            num2.append(res[0].sum() + res[1].sum())
            denom.append(res[2].sum() + res[3].sum())
        res = np.vstack([num1, num2, denom]).T
        m, merr, c, cerr = utils.estimate_m_and_c(res=res, true_shear=0.02)
        print("# of sims:", len(num1), flush=True)
        print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
        print("c: %f +/- %f [1e-4, 3-sigma]" % (c/1e-4, 3*cerr/1e-4), flush=True)
        loc += chunk_size

if __name__ == "__main__":
    main()