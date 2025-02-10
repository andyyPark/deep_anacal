#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser
import numpy as np
from functools import partial

import anacal
from deep_anacal import deep_anacal, simulate, utils
import schwimmbad
import fitsio

case_dict = {
    1: {
        "gal_type": "sersic",
        "fix_psf": True,
        "fix_noise": True,
    },
    2: {
        "gal_type": "sersic",
        "fix_psf": True,
        "fix_noise": False,
    },
    3: {
        "gal_type": "sersic",
        "fix_psf": False,
        "fix_noise": False,
    },
    4: {
        "gal_type": "bulgedisk",
        "fix_psf": False,
        "fix_noise": False,
    },
}

img_dir = "/hildafs/projects/phy200017p/andypark/anl/deep_anacal_paper/isolated_sims"

def run(seed, case=1):
    gal_type = case_dict[case]
    fix_psf = case_dict[case]
    case_dir = os.path.join(img_dir, f"case{case}")
    if not os.path.exists(case_dir):
        os.makedirs(case_dir, exist_ok=True)
    sim = simulate.simulate_wide_deep_isolated(seed, fwhm_w=0.8, fwhm_d=0.8, gal_type=gal_type, fix_psf=fix_psf)
    fitsio.write(
        os.path.join(
            case_dir, f"gal_w_{seed}+.fits"
        ), sim["gal_wp"], clobber=True)
    fitsio.write(
        os.path.join(
            case_dir, f"gal_w_{seed}-.fits"
        ), sim["gal_wm"], clobber=True)
    if not fix_psf:
        fitsio.write(
            os.path.join(
                case_dir, f"gal_d_{seed}+.fits"
            ), sim["gal_dp"], clobber=True)
        fitsio.write(
            os.path.join(
                case_dir, f"gal_d_{seed}-.fits"
            ), sim["gal_dm"], clobber=True)
        fitsio.write(
            os.path.join(
                case_dir, f"psf_w_{seed}.fits"
            ), sim["psf_w"], clobber=True)
        fitsio.write(
            os.path.join(
                case_dir, f"psf_d_{seed}.fits"
            ), sim["psf_d"], clobber=True)
    else:
        fitsio.write(
            os.path.join(
                case_dir, "psf_w.fits"
            ), sim["psf_w"]
        )
        fitsio.write(
            os.path.join(
                case_dir, "psf_d.fits"
            ), sim["psf_d"]
        )
    return


def main():
    parser = ArgumentParser(description="simulate isolated galaxy images")
    parser.add_argument(
        "--case",
        default=1,
        type=int,
        help="case to run",
    )
    parser.add_argument(
        "--nsims",
        default=1000,
        type=int,
        help="number of simulations to run",
    )
    cmd_args = parser.parse_args()
    case = cmd_args.case
    nsims = cmd_args.nsims
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    worker = partial(run, case=case)
    _ = pool.map(worker, np.arange(nsims))
    pool.close()
    sys.exit(0)
    return

if __name__ == "__main__":
    main()