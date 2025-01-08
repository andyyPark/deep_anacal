#!/usr/bin/env python
import os
import gc
import sys
from argparse import ArgumentParser
from astropy.io import fits
import numpy as np
import schwimmbad
from deep_anacal import simulate


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


class Worker(object):
    def __init__(self, case_name, exp_type, scale=0.2, stamp=100):
        self.case_name = case_name
        self.case_number = int(self.case_name[:-1])
        if self.case_number < 5:
            self.psf_name = "gaussian"
        else:
            self.psf_name = "moffat"
        self.img_dir = os.path.join(os.getcwd(), case_name)
        assert exp_type in ["wide", "deep"]
        self.exp_type = exp_type
        if self.exp_type == "deep":
            self.noise_fac = 1 / np.sqrt(10)
            self.fwhm = 0.7
        else:
            self.noise_fac = 1.0
            self.fwhm = 0.9
        self.scale = scale
        self.psf_array = simulate.build_fixed_psf(
            fwhm=self.fwhm, psf_name=self.psf_name
        )
        self.stamp = stamp
        self.ngrid = 64
        self.nx = self.stamp * self.ngrid
        self.ny = self.stamp * self.ngrid
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)
        psf_fname = os.path.join(self.img_dir, f"{self.exp_type}_psf.fits")
        fits.writeto(psf_fname, self.psf_array, overwrite=True)

    def run(self, ifield):
        print(f"Simulating for field: {ifield}")
        # TODO - Right now only supports exponential galaxy
        gal_array, psf_array, noise_std, img_noise, noise_array = (
            simulate.simulate_exponential(
                seed=20 * ifield + self.case_number,
                g1=0.02,
                ngrid=self.ngrid,
                nx=self.nx,
                ny=self.ny,
                scale=self.scale,
                fwhm=self.fwhm,
                psf_name=self.psf_name,
                s2n=19,
                noise_fac=self.noise_fac,
            )
        )
        gal_fname = os.path.join(self.img_dir, f"{self.exp_type}_gal_{ifield}.fits")
        img_noise_fname = os.path.join(
            self.img_dir, f"{self.exp_type}_imgnoise_{ifield}.fits"
        )
        renoise_fname = os.path.join(
            self.img_dir, f"{self.exp_type}_renoise_{ifield}.fits"
        )
        fits.writeto(gal_fname, gal_array, overwrite=True)
        fits.writeto(img_noise_fname, img_noise, overwrite=True)
        fits.writeto(renoise_fname, noise_array, overwrite=True)
        del gal_array, psf_array, noise_std, img_onise, noise_array
        gc.collect()


def run(pool, case_name, exp_type, min_id, max_id):
    input_list = list(range(min_id, max_id))
    worker = Worker(case_name, exp_type)
    for _ in pool.map(worker.run, input_list):
        pass
    pool.close()
    sys.exit(0)
    return


def main():
    parser = ArgumentParser(description="simulate isolated galaxy images")
    parser.add_argument(
        "--case_name",
        type=str,
        help="The case to run, e.g. case_1",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        help="The exposure type, e.g. wide or deep",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="minimum simulation id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=200,
        type=int,
        help="maximum simulation id number, e.g. 100",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    cmd_args = parser.parse_args()
    case_name = cmd_args.case_name
    exp_type = cmd_args.exp_type
    min_id = cmd_args.min_id
    max_id = cmd_args.max_id
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    ncores = get_processor_count(pool, cmd_args)
    run(
        pool=pool,
        case_name=case_name,
        exp_type=exp_type,
        min_id=min_id,
        max_id=max_id,
    )
    return


if __name__ == "__main__":
    main()
