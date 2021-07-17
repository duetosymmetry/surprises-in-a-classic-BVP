#!/usr/bin/env python

"""Generates a plot of the bifurcation point"""

import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from bvp_common import *

if __name__ == "__main__":

    ############################################################
    ## Maps for the three cases
    which_keys = ['B0', 'B1', 'M']
    dps      = {'B0': 30,  'B1': 30,  'M': 25}
    eps_min  = {'B0': .01, 'B1': .02, 'M': .01}
    eps_num  = {'B0': 50,  'B1': 25,  'M': 25}
    ana_func = {'B0': y_prime_B0_asymp,
                'B1': one_minus_y_prime_B1_asymp,
                'M':  one_minus_y_prime_M_asymp}
    num_func = {'B0': y_prime_B0_num,
                'B1': y_prime_B1_num,
                'M':  y_prime_M_num}
    subtract_from_1 = {'B0': False, 'B1': True, 'M': True}

    ############################################################
    ## Args

    parser = argparse.ArgumentParser()

    parser.add_argument('--which', '-w',
                        choices=which_keys,
                        required=True)
    parser.add_argument('--outfile', '-o',
                        type=Path,
                        required=True,
                        help='Name of file to write (default: %(default)s).')

    args = parser.parse_args()
    which = args.which

    ############################################################
    ## Work

    # If you want more digits, change this number:
    mp.dps = dps[which]
    eps_to_plot = np.geomspace( eps_min[which], .2, num=eps_num[which])
    y_p_asymp = np.array(list(map( ana_func[which],
                                   eps_to_plot)),
                         dtype=np.float64)

    pool = Pool()
    y_p_num = np.array( pool.map(num_func[which], eps_to_plot) )
    if (subtract_from_1[which]):
        y_p_num = 1-y_p_num

    ############################################################
    ## Dump

    with args.outfile.open('wb') as pkl_file:
        pickle.dump((eps_to_plot, y_p_num, y_p_asymp), pkl_file)
