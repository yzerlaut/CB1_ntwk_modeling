import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 700 # 0.8s and we discard the first 200ms


# starting from a network with 
KEYS = ['p_AffExcBG_L23Exc',
        'p_AffExcBG_PvInh',
        'p_AffExcBG_CB1Inh',
        'p_PvInh_PvInh',
        'p_CB1Inh_CB1Inh']

pconn_values = [0.025, 0.05, 0.075, 0.1]
VALUES = [pconn_values,
          pconn_values,
          pconn_values,
          pconn_values,
          pconn_values]
          

DESIRED_RATES = {'L23Exc':0.5,
                 'CB1Inh':15.,
                 'PvInh':20.}


def compute_residual_and_update_minimum(data,
                                        current_residual,
                                        current_params,
                                        DESIRED_RATES,
                                        verbose=True):
    # compute residual
    residual = 0.
    for pop in DESIRED_RATES:
        residual += float(np.abs((data['rate_%s'%pop]-DESIRED_RATES[pop])/DESIRED_RATES[pop]))

    # compute residual
    if residual<current_residual:
        current_residual, current_params = residual, {}
        for key in KEYS:
            current_params[key] = float(data[key])
        if verbose:
            print(40*'--')
            print('   residual', current_residual)
            print('-> update ', [data['rate_%s'%pop] for pop in DESIRED_RATES])
            print(current_params)

    return current_residual, current_params


def running_sim_func(Model, a=0):
    run_single_sim(Model,
                   REC_POPS=['L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG'],
                   build_pops_args=dict(with_raster=False,
                                        with_Vm=0,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])


if __name__=='__main__':

    if sys.argv[-1] in ['V1', 'V2']:

        Model = update_model(Model, sys.argv[-1])
        
        run_single_sim(Model,
                       REC_POPS=['L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG'],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/L23-circuit-%s.h5' % sys.argv[-1])
        
    elif 'plot' in sys.argv[-1]:
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        if 'plot-' in sys.argv[-1]:
            CONDS = [sys.argv[-1].split('plot-')[-1]]
        else:
            CONDS = ['V1', 'V2']
            CONDS = ['V1', 'V2']
            
        from plot import raw_data_fig_multiple_sim, summary_fig_multiple_sim

        fig_raw, AX2 = raw_data_fig_multiple_sim([('data/L23-circuit-%s.h5' % cond) for cond in CONDS],
                                                 subsampling=10, tzoom=[200,Model['tstop']], verbose=True)
        fig_raw.savefig('fig.png')

        
    elif sys.argv[-1]=='scan-analysis':
        # means scan

        Model2 = {'data_folder': './data/', 'zip_filename':'data/pconn-scan.zip'}
        Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2,
                                               filenames_only=True)

        current_residual, current_params = np.inf, {}
        for filename in PARAMS_SCAN['FILENAMES']:
            data = ntwk.recording.load_dict_from_hdf5(filename)
            current_residual, current_params = compute_residual_and_update_minimum(data,
                                                                                   current_residual,
                                                                                   current_params,
                                                                                   DESIRED_RATES)
    elif 'scan' in sys.argv[-1]:
        # means scan, possible options:
        # "scan", "scan-fix-missing", "scan-with-repeat" or "scan-fix-missing-with-repeat"
        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan.zip'
        
        ntwk.scan.run(Model, KEYS, VALUES, running_sim_func,
                      fix_missing_only=('fix-missing' in sys.argv[-1]),
                      parallelize=True)

        if 'with-repeat' in sys.argv[-1]:
            # "scan-with-repeat" or "scan-fix-missing-with-repeat"
            for i in range(5):
                ntwk.scan.run(Model, KEYS, VALUES, running_sim_func,
                              fix_missing_only=True, parallelize=True)
                


    else:
        print('need args')

