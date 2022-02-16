import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 1000 # 1s and we discard the first 200ms


pconn_values = [0.025, 0.05, 0.075, 0.1, 0.15]

# starting from a network with 
KEYS = ['p_AffExcBG_L23Exc',
        'p_AffExcBG_PvInh',
        'p_AffExcBG_CB1Inh',
        'p_PvInh_L23Exc',
        'p_PvInh_PvInh',
        'p_CB1Inh_L23Exc',
        'p_CB1Inh_CB1Inh']

DESIRED_RATES = {'L23Exc':0.1,
                 'CB1Inh':20.,
                 'PvInh':30.}


def compute_residual_and_update_minimum(data,
                                        current_residual,
                                        DESIRED_RATES,
                                        verbose=True):
    # compute residual
    residual = 0
    for pop in DESIRED_RATES:
        residual += (data['rate_%s'%pop]-DESIRED_RATES[pop])/DESIRED_RATES[pop]

    # compute residual
    current_params = {}
    if residual<current_residual:
        current_residual = residual
        for key in KEYS:
            current_params[key] = data[key]
        if verbose:
            print(40*'--')
            print('-> update ', [data['rate_%s'%pop] for pop in DESIRED_RATES])

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

    if sys.argv[-1]=='test':
        
        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan-test.zip'
        
        ntwk.scan.run(Model,
                      KEYS[:1], [pconn_values],
                      running_sim_func,
                      parallelize=True)

    elif sys.argv[-1]=='test-analysis':

        Model2 = {'data_folder': './data/', 'zip_filename':'data/pconn-scan-test.zip'}
        Model2, PARAMS_SCAN, DATA = ntwk.scan.get(Model2)

        current_residual, current_params = compute_residual_and_update_minimum(data,
                                                                               current_residual,
                                                                               DESIRED_RATES)

        
        for data in DATA:
            print(40*'--')
            i=0
            while str(i) in data:
                pop = data[str(i)]['name']
                print(pop, data['rate_%s'%pop])
                i+=1
                
    elif sys.argv[-1]=='scan':
        # means scan
        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan.zip'
        
        ntwk.scan.run(Model,
                      KEYS, [pconn_values for k in KEYS],
                      running_sim_func,
                      parallelize=True)

    elif sys.argv[-1]=='scan-analysis':
        # means scan

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan.zip'
        
        Model2 = {'data_folder': './data/', 'zip_filename':'data/pconn-scan.zip'}
        Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2,
                                                  filenames_only=True)

        for filename in PARAMS_SCAN['filenames']:
            data = ntwk.recording.load_dict_from_hdf5(filename)
            current_residual, current_params = compute_residual_and_update_minimum(data,
                                                                                   np.inf,
                                                                                   DESIRED_RATES)
        


