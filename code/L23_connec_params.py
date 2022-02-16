import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 1000 # 1s and we discard the first 200ms

def running_sim_func(Model, a=0):
    run_single_sim(Model,
                   REC_POPS=['L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG'],
                   build_pops_args=dict(with_raster=False,
                                        with_Vm=0,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])


pconn_values = [0.025, 0.05, 0.075, 0.1, 0.15]

# starting from a network with 
KEYS = ['p_AffExcBG_L23Exc',
        'p_AffExcBG_PvInh',
        'p_AffExcBG_CB1Inh',
        'p_PvInh_L23Exc',
        'p_PvInh_PvInh',
        'p_CB1Inh_L23Exc',
        'p_CB1Inh_CB1Inh']

if __name__=='__main__':

    if sys.argv[-1]=='test':
        
        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan-test.zip'
        
        ntwk.scan.run(Model,
                      KEYS[:1], [pconn_values],
                      running_sim_func,
                      parallelize=True)

    if sys.argv[-1]=='test-analysis':

        Model2 = {'data_folder': './data/', 'zip_filename':'data/pconn-scan-test.zip'}
        Model2, PARAMS_SCAN, DATA = ntwk.scan.get(Model2)
        
        
    else:
        # means scan
        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/pconn-scan.zip'
        
        ntwk.scan.run(Model,
                      KEYS, [pconn_values for k in KEYS],
                      running_sim_func,
                      parallelize=True)


            
