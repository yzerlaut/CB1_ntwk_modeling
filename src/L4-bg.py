import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 1000 # 1s and we discard the first 200ms

if sys.argv[-1]=='bg-analysis':
    # means scan

    Model2 = {'data_folder': './data/', 'zip_filename':'data/L4-bg-scan.zip'}
    Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2, filenames_only=True)

    current_residual, current_params = np.inf, {}
    for filename in PARAMS_SCAN['FILENAMES']:
        data = ntwk.recording.load_dict_from_hdf5(filename)
        print('- p_AffExcBG_L4Exc=%.3f: L4=%.2fHz, L23=%.2fHz' % (data['p_AffExcBG_L4Exc'],
                                                                  data['rate_L4Exc'],data['rate_L23Exc']))

elif 'bg' in sys.argv[-1]:
    # means scan, possible options:
    # "bg", "bg-fix-missing", "bg-with-repeat" or "bg-fix-missing-with-repeat"
    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/L4-bg-scan.zip'

    Nscan = 2*8

    def running_sim_func(Model):
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG'],
                       build_pops_args=dict(with_raster=False,
                                            with_Vm=0,
                                            with_pop_act=True,
                                            verbose=False),
                       filename=Model['filename'])

    pconn_values = np.linspace(0.01, 0.1, Nscan)
    
    ntwk.scan.run(Model, ['p_AffExcBG_L4Exc'], [pconn_values], running_sim_func,
                  fix_missing_only=('fix-missing' in sys.argv[-1]),
                  parallelize=True)

    if 'with-repeat' in sys.argv[-1]:
        # "scan-with-repeat" or "scan-fix-missing-with-repeat"
        for i in range(5):
            ntwk.scan.run(Model, ['p_AffExcBG_L4Exc'], [pconn_values], running_sim_func,
                          fix_missing_only=True, parallelize=True)

else:
    print('need args')    
