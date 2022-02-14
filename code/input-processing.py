import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
# from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
# from analyz.processing.signanalysis import gaussian_smoothing
import ntwk

#####################################
########## adding input props #######
#####################################

Model['event_amplitude'] = 5
Model['event_width'] = 150
Model['event_times'] = [9000, 13000]
Model['tstop'] = 15000

def running_sim_func(Model, a=0, NVm=3):
    run_single_sim(Model,
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=NVm,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])
    

if __name__=='__main__':

    
    if sys.argv[-1]=='input-bg-scan-plot':

        Model = {'data_folder': './data/', 'zip_filename':'data/input-bg-space-scan.zip'}
        Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model)
        
    elif sys.argv[-1]=='input-bg-scan':

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/input-bg-space-scan.zip'

        Nscan = 10

        ntwk.scan.run(Model,
                      ['F_AffExcBG', 'input_amplitude'],
                      [np.linspace(1, 8, Nscan), np.linspace(2, 10, Nscan)],
                      running_sim_func,
                      fix_missing_only=True,
                      parallelize=True)
        
    elif 'plot' in sys.argv[-1]:
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        if 'plot-' in sys.argv[-1]:
            CONDS = [sys.argv[-1].split('plot-')[-1]]
        else:
            CONDS = ['V1', 'V2', 'V2-CB1-KO']
            CONDS = ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']
        from plot import raw_data_fig_multiple_sim, summary_fig_multiple_sim

        fig_raw, AX2 = raw_data_fig_multiple_sim([('data/input-processing-%s.h5' % cond) for cond in CONDS],
                                                 subsampling=100, tzoom=[200,Model['tstop']])
        fig_raw.savefig('fig_raw.png')

        fig_summary, AX2 = summary_fig_multiple_sim([('data/input-processing-%s.h5' % cond) for cond in CONDS],
                                                    LABELS=CONDS)
        fig_summary.savefig('fig_summary.png')
        
    elif sys.argv[-1]=='Aff':
        
        NTWK = ntwk.build.populations(Model, ['L4Exc'],
                                      AFFERENT_POPULATIONS=['AffExcTV'],
                                      with_raster=True,
                                      with_Vm=3,
                                      with_pop_act=True,
                                      verbose=False)

        ntwk.build.recurrent_connections(NTWK, SEED=Model['SEED']*(Model['SEED']+1),
                                         verbose=False)

        t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
        
        # constant background input
        ntwk.stim.construct_feedforward_input(NTWK, 'L4Exc', 'AffExcTV',
                                              t_array, build_Faff_array(Model)[1], #Model['F_L4Exc']+0.*t_array,
                                              verbose=False,
                                              SEED=1+2*Model['SEED']*(Model['SEED']+1))
        ntwk.build.initialize_to_rest(NTWK)
        network_sim = ntwk.collect_and_run(NTWK, verbose=True)

        ntwk.recording.write_as_hdf5(NTWK, filename='data/input-processing-Aff.h5')
    
    elif sys.argv[-1] in ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']:

        Model = update_model(Model, sys.argv[-1])
        
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/input-processing-%s.h5' % sys.argv[-1])
        
    elif sys.argv[-1]=='run':
        import subprocess
        for model in ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']:
            subprocess.Popen('python code/input-processing.py %s &' % model,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        

        
    else:
        print('need args')

