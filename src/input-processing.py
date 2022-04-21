import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
# from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
# from analyz.processing.signanalysis import gaussian_smoothing
import ntwk

from plot import raw_data_fig_multiple_sim_with_zoom, summary_fig_multiple_sim

#####################################
########## adding input props #######
#####################################
def set_events(Model, seed=None, nmax=200):
    
    if seed is None:
        np.random.seed(Model['seed']+1)
    else:
        np.random.seed(seed)
        
    times = 4e3+np.cumsum(Model['event_width']+np.random.exponential(1e3/Model['event_freq'], size=nmax))
    events = np.random.uniform(0, Model['event_max_level'], size=nmax)
    Model['event_amplitudes'] = events[times<Model['tstop']]
    Model['event_times'] = times[times<Model['tstop']]

Model['tstop'] = 30000
Model['event_width'] = 200
Model['event_freq'] = 1 # Hz 
Model['event_max_level'] = 4.5 # Hz 
set_events(Model, seed=4)

def running_sim_func(Model, a=0,
                     NVm=3, 
                     nmax=200):
    """
    
    """

    # update model according to config
    if 'Model-key' in Model:
        Model = update_model(Model, Model['Model-key'])
    # set the input
    set_events(Model, seed=Model['input-seed'])
    # run
    run_single_sim(Model,
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=NVm,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])
    

if sys.argv[-1]=='input-bg-scan-plot':

    Model = {'data_folder': './data/', 'zip_filename':'data/input-bg-space-scan.zip'}
    Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model)

elif sys.argv[-1]=='input-bg-scan':

    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/input-bg-space-scan.zip'

    Nscan = 10

    ntwk.scan.run(Model,
                  ['F_AffExcBG', 'input_amplitude'],
                  [np.linspace(1, 8, Nscan),
                   np.linspace(2, 10, Nscan)],
                  running_sim_func,
                  fix_missing_only=True,
                  parallelize=True)
    

elif sys.argv[-1]=='seed-input-scan':


    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/seed-input-scan.zip'

    ntwk.scan.run(Model,
                  ['input-seed',
                   'event_max_level',
                   'Model-key'],
                  [np.arange(20, 40),
                   np.linspace(3, 5, 7),
                   ['V1', 'V2', 'V2-CB1-KO']],
                  running_sim_func,
                  fix_missing_only=True,
                  parallelize=True)

elif sys.argv[-1]=='seed-input-analysis':

    
    Model = {'data_folder': './data/', 'zip_filename':'data/seed-input-scan.zip'}
    Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model)
    
    FILES = {'V1':[], 'V2':[], 'V2-CB1-KO':[], 'filename':[]}
    
    for filename in PARAMS_SCAN['FILENAMES']:
        for m in ['V1', 'V2', 'V2-CB1-KO']:
            if filename.split('Model-key_')[1].split('_')[0]==m:
                FILES[m].append(filename)
                if m=='V1':
                    FILES['filename'].append(filename.split('Model-key_')[0].split(os.path.sep)[-1])

    for i in (range(len(FILES['V1']))):
        print(i+1, '/', len(FILES['V1']))
        fig_raw, AX = raw_data_fig_multiple_sim_with_zoom([FILES[m][i] for m in ['V1', 'V2', 'V2-CB1-KO']],
                                                          tzoom=[200,Model['tstop']],
                                                          tzoom2=[1000,1500],
                                                          raster_subsampling=10,
                                                          min_pop_act_for_log=0.1)
        fig_raw.suptitle(FILES['filename'][i])
        fig_raw.savefig('doc/all/full_dynamics_raw_%i.png'%i)
        fig_summary, AX2 = summary_fig_multiple_sim([FILES[m][i] for m in ['V1', 'V2', 'V2-CB1-KO']],
                                                    LABELS=['V1', 'V2', 'V2-CB1-KO'],
                                                    sttc_lim=[0.009, 0.2])
        fig_summary.suptitle(FILES['filename'][i])
        fig_summary.savefig('doc/all/full_dynamics_summary_%i.png'%i)


    
elif 'plot' in sys.argv[-1]:
    
    # ######################
    # ## ----- Plot ----- ##
    # ######################

    if 'plot-' in sys.argv[-1]:
        CONDS = [sys.argv[-1].split('plot-')[-1]]
    else:
        CONDS = ['V1', 'V2', 'V2-CB1-KO']
        # CONDS = ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']

    FILES = [('data/input-processing-%s.h5' % cond) for cond in CONDS]
    fig_raw, AX = raw_data_fig_multiple_sim_with_zoom(FILES,
                                                      tzoom=[200,Model['tstop']],
                                                      tzoom2=[1000,1500],
                                                      raster_subsampling=10,
                                                      min_pop_act_for_log=0.1)
    fig_raw.savefig('doc/full_dynamics_raw.png')

    fig_summary, AX2 = summary_fig_multiple_sim(FILES,
                                                LABELS=CONDS,
                                                sttc_lim=[0.001, 0.2])
    fig_summary.savefig('doc/full_dynamics_summary.png')
    # ge.save_on_desktop(fig_summary, 'fig.svg')

    ge.show()
    
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

