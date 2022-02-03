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

Model['event_amplitude'] = 6
Model['event_width'] = 150
Model['event_times'] = [3000, 9000]
Model['tstop'] = 10000

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

    
    if sys.argv[-1]=='main-scan':

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/main-space-scan.zip'
        
        Nscan = 3

        # variations of CB1-signalling level
        psyn0 = Model['psyn_CB1Inh_L23Exc']
        psyn_variations = np.linspace(0.1, 1, Nscan)*Model['psyn_CB1Inh_L23Exc']
        pconn = Model['p_CB1Inh_L4Exc']*np.linspace(1, 5, Nscan)
        
        ntwk.scan.run(Model,
                      ['psyn_CB1Inh_L23Exc', 'p_CB1Inh_L4Exc'],
                      [psyn_variations, pconn],
                      running_sim_func,
                      parallelize=True)

    elif sys.argv[-1]=='main-scan-plot':

        Model = {'data_folder': './data/', 'zip_filename':'data/main-space-scan.zip'}
        Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model)

    elif sys.argv[-1]=='input-bg-scan-plot':

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

        fig, AX = ge.figure(axes=(2,1), figsize=(.8,1.), hspace=2., wspace=3., bottom=1.5, top=0.3)

        if 'plot-' in sys.argv[-1]:
            CONDS = [sys.argv[-1].split('plot-')[-1]]
        else:
            CONDS = ['V1', 'V2', 'V2-CB1-KO']

        from plot import raw_data_fig_multiple_sim

        fig2, AX2 = raw_data_fig_multiple_sim([('data/input-processing-%s.h5' % cond) for cond in CONDS],
                                              subsampling=20, tzoom=[200,1e4])

        sumup = {'rate':[], 'sttc':[]}
        for i, cond in enumerate(CONDS):

            if os.path.isfile('data/input-processing-%s.h5' % cond):
                data = ntwk.recording.load_dict_from_hdf5('data/input-processing-%s.h5' % cond)
                try:
                    sumup['rate'].append(ntwk.analysis.get_mean_pop_act(data, pop='L23Exc',
                                                                        tdiscard=200))
                    AX[0].bar([i], [sumup['rate'][-1]], color='gray')
                    sumup['sttc'].append(ntwk.analysis.get_synchrony_of_spiking(data, pop='L23Exc',
                                                                        method='STTC',
                                                                        Tbin=300, Nmax_pairs=5000))
                except KeyError:
                    pass

        # np.save('sumup.npy', sumup)
        sttc = np.array(sumup['sttc'])
        AX[1].bar(range(len(sttc)), sttc, bottom=sttc.min()-.1*sttc.min(), color=ge.gray)
        
        ge.set_plot(AX[0], xticks=range(len(CONDS)), xticks_labels=CONDS, xticks_rotation=70,
                    ylabel='L23 PN rate (Hz)')
        ge.set_plot(AX[1], xticks=range(len(CONDS)), xticks_labels=CONDS, xticks_rotation=70,
                    ylabel='L23 PN STTC', yscale='log')

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
    
    elif sys.argv[-1] in ['V1', 'V2', 'V2-CB1-KO']:

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
        for model in ['V1', 'V2', 'V2-CB1-KO']:
            subprocess.Popen('python code/input-processing.py %s &' % model,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        

        
    else:
        print('need args')

