import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import run_single_sim
from Model import Model_v2 as Model
# from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
# from analyz.processing.signanalysis import gaussian_smoothing
import ntwk

Model['tstop'] = 6000

def gaussian(x, mean=0., std=1.):
    return np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std


def build_Faff_array(Model, 
                     # mean=-4, std=10, # FOR OU
                     t0=4000, sT=300, amp=3,
                     seed=100):

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    g = gaussian(t_array, 4000, sT) + gaussian(t_array, 15000, sT)
    return t_array, amp/g.max()*g
    

if __name__=='__main__':
    
    if sys.argv[-1]=='input':

        ge.plot(*build_Faff_array(Model))
        ge.show()
        
    elif 'plot' in sys.argv[-1]:
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        fig, AX = ge.figure(axes=(2,1), figsize=(.8,1.), hspace=2., wspace=3., bottom=1.5, top=0.3)

        if 'plot-' in sys.argv[-1]:
            CONDS = sys.argv[-1].split('plot-')
        else:
            CONDS = ['V1', 'V2', 'V2-CB1-KO']

        sumup = {'rate':[], 'sttc':[]}
        for i, cond in enumerate(CONDS):

            if os.path.isfile('data/model-with-L4-%s.h5' % cond):
                data = ntwk.recording.load_dict_from_hdf5('data/model-with-L4-%s.h5' % cond)
                # # ## plot
                fig1, _ = ntwk.plots.activity_plots(data,
                                                    smooth_population_activity=10.,
                                                    COLORS=[ge.tab10(i) for i in [0,2,3,1]],
                                                    raster_plot_args={'subsampling':10},
                                                    Vm_plot_args={'subsampling':10, 'clip_spikes':True})
                fig1.suptitle(cond)

                try:
                    sumup['rate'].append(ntwk.analysis.get_mean_pop_act(data, pop='L23Exc',
                                                                        tdiscard=200))

                    AX[0].bar([i], [sumup['rate'][-1]], color='gray')
                    sumup['sttc'].append(ntwk.analysis.get_synchrony_of_spiking(data, pop='L23Exc',
                                                                        method='STTC',
                                                                        Tbin=300, Nmax_pairs=2000))

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

        ntwk.recording.write_as_hdf5(NTWK, filename='data/model-with-L4-Aff.h5')
    
    elif sys.argv[-1]=='V1':

        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-L4-V1.h5')
        
    elif sys.argv[-1]=='V2':

        decrease = 50./100.
        # decreasing CB1 efficacy on PN
        Model['psyn_CB1Inh_L23Exc'] = (1-decrease)*Model['psyn_CB1Inh_L23Exc']
        # Model['Q_CB1Inh_L23Exc'] = (1-decrease)*Model['psyn_CB1Inh_L23Exc']
        
        # adding CB1 inhibition on L4
        Model['p_CB1Inh_L4Exc'] = 0.1
        Model['psyn_CB1Inh_L4Exc'] = 0.5
        Model['Q_CB1Inh_L4Exc'] = 10.
            
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-L4-V2.h5')

    elif sys.argv[-1]=='V2-CB1-KO':

        # NOT decreasing CB1 efficacy on PN
        
        # still adding CB1 inhibition on <4
        Model['p_CB1Inh_L4Exc'] = 0.1
        Model['psyn_CB1Inh_L4Exc'] = 0.5
        Model['Q_CB1Inh_L4Exc'] = 10.
            
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-L4-V2-CB1-KO.h5')

        
    else:
        print('need args')

