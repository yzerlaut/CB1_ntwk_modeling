import os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt

from datavyz import ge
sys.path.append(os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'neural_network_dynamics'))
import ntwk

Model = {
    ## -----------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## -----------------------------------------------------------------------
    # numbers of neurons in population
    'N_Exc':4000, 'N_PvInh':800, 'N_CB1Inh':500, 'N_AffExcBG':200, 'N_AffExcTV':1000,
    # synaptic weights
    'Q_Exc_Exc':2., 'Q_Exc_PvInh':2.,  'Q_Exc_CB1Inh':2.,
    'Q_PvInh_Exc':10., 'Q_PvInh_PvInh':10., 'Q_PvInh_CB1Inh':10., 
    'Q_CB1Inh_Exc':10., 'Q_CB1Inh_PvInh':10., 'Q_CB1Inh_CB1Inh':10., 
    'Q_AffExcBG_Exc':4., 'Q_AffExcBG_PvInh':4., 'Q_AffExcBG_CB1Inh':4., 
    'Q_AffExcTV_Exc':2., 'Q_AffExcTV_PvInh':2., 'Q_AffExcTV_CB1Inh':2., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.04, 'p_Exc_PvInh':0.04, 'p_Exc_CB1Inh':0.04, 
    'p_PvInh_Exc':0.1, 'p_PvInh_PvInh':0.05, 'p_PvInh_CB1Inh':0.05, 
    'p_CB1Inh_Exc':0.2, 'p_CB1Inh_PvInh':0.2, 'p_CB1Inh_CB1Inh':0.2,
    'psyn_CB1Inh_Exc':0.5, 'psyn_CB1Inh_PvInh':0.5, 'psyn_CB1Inh_CB1Inh':0.5,  # probabilities of syn. transmission for CB1 synapses
    'p_AffExcBG_Exc':0.1, 'p_AffExcBG_PvInh':0.1, 'p_AffExcBG_CB1Inh':0.1,
    'p_AffExcTV_Exc':0.1, 'p_AffExcTV_PvInh':0.02, 'p_AffExcTV_CB1Inh':0.02,
    # afferent stimulation (0 by default)
    'F_AffExcBG':5.,
    # simulation parameters
    'dt':0.1, 'tstop': 1000., 'SEED':5, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':5.,
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_deltaV':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> PV-Inhibitory population (Inh, recurrent inhibition)
    'PvInh_Gl':10., 'PvInh_Cm':200.,'PvInh_Trefrac':5.,
    'PvInh_El':-70., 'PvInh_Vthre':-53., 'PvInh_Vreset':-70., 'PvInh_deltaV':0.,
    'PvInh_a':0., 'PvInh_b': 0., 'PvInh_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'CB1Inh_Gl':10., 'CB1Inh_Cm':200.,'CB1Inh_Trefrac':5.,
    'CB1Inh_El':-70., 'CB1Inh_Vthre':-53., 'CB1Inh_Vreset':-70., 'CB1Inh_deltaV':0.,
    'CB1Inh_a':0., 'CB1Inh_b': 0., 'CB1Inh_tauw':1e9
}


Model_v2 = {
    ## -----------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## -----------------------------------------------------------------------
    # numbers of neurons in population
    'N_Exc':4000, 'N_PvInh':800, 'N_CB1Inh':500, 'N_AffExcBG':200, 'N_AffExcTV':400, 
    # synaptic weights
    'Q_Exc_Exc':2., 'Q_Exc_PvInh':2.,  'Q_Exc_CB1Inh':2.,
    'Q_PvInh_Exc':10., 'Q_PvInh_PvInh':10.,
    'Q_CB1Inh_Exc':10.,
    'Q_AffExcBG_Exc':2., 'Q_AffExcBG_PvInh':2., 'Q_AffExcBG_CB1Inh':2., 
    'Q_AffExcTV_Exc':2., 'Q_AffExcTV_PvInh':2., 'Q_AffExcTV_CB1Inh':2., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.04, 'p_Exc_PvInh':0.04, 'p_Exc_CB1Inh':0.04, 
    'p_PvInh_Exc':0.1, 'p_PvInh_PvInh':0.05, 
    'p_CB1Inh_Exc':0.2, 
    'psyn_CB1Inh_Exc':0.5,  # probabilities of syn. transmission for CB1 synapses
    'p_AffExcBG_Exc':0.1, 'p_AffExcBG_PvInh':0.1, 'p_AffExcBG_CB1Inh':0.1,
    'p_AffExcTV_Exc':0.1, 'p_AffExcTV_PvInh':0.1, 'p_AffExcTV_CB1Inh':0.1,
    # afferent stimulation (0 by default)
    'F_AffExcBG':5.,
    # simulation parameters
    'dt':0.1, 'tstop': 1000., 'SEED':5, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'Exc_Gl':10., 'Exc_Cm':200.,'Exc_Trefrac':5.,
    'Exc_El':-70., 'Exc_Vthre':-50., 'Exc_Vreset':-70., 'Exc_deltaV':0.,
    'Exc_a':0., 'Exc_b': 0., 'Exc_tauw':1e9,
    # --> PV-Inhibitory population (Inh, recurrent inhibition)
    'PvInh_Gl':10., 'PvInh_Cm':200.,'PvInh_Trefrac':5.,
    'PvInh_El':-70., 'PvInh_Vthre':-53., 'PvInh_Vreset':-70., 'PvInh_deltaV':0.,
    'PvInh_a':0., 'PvInh_b': 0., 'PvInh_tauw':1e9,
    # --> Inhibitory population (Inh, recurrent inhibition)
    'CB1Inh_Gl':10., 'CB1Inh_Cm':200.,'CB1Inh_Trefrac':5.,
    'CB1Inh_El':-70., 'CB1Inh_Vthre':-53., 'CB1Inh_Vreset':-70., 'CB1Inh_deltaV':0.,
    'CB1Inh_a':0., 'CB1Inh_b': 0., 'CB1Inh_tauw':1e9
}

def run_single_sim(Model,
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=2,
                                        with_pop_act=True,
                                        with_synaptic_currents=False,
                                        with_synaptic_conductances=False,
                                        verbose=False),
                   REC_POPS=['Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG'],
                   specific_record_function=None, srf_args={},
                   Faff_array=None,
                   filename='CB1_ntwk_model_data.h5',
                   seed=0):

    if ('inh_exc_ratio' in Model) and ('CB1_PV_ratio' in Model):
        # adjust cell numbers
        Model['N_Inh'] = int(Model['inh_exc_ratio']*Model['N_Exc'])
        Model['N_CB1Inh'] = int(Model['CB1_PV_ratio']*Model['N_Inh'])
        Model['N_PvInh'] = Model['N_Inh']-Model['N_CB1Inh']
        # adjust proba
        for target in REC_POPS:
            Model['p_CB1Inh_%s' % target] = Model['p_PvInh_%s' % target]/Model['psyn_CB1Inh_%s' % target]

    if ('common_Vthre_Inh' in Model):
        Model['CB1Inh_Vthre'] = Model['common_Vthre_Inh']
        Model['PvInh_Vthre'] = Model['common_Vthre_Inh']

    NTWK = ntwk.build.populations(Model, REC_POPS,
                                  AFFERENT_POPULATIONS=AFF_POPS,
                                  **build_pops_args)

    ntwk.build.recurrent_connections(NTWK, SEED=seed+Model['SEED']*(Model['SEED']+1),
                                     verbose=build_pops_args['verbose'],
                                     store_connections=(specific_record_function is not None))


    if specific_record_function is not None:
        specific_record_function(NTWK, Model, **srf_args)
        
    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    if (Faff_array is not None) and (len(Faff_array)!=len(t_array)):
        print('/!\ len(Faff_array)!=len(t_array), size are %i vs %i /!\  ' % (len(Faff_array), len(t_array)))

    # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(REC_POPS): # both on excitation and inhibition
        # constant background input
        ntwk.stim.construct_feedforward_input(NTWK, tpop, AFF_POPS[0],
                                              t_array, Model['F_%s' % AFF_POPS[0]]+0.*t_array,
                                              verbose=build_pops_args['verbose'],
                                              SEED=1+2*Model['SEED']*(Model['SEED']+1))
        # time-varying input
        if len(AFF_POPS)>0 and (Faff_array is not None):
            ntwk.stim.construct_feedforward_input(NTWK, tpop, AFF_POPS[1],
                                                  t_array, Faff_array,
                                                  verbose=build_pops_args['verbose'],
                                                  SEED=1+2*Model['SEED']*(Model['SEED']+1))

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.build.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.recording.write_as_hdf5(NTWK, filename=filename)


if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        
        fig, AX = ge.figure(axes=(2,1), figsize=(.8,1.), left=1.3, wspace=2.)

        CONDS = ['V1', 'V2']
        
        for i, cond in enumerate(CONDS):

            if os.path.isfile('data/CB1_ntwk_model-%s.h5' % cond):
                data = ntwk.recording.load_dict_from_hdf5('data/CB1_ntwk_model-%s.h5' % cond)
                # # ## plot
                fig1, _ = ntwk.plots.activity_plots(data,
                                                    smooth_population_activity=10.,
                                                    COLORS=[plt.cm.tab10(i) for i in [2,3,1]],
                                                    raster_plot_args={'subsampling':1, 'ms':1},
                                                    Vm_plot_args={'subsampling':2, 'clip_spikes':True})
                
                # fig1.suptitle(cond)

                # fig1 = ntwk.plots.raster(data,
                #                             COLORS=[plt.cm.tab10(i) for i in [2,3,1]])
                
                fig2 = ntwk.plots.few_Vm_plot(data,
                                              COLORS=[plt.cm.tab10(i) for i in [2,3,1]],
                                              tzoom=[0, np.inf],
                                              clip_spikes=False,
                                              vpeak=-40, vbottom=-80, shift=20.,
                                              Tbar=50., Vbar=20.)                
                
                try:
                    AX[0].bar([i], [ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)],
                           color='gray')
                    AX[1].bar([i], [ntwk.analysis.get_mean_pop_act(data, pop='CB1Inh', tdiscard=200)],
                              color='gray')
                except KeyError:
                    pass
                # ge.save_on_desktop(fig1, '%s-1.png' % cond)
                # ge.save_on_desktop(fig2, '%s-2.png' % cond)

                        
        ge.set_plot(AX[0], xticks=range(len(CONDS)), xticks_labels=CONDS, xticks_rotation=70,
                    ylabel='exc. rate (Hz)')
        ge.set_plot(AX[1], xticks=range(len(CONDS)), xticks_labels=CONDS, xticks_rotation=70,
                    ylabel='CB1 rate (Hz)')
        ge.save_on_desktop(fig, 'fig.png')
        plt.show()
    
    elif sys.argv[-1]=='V1':
        run_single_sim(Model,
                       filename='data/CB1_ntwk_model-V1.h5',
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=4,
                                            with_pop_act=True,
                                            with_synaptic_currents=False,
                                            with_synaptic_conductances=False,
                                            verbose=False), seed=3)
                   
    elif sys.argv[-1]=='V2':

        decrease = 20./100.
        # decreasing CB1 efficacy on PN
        Model['psyn_CB1Inh_Exc'] = (1-decrease)*Model['psyn_CB1Inh_Exc']
        Model['Q_CB1Inh_Exc'] = (1-decrease)*Model['psyn_CB1Inh_Exc']
        
        run_single_sim(Model, filename='data/CB1_ntwk_model-V2.h5')
    else:
        run_single_sim(Model, filename='data/CB1_ntwk_model-V1.h5')
        print('Results of the simulation are stored as:', 'data/CB1_ntwk_model.h5')
        print('--> Run \"python CB1_ntwk_model.py plot\" to plot the results')

        
