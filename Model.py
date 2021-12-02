import os
import numpy as np
import matplotlib.pylab as plt

import sys, pathlib
sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'neural_network_dynamics'))

import main as ntwk

Model = {
    ## -----------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## -----------------------------------------------------------------------

    # numbers of neurons in population
    'N_Exc':4000, 'N_PvInh':1000, 'N_CB1Inh':1000, 'N_AffExc':100,
    # synaptic weights
    'Q_Exc_Exc':2., 'Q_Exc_PvInh':2.,  'Q_Exc_CB1Inh':2.,
    'Q_PvInh_Exc':10., 'Q_PvInh_PvInh':10., 'Q_PvInh_CB1Inh':10., 
    'Q_CB1Inh_Exc':10., 'Q_CB1Inh_PvInh':10., 'Q_CB1Inh_CB1Inh':10., 
    'Q_AffExc_Exc':4., 'Q_AffExc_PvInh':4., 'Q_AffExc_CB1Inh':4., 
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # connectivity parameters
    'p_Exc_Exc':0.05, 'p_Exc_PvInh':0.05, 'p_Exc_CB1Inh':0.05, 
    'p_PvInh_Exc':0.05, 'p_PvInh_PvInh':0.05, 'p_PvInh_CB1Inh':0.05, 
    'p_CB1Inh_Exc':0.05, 'p_CB1Inh_PvInh':0.05, 'p_CB1Inh_CB1Inh':0.05,
    'psyn_CB1Inh_Exc':0.2, 'psyn_AffExc_Exc':0.01, 'psyn_Exc_Exc':0.1,
    'p_AffExc_Exc':0.1, 'p_AffExc_PvInh':0.1, 'p_AffExc_CB1Inh':0.1,
    # afferent stimulation (0 by default)
    'F_AffExc':0.,
    # simulation parameters
    'dt':0.1, 'tstop': 1000., 'SEED':3, # low by default, see later
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


if sys.argv[-1]=='plot':
    # ######################
    # ## ----- Plot ----- ##
    # ######################
    
    ## load file
    data = ntwk.load_dict_from_hdf5('CB1_ntwk_model_data.h5')

    # ## plot
    fig, _ = ntwk.activity_plots(data, smooth_population_activity=10.)
    
    plt.show()
else:
    NTWK = ntwk.build_populations(Model, ['Exc', 'PvInh', 'CB1Inh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=True,
                                  with_Vm=4,
                                  # with_synaptic_currents=True,
                                  # with_synaptic_conductances=True,
                                  verbose=True)

    ntwk.build_up_recurrent_connections(NTWK, SEED=5, verbose=True)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    faff = 5.
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['Exc', 'PvInh', 'CB1Inh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff+0.*t_array,
                                         verbose=True,
                                         SEED=int(37*faff+i)%37)

    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)

    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=True)

    ntwk.write_as_hdf5(NTWK, filename='CB1_ntwk_model_data.h5')
    print('Results of the simulation are stored as:', 'CB1_ntwk_model_data.h5')
    print('--> Run \"python CB1_ntwk_model.py plot\" to plot the results')

