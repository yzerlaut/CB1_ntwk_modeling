import os
import numpy as np
import matplotlib.pylab as plt

import sys, pathlib
# sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'neural_network_dynamics'))
sys.path.append(os.path.join(str(pathlib.Path(__file__).resolve().parents[0]), 'neural_network_dynamics'))

import ntwk

Model = {
    ## -----------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## -----------------------------------------------------------------------

    # numbers of neurons in population
    'N_Exc':4000, 'N_PvInh':500, 'N_CB1Inh':500, 'N_AffExc':200,
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
    'p_Exc_Exc':0.02, 'p_Exc_PvInh':0.05, 'p_Exc_CB1Inh':0.05, 
    'p_PvInh_Exc':0.05, 'p_PvInh_PvInh':0.05, 'p_PvInh_CB1Inh':0.05, 
    'p_CB1Inh_Exc':0.25, 'p_CB1Inh_PvInh':0.25, 'p_CB1Inh_CB1Inh':0.25,
    'psyn_CB1Inh_Exc':0.2, #'psyn_AffExc_Exc':0.01, 'psyn_Exc_Exc':0.1,
    'p_AffExc_Exc':0.1, 'p_AffExc_PvInh':0.1, 'p_AffExc_CB1Inh':0.1,
    # afferent stimulation (0 by default)
    'F_AffExc':5.,
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

def run_single_sim(Model,
                   build_pops_args=dict(with_raster=True,
                                        # with_Vm=2,
                                        with_pop_act=True,
                                        with_synaptic_currents=False,
                                        with_synaptic_conductances=False,
                                        verbose=False),
                   Npairs=4,
                   filename='CB1_ntwk_model_data.h5'):

    NTWK = ntwk.build.populations(Model, ['Exc', 'PvInh', 'CB1Inh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  **build_pops_args)

    ntwk.build.recurrent_connections(NTWK, SEED=Model['SEED']*(Model['SEED']+1),
                                     verbose=build_pops_args['verbose'],
                                     store_connections=True)


    # we manually add the recording of a few connection pairs for the two inhibitory to excitatory connections
    if not 'VMS' in NTWK:
        NTWK['VMS'] = []

    exc_rec, pv_rec, cb1_rec = [], [], []
    while len(exc_rec)!=Npairs:
        # PV -> Exc
        index = np.random.choice(len(NTWK['connections'][1,0]['j']),1)[0]
        if (NTWK['connections'][1,0]['j'][index] not in exc_rec) and (NTWK['connections'][1,0]['i'][index] not in pv_rec):
            pv_rec.append(NTWK['connections'][1,0]['i'][index])
            exc_rec.append(NTWK['connections'][1,0]['j'][index])
            
    while len(exc_rec)!=(2*Npairs):
        # CB1 -> Exc
        index = np.random.choice(len(NTWK['connections'][2,0]['j']),1)[0]
        if (NTWK['connections'][2,0]['j'][index] not in exc_rec) and (NTWK['connections'][2,0]['i'][index] not in cb1_rec):
            cb1_rec.append(NTWK['connections'][2,0]['i'][index])
            exc_rec.append(NTWK['connections'][2,0]['j'][index])

    for pop, irecs in zip(NTWK['POPS'], [exc_rec, pv_rec, cb1_rec]):
        NTWK['VMS'].append(ntwk.StateMonitor(pop, 'V', record=irecs))
        
    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################
    faff = Model['F_AffExc']
    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['Exc', 'PvInh', 'CB1Inh']): # both on excitation and inhibition
        ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                              t_array, faff+0.*t_array,
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



def plot_Vm_pairs(data,
                  POP_KEYS = None, COLORS=None, NVMS=None,
                  tzoom=[0, np.inf],
                  vpeak=-40, vbottom=-80, shift=20.,
                  Tbar=50., Vbar=20.,
                  lw=1, ax=None):

    fig, AX = plt.subplots(len(data['VMS_PvInh']), 2, figsize=(15,5))
    plt.subplots_adjust(left=.03, bottom=.1, hspace=0.1, wspace=.1, right=.99, top=.99)

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    window = -10+np.arange(int(30/data['dt']))*data['dt']
    
    for i in range(len(data['VMS_PvInh'])):

        VmI = data['VMS_PvInh'][i].flatten()
        VmE = data['VMS_Exc'][i].flatten()

        AX[i][0].plot(t, VmI, color=plt.cm.tab10(3))
        AX[i][0].plot(t, 10+VmE, color=plt.cm.tab10(2))

        # adding spikes
        tspikes, threshold, Vevoked = *ntwk.plots.find_spikes_from_Vm(t, VmI, data, 'PvInh'), []
        for ts in tspikes:
            AX[i][0].plot([ts, ts], np.array([threshold, threshold+10]), '--', color=plt.cm.tab10(3), lw=lw)
            new_V = VmE[t>ts+window[0]][:len(window)]
            if len(new_V)==len(window):
                Vevoked.append(new_V)
        if len(Vevoked)>0:
            inset = AX[i][0].inset_axes([-0.05, 0.5, 0.1, 0.4])
            inset.plot(window, np.mean(Vevoked, axis=0), color=plt.cm.tab10(2))
        inset.set_xticks([0,10])
        inset.set_yticks([])
        
        AX[i][0].axis('off')

    for i in range(len(data['VMS_CB1Inh'])):

        VmI = data['VMS_CB1Inh'][i].flatten()
        VmE = data['VMS_Exc'][i+len(data['VMS_PvInh'])].flatten()

        AX[i][1].plot(t, VmI, color=plt.cm.tab10(1))
        AX[i][1].plot(t, 10+VmE, color=plt.cm.tab10(2))
        
        # adding spikes
        tspikes, threshold, Vevoked = *ntwk.plots.find_spikes_from_Vm(t, VmI, data, 'CB1Inh'), []
        for ts in tspikes:
            AX[i][1].plot([ts, ts], np.array([threshold, threshold+10]), '--', color=plt.cm.tab10(1), lw=lw)
            new_V = VmE[t>ts+window[0]][:len(window)]
            if len(new_V)==len(window):
                Vevoked.append(new_V)
        if len(Vevoked)>0:
            inset = AX[i][1].inset_axes([-0.05, 0.5, 0.1, 0.4])
            inset.plot(window, np.mean(Vevoked, axis=0), color=plt.cm.tab10(2))
        inset.set_xticks([0,10])
        inset.set_yticks([])
        AX[i][1].axis('off')
            
    return fig, AX
    
if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        ## load file
        data = ntwk.recording.load_dict_from_hdf5('CB1_ntwk_model_data.h5')

        # ## plot
        fig, _ = ntwk.plots.activity_plots(data, smooth_population_activity=10., COLORS=[plt.cm.tab10(i) for i in [2,3,1]])

        fig, AX = plot_Vm_pairs(data)
        
        # print(' synchrony=%.2f' % ntwk.analysis.get_synchrony_of_spiking(data))
        plt.show()
    else:
        run_single_sim(Model, filename='CB1_ntwk_model_data.h5')
        print('Results of the simulation are stored as:', 'CB1_ntwk_model_data.h5')
        print('--> Run \"python CB1_ntwk_model.py plot\" to plot the results')

