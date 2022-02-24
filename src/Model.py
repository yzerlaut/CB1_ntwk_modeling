import os, sys, pathlib, time

import numpy as np
import matplotlib.pylab as plt

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
import ntwk

Model = {
    ## ----------------------------------------------------------------------------------
    ### ---- NETWORK MODEL PARAMETERS ---------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)-
    ## ----------------------------------------------------------------------------------
    # numbers of neurons in the different population population (integer)
    'N_L23Exc':4000, 'N_PvInh':500, 'N_CB1Inh':500, 'N_L4Exc':4000, # population of recurrent network model
    'N_AffExcBG':4000, # population for afferent background activity
    'N_AffExcTV':4000, # population for afferent input activity (i.e. time varying)
    # synaptic weights -- Q_PRE_POST --  (in nS)
    'Q_AffExcBG_L4Exc':2., 'Q_AffExcBG_L23Exc':2., 'Q_AffExcBG_PvInh':2., 'Q_AffExcBG_CB1Inh':2., 
    'Q_L4Exc_L23Exc':2., 'Q_L4Exc_PvInh':2.,  'Q_L4Exc_CB1Inh':2.,
    'Q_L23Exc_L23Exc':2., 'Q_L23Exc_PvInh':2.,  'Q_L23Exc_CB1Inh':2.,
    'Q_PvInh_L23Exc':10., 'Q_PvInh_PvInh':10., 'Q_PvInh_CB1Inh':10.,
    'Q_CB1Inh_L4Exc':10., 'Q_CB1Inh_L23Exc':10., 'Q_CB1Inh_PvInh':10., 'Q_CB1Inh_CB1Inh':10.,
    'Q_AffExcTV_L4Exc':2.,
    # synaptic time constants (in ms) - "e": excitation, "i": inhibition
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials (in mV) - "e": excitation, "i": inhibition
    'Ee':0., 'Ei': -80.,
    # connectivity parameters  -- p_PRE_POST -- (as a probability of connection, i.e. 0<=p<1 )
    ## === N.B. some 0 connectivities are set after optimization, see below === ##
    # L23 circuit
    'p_AffExcBG_L23Exc':0, 'p_AffExcBG_PvInh':0, 'p_AffExcBG_CB1Inh':0,
    'p_L23Exc_L23Exc':0.05, 'p_L23Exc_PvInh':0.05, 'p_L23Exc_CB1Inh':0.05,
    'p_PvInh_L23Exc':0.067, 'p_PvInh_PvInh':0,
    'p_CB1Inh_L23Exc':0.067,'p_CB1Inh_CB1Inh':0,
    'psyn_CB1Inh_L23Exc':0.5, # probabilities of syn. transmission for CB1 synapses
    # L4-L23 circuit
    'p_AffExcBG_L4Exc':0,
    'p_L4Exc_L23Exc':0, 'p_L4Exc_PvInh':0, 'p_L4Exc_CB1Inh':0,
    'p_CB1Inh_L4Exc':0.025, # CB1 to L4 connection !
    'psyn_CB1Inh_L4Exc':0.5, # probabilities of syn. transmission for CB1 synapses
    # time-varying input to L4 only
    'p_AffExcTV_L4Exc':0.025, 'p_AffExcTV_L23Exc':0, 'p_AffExcTV_CB1Inh':0, 'p_AffExcTV_PvInh':0,
    # background afferent activity level (in Hz)
    'F_AffExcBG':4,
    # simulation parameters 
    'dt':0.1, 'tstop': 1000., 'SEED':5, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # Gl: input conductance at rest (in nS), Cm; input capacitance (in pF)
    # Trefrac: refractory period (in ms), El: leak reversal potential (in mV)
    # Vthre: spiking threshold (in mV), Vreset: rset potential after spike (in mV),
    # deltaV, a, b, tauw NOT USED HERE, AdExp params but here we stick on LIF model
    # ---------------------------------------------------------------------------------
    # --> L4 Excitatory population (L4Exc, L4 pyramidal cells)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':5.,
    'L4Exc_El':-70., 'L4Exc_Vthre':-50., 'L4Exc_Vreset':-70., 'L4Exc_deltaV':0.,
    'L4Exc_a':0., 'L4Exc_b': 0., 'L4Exc_tauw':1e9,
    # --> L23 Excitatory population (L23Exc, L23 pyramidal cells)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':5.,
    'L23Exc_El':-70., 'L23Exc_Vthre':-50., 'L23Exc_Vreset':-70., 'L23Exc_deltaV':0.,
    'L23Exc_a':0., 'L23Exc_b': 0., 'L23Exc_tauw':1e9,
    # --> PV-Inhibitory population (Inh, PV+ interneurons)
    'PvInh_Gl':10., 'PvInh_Cm':200.,'PvInh_Trefrac':5.,
    'PvInh_El':-70., 'PvInh_Vthre':-53., 'PvInh_Vreset':-70., 'PvInh_deltaV':0.,
    'PvInh_a':0., 'PvInh_b': 0., 'PvInh_tauw':1e9,
    # --> CB1-Inhibitory population (Inh, CB1+ interneurons)
    'CB1Inh_Gl':10., 'CB1Inh_Cm':200.,'CB1Inh_Trefrac':5.,
    'CB1Inh_El':-70., 'CB1Inh_Vthre':-53., 'CB1Inh_Vreset':-70., 'CB1Inh_deltaV':0.,
    'CB1Inh_a':0., 'CB1Inh_b': 0., 'CB1Inh_tauw':1e9
}

    
########################################
########## update default params #######
########################################

# FROM: "$python code/L23_connec_params.py scan-analysis"
Model.update({'p_AffExcBG_L23Exc': 0.075, 'p_AffExcBG_PvInh': 0.1, 'p_AffExcBG_CB1Inh': 0.025,
              'p_PvInh_PvInh': 0.075, 'p_CB1Inh_CB1Inh': 0.025})

# FROM: "$python code/L4.py bg; python code/L4.py bg-analysis"
Model.update({'p_AffExcBG_L4Exc': 0.01})

# FROM: "$python code/L4.py L23"
# Model.update({'p_L4Exc_L23Exc': 0.1, 'p_L4Exc_Inh': 0.025})
Model.update({'p_L4Exc_L23Exc': 0.01})


def decrease_CB1_efficacy_on_L23PN(Model,
                                   decrease=0.5):
    Model2 = Model.copy()
    Model2['psyn_CB1Inh_L23Exc'] = (1.-decrease)*Model2['psyn_CB1Inh_L23Exc']
    return Model2

def increase_CB1_inhibition_on_L4PN(Model,
                                    pconn_increase_factor=2,
                                    Q=None, psyn=None):
    Model2 = Model.copy()
    Model2['p_CB1Inh_L4Exc'] = pconn_increase_factor*Model2['p_CB1Inh_L4Exc']
    return Model2

def update_model(Model, key,
                 CB1_L23PN_decrease_factor=0.5,
                 p_CB1_L4PN_increase_factor=2):

    if 'V2' in key:
        # add CB1 inhibition on L4
        if not ('no-CB1-L4' in key):
            Model = increase_CB1_inhibition_on_L4PN(Model,
                                                    pconn_increase_factor=p_CB1_L4PN_increase_factor)
        if not ('CB1-KO' in key):
            # decreasing CB1 efficacy on PN if not V2-CB1-KO
            Model = decrease_CB1_efficacy_on_L23PN(Model,
                                                   decrease=CB1_L23PN_decrease_factor)

    return Model


################################
########## input ###############
################################

def gaussian(x, mean=0., std=1.):
    return np.exp(-(x-mean)**2/2./std**2)/np.sqrt(2.*np.pi)/std

def build_time_varying_afferent_array(Model,
                                      event_amplitudes=[6,6],
                                      event_width=150, 
                                      event_times = [4000, 9000]):

    t_array = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    g = 0*t_array
    norm_factor = float(1./gaussian(0, 0, event_width))
    for t, amp in zip(event_times, event_amplitudes):
        g[:] += amp*gaussian(t_array[:], t, event_width)*norm_factor

    return t_array, g

##############################
########## run ###############
##############################

def run_single_sim(Model,
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=2,
                                        with_pop_act=True,
                                        with_synaptic_currents=False,
                                        with_synaptic_conductances=False,
                                        verbose=False),
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG'],
                   specific_record_function=None, srf_args={},
                   Faff_array=None,
                   filename='CB1_ntwk_model_data.h5',
                   seed=0,
                   verbose=True):

    start = time.time()
    try:
        print('running sim with ntwk v%s' % ntwk.version)
    except NameError:
        import ntwk
    
    if ('p_Inh_L23Exc' in Model):
        # used in code/L23_connec_params.py
        Model['p_CB1Inh_L23Exc'] = Model['p_Inh_L23Exc']
        Model['p_PvInh_L23Exc'] = Model['p_Inh_L23Exc']

    if ('p_L4Exc_Inh' in Model):
        # used in code/L4.py
        Model['p_L4Exc_CB1Inh'] = Model['p_L4Exc_Inh']
        Model['p_L4Exc_PvInh'] = Model['p_L4Exc_Inh']
        
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
    if (Faff_array is None) and ('event_times' in Model):
        Faff_array = build_time_varying_afferent_array(Model,
                                                       event_amplitudes=Model['event_amplitudes'],
                                                       event_width=Model['event_width'],
                                                       event_times=Model['event_times'])[1]    
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
    network_sim = ntwk.collect_and_run(NTWK, verbose=verbose)


    ##########################
    ## ----- Analysis ----- ##
    ##########################
    KEY_NOT_TO_RECORD=['Raster_AFF_POP', 'iRASTER_PRE', 'iRASTER_PRE_in_terms_of_Pre_Pop']
    if 'RASTER' in NTWK:
        NTWK['STTC_L23Exc'] = ntwk.analysis.get_synchrony_of_spiking(NTWK, pop='L23Exc',
                                                                     method='STTC',
                                                                     tzoom=[200,Model['tstop']],
                                                                     Tbin=300, Nmax_pairs=2000)
        NTWK['STTC_L23Exc_pre_stim'] = ntwk.analysis.get_synchrony_of_spiking(NTWK, pop='L23Exc',
                                                                              method='STTC',
                                                                              tzoom=[200,8000],
                                                                              Tbin=300, Nmax_pairs=2000)
        SINGLE_VALUES_KEYS=['STTC_L23Exc', 'STTC_L23Exc_pre_stim']
    else:
        SINGLE_VALUES_KEYS=[]
        for i in range(len(NTWK['NEURONS'])):
            name = NTWK['NEURONS'][i]['name']
            NTWK['rate_%s'%name] = ntwk.analysis.get_mean_pop_act(NTWK, pop=name)
            SINGLE_VALUES_KEYS.append('rate_%s'%name)
            
        KEY_NOT_TO_RECORD+=['POP_ACT', 'Rate_AFF_POP']
    
    ######################
    ## ----- Save ----- ##
    ######################
    ntwk.recording.write_as_hdf5(NTWK,
                                 filename=filename,
                                 SINGLE_VALUES_KEYS=SINGLE_VALUES_KEYS,
                                 KEY_NOT_TO_RECORD=KEY_NOT_TO_RECORD)
    
    if verbose:
        print('--> done ! simulation took %.1fs' % (time.time()-start))


if __name__=='__main__':
    
    if sys.argv[-1] in ['syn', 'connec', 'matrix']:

        fig, _, _ = ntwk.plots.connectivity_matrix(Model,
                                                   REC_POPS=['L23Exc', 'PvInh', 'CB1Inh', 'L4Exc'],
                                                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                                                   COLORS=[ge.green, ge.red, ge.orange, ge.blue, 'k', ge.brown],
                                                   blank_zero=True,
                                                   graph_env=ge)
        ge.save_on_desktop(fig, 'fig.svg')
        # ge.show()
            
    elif 'plot' in sys.argv[-1]:
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        if 'plot-' in sys.argv[-1]:
            CONDS = [sys.argv[-1].split('plot-')[-1]]
        else:
            CONDS = ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']

        from plot import raw_data_fig_multiple_sim

        fig, AX = raw_data_fig_multiple_sim([('data/CB1_ntwk_model-%s.h5' % cond) for cond in CONDS if os.path.isfile('data/CB1_ntwk_model-%s.h5' % cond)],
                                            with_log_scale_for_act=True, verbose=True)
        
        fig.savefig('fig.png')
    
    
    elif sys.argv[-1] in ['V1', 'V2', 'V2-CB1-KO', 'V2-no-CB1-L4']:

        Model = update_model(Model, sys.argv[-1])

        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG'],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=1,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/CB1_ntwk_model-%s.h5' % sys.argv[-1])
                
    else:
        print('need args')


        
