from Model import *
from datavyz import ges as ge
# from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
# from analyz.processing.signanalysis import gaussian_smoothing
from analyz.signal_library.classical_functions import gaussian


def build_Faff_array(Model,
                     # mean=-4, std=10, # FOR OU
                     t0=4000, sT=300, amp=14.5,
                     seed=100):

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    ## --- Ornstein Uhlenbeck_Process ---
    # OU = OrnsteinUhlenbeck_Process(mean, std, 1000, dt=Model['dt'], tstop=Model['tstop'], seed=seed)
    # OU_clipped = gaussian_smoothing(np.clip(OU, 0, np.inf), int(50/Model['dt']))
    # OU_clipped[t_array>3000] = 0
    # return t_array, OU_clipped

    # g = gaussian(t_array, t0, sT) + gaussian(t_array, t0, sT)
    g = gaussian(t_array, 4000, sT) + gaussian(t_array, 15000, sT)
    return t_array, amp/g.max()*g
    

Model_pre = {
    ## -----------------------------------------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## -----------------------------------------------------------------------

    # numbers of neurons in population
    'N_thalExcTV':400,
    # synaptic weights
    'Q_thalExcTV_AffExcTV':1., 
    # synaptic time constants
    'Tse':5.,
    # synaptic reversal potentials
    'Ee':0.,
    # connectivity parameters
    'p_thalExcTV_AffExcTV':0.1,
    'p_AffExcBG_AffExcTV':0,
    'p_thalExcTV_Exc':0, 'p_thalExcTV_PvInh':0, 'p_thalExcTV_CB1Inh':0, 
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Excitatory population (Exc, recurrent excitation)
    'AffExcTV_Gl':10., 'AffExcTV_Cm':200.,'AffExcTV_Trefrac':5.,
    'AffExcTV_El':-70., 'AffExcTV_Vthre':-50., 'AffExcTV_Vreset':-70., 'AffExcTV_deltaV':0.,
    'AffExcTV_a':0., 'AffExcTV_b': 0., 'AffExcTV_tauw':1e9,
    #
    'tstop':20e3
}


Model.update(Model_pre)

###################################################
##########  renaming pops for convenience #########
###################################################
keys = list(Model.keys())
for key in keys:
    if ('AffExcTV_Exc' in key):
        Model[key.replace('AffExcTV_Exc', 'L4Exc_L23Exc')] = Model[key]
    if ('AffExcTV_' in key):
        Model[key.replace('AffExcTV_', 'L4Exc_')] = Model[key]
    if ('_AffExcTV' in key):
        Model[key.replace('_AffExcTV', '_L4Exc')] = Model[key]
    if ('Exc_' in key):
        Model[key.replace('Exc_', 'L23Exc_')] = Model[key]
    if ('_Exc' in key):
        Model[key.replace('_Exc', '_L23Exc')] = Model[key]
    
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

            if os.path.isfile('data/model-with-thal-%s.h5' % cond):
                data = ntwk.recording.load_dict_from_hdf5('data/model-with-thal-%s.h5' % cond)
                # # ## plot
                fig1, _ = ntwk.plots.activity_plots(data,
                                                    smooth_population_activity=10.,
                                                    COLORS=[plt.cm.tab10(i) for i in [0,2,3,1]],
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
        
        plt.show()
        
    elif sys.argv[-1]=='Aff':
        
        NTWK = ntwk.build.populations(Model, ['L4Exc'],
                                      AFFERENT_POPULATIONS=['thalExcTV'],
                                      with_raster=True,
                                      with_Vm=3,
                                      with_pop_act=True,
                                      verbose=False)

        ntwk.build.recurrent_connections(NTWK, SEED=Model['SEED']*(Model['SEED']+1),
                                         verbose=False)

        t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
        
        # constant background input
        ntwk.stim.construct_feedforward_input(NTWK, 'L4Exc', 'thalExcTV',
                                              t_array, build_Faff_array(Model)[1], #Model['F_L4Exc']+0.*t_array,
                                              verbose=False,
                                              SEED=1+2*Model['SEED']*(Model['SEED']+1))
        ntwk.build.initialize_to_rest(NTWK)
        network_sim = ntwk.collect_and_run(NTWK, verbose=True)

        ntwk.recording.write_as_hdf5(NTWK, filename='data/model-with-thal-Aff.h5')
    
    elif sys.argv[-1]=='V1':

        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'thalExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-thal-V1.h5')
        
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
                       AFF_POPS=['AffExcBG', 'thalExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-thal-V2.h5')

    elif sys.argv[-1]=='V2-CB1-KO':

        # NOT decreasing CB1 efficacy on PN
        
        # still adding CB1 inhibition on <4
        Model['p_CB1Inh_L4Exc'] = 0.1
        Model['psyn_CB1Inh_L4Exc'] = 0.5
        Model['Q_CB1Inh_L4Exc'] = 10.
            
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'thalExcTV'],
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/model-with-thal-V2-CB1-KO.h5')

        
    else:
        print('need args')

