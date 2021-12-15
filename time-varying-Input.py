from Model import *
from datavyz import ges as ge
from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
from analyz.processing.signanalysis import gaussian_smoothing

def build_Faff_array(Model, mean=-4, std=20, seed=7):

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    OU = OrnsteinUhlenbeck_Process(mean, std, 1000, dt=Model['dt'], tstop=Model['tstop'], seed=seed)
    OU_clipped = gaussian_smoothing(np.clip(OU, 0, np.inf), int(50/Model['dt']))
    OU_clipped[t_array>4000] = 0
    return t_array, OU_clipped

    
Model['tstop'] = 15e3
# Model['common_Vthre_Inh'] = -50.

Model['F_AffExcBG'] = 2.5

if __name__=='__main__':
    
    if sys.argv[-1]=='input':

        ge.plot(*build_Faff_array(Model))
        ge.show()
        
    elif 'plot' in sys.argv[-1]:
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        fig, AX = ge.figure(axes=(2,1), figsize=(.8,1.), hspace=2.)


        if 'plot-' in sys.argv[-1]:
            CONDS = sys.argv[-1].split('plot-')
        else:
            CONDS = ['V1', 'V2', 'V2-low-Aff']

        for i, cond in enumerate(CONDS):

            if os.path.isfile('data/time-varying-Input-%s.h5' % cond):
                data = ntwk.recording.load_dict_from_hdf5('data/time-varying-Input-%s.h5' % cond)
                # # ## plot
                fig1, _ = ntwk.plots.activity_plots(data,
                                                    smooth_population_activity=10.,
                                                    COLORS=[plt.cm.tab10(i) for i in [2,3,1]],
                                                    Vm_plot_args={'subsampling':2, 'clip_spikes':True})
                fig1.suptitle(cond)
                AX[0].bar([i], [ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)], color='gray')
                AX[1].bar([i], [ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                                       method='STTC',
                                                                       Tbin=100, Nmax_pairs=2000)], color='gray')
                        
        ge.set_plot(AX[0], xticks=[0,1,2], xticks_labels=['V1', 'V2', 'V2-lA'], xticks_rotation=70,
                    ylabel='exc. rate (Hz)')
        ge.set_plot(AX[1], xticks=[0,1,2], xticks_labels=['V1', 'V2', 'V2-lA'], xticks_rotation=70,
                    ylabel='exc. STTC')
        
        plt.show()
        
    if sys.argv[-1]=='V1':

        run_single_sim(Model,
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/time-varying-Input-V1.h5')
        
    elif sys.argv[-1]=='V2':

        # decreasing V2 efficacy
        for target in ['Exc', 'PvInh', 'CB1Inh']:
            # Model['p_CB1Inh_%s' % target] = 0.15
            Model['Q_CB1Inh_%s' % target] = 5.
            
        run_single_sim(Model,
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/time-varying-Input-V2.h5')

    elif sys.argv[-1]=='V2-low-Aff':

        # decreasing V2 efficacy
        for target in ['Exc', 'PvInh', 'CB1Inh']:
            Model['Q_CB1Inh_%s' % target] = 5.
            Model['p_AffExcTV_%s' % target] = 0.01
            
        run_single_sim(Model,
                       Faff_array=build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            verbose=False),
                       filename='data/time-varying-Input-V2-low-Aff.h5')
        
    else:
        print('need args')

