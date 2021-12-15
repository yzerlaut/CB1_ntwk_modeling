from Model import *
from datavyz import ges as ge
from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
from analyz.processing.signanalysis import gaussian_smoothing

def build_Faff_array(Model, mean=-2, std=8, seed=2):

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    OU = OrnsteinUhlenbeck_Process(mean, std, 1000, dt=Model['dt'], tstop=Model['tstop'], seed=seed)
    return t_array, gaussian_smoothing(np.clip(OU, 0, np.inf), int(50/Model['dt']))

    
Model['tstop'] = 6e3
# Model['common_Vthre_Inh'] = -50.

Faff0 = 3.

if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        fig, AX = ge.figure(axes=(2,1), figsize=(.8,1.), hspace=2.)

        data = ntwk.recording.load_dict_from_hdf5('data/time-varying-Input-V1.h5')
    
        
        # # ## plot
        fig1, _ = ntwk.plots.activity_plots(data,
                                            smooth_population_activity=10.,
                                            COLORS=[plt.cm.tab10(i) for i in [2,3,1]],
                                            Vm_plot_args={'subsampling':2, 'clip_spikes':True})
        fig1.suptitle('V1')
        AX[0].bar([0], [ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)], color='lightgray')
        AX[1].bar([0], [ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                               method='STTC',
                                                               Tbin=100, Nmax_pairs=2000)], color='lightgray')
                        
        """
        
        data = ntwk.recording.load_dict_from_hdf5('data/time-varying-Input-V2.h5')
        # # ## plot
        fig2, _ = ntwk.plots.activity_plots(data, smooth_population_activity=10., COLORS=[plt.cm.tab10(i) for i in [2,3,1]])
        fig2.suptitle('V2')
        AX[0].bar([1], [ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)], color='gray')
        AX[1].bar([1], [ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                               method='STTC',
                                                               Tbin=100, Nmax_pairs=2000)], color='gray')
        
        ge.set_plot(AX[0], xticks=[0,1], xticks_labels=['V1', 'V2'], ylabel='exc. rate (Hz)')
        ge.set_plot(AX[1], xticks=[0,1], xticks_labels=['V1', 'V2'], ylabel='exc. STTC')
        """
        
        plt.show()
        
    if sys.argv[-1]=='V1':

        # Model['inh_exc_ratio'] = 0.3
        # Model['CB1_PV_ratio'] = 1./3.
        
        run_single_sim(Model,
                       Faff_array=Faff0+build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            with_synaptic_currents=True,
                                            # with_synaptic_conductances=False,
                                            verbose=False),
                       filename='data/time-varying-Input-V1.h5')
        
    elif sys.argv[-1]=='V2':

        Model['inh_exc_ratio'] = 0.2
        Model['CB1_PV_ratio'] = 0.05
        
        run_single_sim(Model,
                       Faff_array=Faff0+build_Faff_array(Model)[1],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            with_synaptic_currents=True,
                                            # with_synaptic_conductances=False,
                                            verbose=False),
                       filename='data/time-varying-Input-V2.h5')

    else:
        print('need args')

