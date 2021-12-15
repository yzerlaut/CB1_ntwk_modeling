from Model import *
from datavyz import ges as ge
from analyz.signal_library.stochastic_processes import OrnsteinUhlenbeck_Process
from analyz.processing.signanalysis import gaussian_smoothing

def build_Faff_array(Model, mean=0, std=10):

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
    OU = OrnsteinUhlenbeck_Process(mean, std, 1000, dt=Model['dt'], tstop=Model['tstop'], seed=1)
    return t_array, gaussian_smoothing(np.clip(OU, 0, np.inf), int(50/Model['dt']))

    

if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        Model['tstop'] = 10000
        ge.plot(*build_Faff_array(Model))
        
        ## load file
        # data = ntwk.recording.load_dict_from_hdf5('data/time-varying-Input.h5')

        # # ## plot
        # fig, _ = ntwk.plots.activity_plots(data, smooth_population_activity=10., COLORS=[plt.cm.tab10(i) for i in [2,3,1]])

        plt.show()
    else:

        Model['tstop'] = 10000

        ge.plot(*build_Faff_array(Model))
        run_single_sim(Model,
                       Faff_array=build_Faff_array({}),
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=3,
                                            with_pop_act=True,
                                            with_synaptic_currents=True,
                                            # with_synaptic_conductances=False,
                                            verbose=False),
                       filename='data/time-varying-Input.h5')
        
        print('Results of the simulation are stored as:', 'data/time-varying-Input.h5')
        print('--> Run \"python time-varying-Input.py plot\" to plot the results')

