from Model import *
from datavyz import ges as ge

def build_Faff_array():

    return

    

if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        ## load file
        data = ntwk.recording.load_dict_from_hdf5('data/time-varying-Input.h5')

        # ## plot
        fig, _ = ntwk.plots.activity_plots(data, smooth_population_activity=10., COLORS=[plt.cm.tab10(i) for i in [2,3,1]])

        print(' synchrony=%.2f' % ntwk.analysis.get_synchrony_of_spiking(data))
        plt.show()
    else:


        
        run_single_sim(Model,
                       Faff_array=None,
                       filename='data/time-varying-Input.h5')
        
        print('Results of the simulation are stored as:', 'data/time-varying-Input.h5')
        print('--> Run \"python CB1_ntwk_model.py plot\" to plot the results')

