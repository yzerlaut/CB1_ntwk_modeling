from Model import *
from datavyz import ges as ge

def running_sim_func(Model, a=0):
    run_single_sim(Model, filename=Model['filename'])

if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        Model = {'data_folder': 'data/', 'zip_filename':'data/exc-inh-CB1-PV-scan.zip'}
        Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model)

        print(PARAMS_SCAN)
        
    else:

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/exc-inh-CB1-PV-scan.zip'

        ntwk.scan.run(Model,
                      ['inh_exc_ratio', 'CB1_PV_ratio'],
                      [np.linspace(0.2,0.4,3), np.linspace(0.05,0.95,3)],
                      running_sim_func,
                      parallelize=False)
        

