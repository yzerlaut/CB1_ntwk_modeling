import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 1000 # 1s and we discard the first 200ms

def running_sim_func(Model, a=0, NVm=3):
    run_single_sim(Model,
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=NVm,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])

if __name__=='__main__':


    if sys.argv[-1]=='main-scan':

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/main-space-scan.zip'
        
        Nscan = 7

        # variations of CB1-signalling level
        psyn0 = Model['psyn_CB1Inh_L23Exc']
        psyn_variations = np.linspace(0.1, 1, Nscan)*Model['psyn_CB1Inh_L23Exc']
        pconn = Model['p_CB1Inh_L4Exc']*np.linspace(0.9, 5, Nscan)

        ntwk.scan.run(Model,
                      ['psyn_CB1Inh_L23Exc', 'p_CB1Inh_L4Exc'],
                      [psyn_variations, pconn],
                      running_sim_func,
                      parallelize=True)

    elif sys.argv[-1]=='main-scan-plot':

        Model2 = {'data_folder': './data/', 'zip_filename':'data/main-space-scan.zip'}
        Model2, PARAMS_SCAN, DATA = ntwk.scan.get(Model2)

        sumup = {'rate':[]}
        for data in DATA:
            sumup['rate'].append(ntwk.analysis.get_mean_pop_act(data, pop='L23Exc',
                                                                tdiscard=200))

        fig, ax, cb = ge.twoD_plot(np.array(PARAMS_SCAN['p_CB1Inh_L4Exc'])/Model['p_CB1Inh_L4Exc'],
                     1-np.array(PARAMS_SCAN['psyn_CB1Inh_L23Exc'])/Model['psyn_CB1Inh_L23Exc'],
                     # np.array(PARAMS_SCAN['psyn_CB1Inh_L23Exc']),
                     np.array(sumup['rate']),
                     # ylabel=r'psyn$\,_{CB1 \rightarrow L23}$',
                     bar_legend_args={'label':'L23 PN rate (Hz)'})

        for x, y, label, color in zip([1, 4, 4],
                                      [0, 0, 0.5],
                                      ['V1', 'V2M-CB1-KO', 'V2M'],
                                      ge.colors[5:]):
            ax.scatter([x], [y], s=20, color='r', facecolor='none')
            ge.annotate(ax, label+'\n', (x, y), xycoords='data', ha='center', va='center', color='k', size='x-small')
        ge.set_plot(ax,
                    xlabel=r'p$\,_{CB1 \rightarrow L4}$ factor',
                    ylabel='s$\,_{CB1}$')
                    
        ge.show()
