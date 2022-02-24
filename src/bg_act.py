import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

Model['tstop'] = 1000 # 1s and we discard the first 200ms


if sys.argv[-1]=='L23-psyn-scan':

    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/L23-psyn-scan.zip'

    Nscan = 2*8

    def running_sim_func(Model, a=0, NVm=3):
        run_single_sim(Model,
                       REC_POPS=['L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG'],
                       build_pops_args=dict(with_raster=False,
                                            with_Vm=NVm,
                                            with_pop_act=True,
                                            verbose=False),
                       filename=Model['filename'])
        
    psyn_variations = np.linspace(0.2, 0.55, Nscan)
    ntwk.scan.run(Model, ['psyn_CB1Inh_L23Exc'], [psyn_variations], running_sim_func,
                  parallelize=True)

elif sys.argv[-1]=='L23-psyn-analysis':

    Model2 = {'data_folder': './data/', 'zip_filename':'data/L23-psyn-scan.zip'}
    Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2,
                                           filenames_only=True)

    fig, ax = ge.figure()
    ge.title(ax, 'L23 circuit\n(spont. act.)', size='small')

    rates, psyn = {'L23Exc':[], 'PvInh':[], 'CB1Inh':[]}, []
    for filename in PARAMS_SCAN['FILENAMES']:
        data = ntwk.recording.load_dict_from_hdf5(filename)
        for key in rates:
            rates[key].append(data['rate_%s'%key])
        psyn.append(float(data['psyn_CB1Inh_L23Exc']))

    for i, key, color in zip(range(3), ['L23Exc', 'PvInh', 'CB1Inh'], [ge.green, ge.red, ge.orange]):
        ax.plot(psyn, rates[key], color=color, lw=1)
        # ge.annotate(ax, i*'\n'+key, (1,1), ha='right', va='top')
        
    ge.set_plot(ax, xlabel='$p_{rel}$ CB1->L23PN', xticks=[0.25, 0.5],
                yscale='log', yticks=[1,10], yticks_labels=['1','10'],
                ylabel='rate (Hz)')

    ge.show()
    ge.save_on_desktop(fig, 'fig.svg')
    fig.savefig('doc/L23-psyn-variations.png')
    
if sys.argv[-1]=='L4-L23-psyn-pconn-scan':
    
    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/L4-L23-psyn-pconn-scan.zip'
    
    def running_sim_func(Model, a=0, NVm=3):
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                       build_pops_args=dict(with_raster=True,
                                            with_Vm=NVm,
                                            with_pop_act=True,
                                            verbose=False),
                       filename=Model['filename'])

    Nscan = 8
    # variations of CB1-signalling level
    psyn_variations = Model['psyn_CB1Inh_L23Exc']*np.linspace(0.2, 1.2, Nscan)
    pconn = Model['p_CB1Inh_L4Exc']*np.linspace(0.9, 4, Nscan)

    ntwk.scan.run(Model,
                  ['psyn_CB1Inh_L23Exc', 'p_CB1Inh_L4Exc'],
                  [psyn_variations, pconn],
                  running_sim_func,
                  parallelize=True)

elif sys.argv[-1]=='L4-L23-psyn-pconn-scan':

    Model2 = {'data_folder': './data/', 'zip_filename':'data/L4-L23-psyn-pconn-scan.zip'}
    Model2, PARAMS_SCAN, DATA = ntwk.scan.get(Model2)
    
    sumup = {'rate':[]}
    for data in DATA:
        sumup['rate'].append(ntwk.analysis.get_mean_pop_act(data, pop='L23Exc',
                                                            tdiscard=200))

    fig, ax, cb = ge.twoD_plot(np.array(PARAMS_SCAN['p_CB1Inh_L4Exc'])/Model['p_CB1Inh_L4Exc'],
                               np.array(PARAMS_SCAN['psyn_CB1Inh_L23Exc']),
                               np.array(sumup['rate']),
                               bar_legend_args={'label':'L23 PN rate (Hz)'})

    for x, y, label, color in zip([1, 4, 4],
                                  [0.5, 0.5, 0.25],
                                  ['V1', 'V2M-CB1-KO', 'V2M'],
                                  ge.colors[5:]):
        ax.scatter([x], [y], s=20, color='r', facecolor='none')
        ge.annotate(ax, label+'\n', (x, y), xycoords='data', ha='center', va='center', color='k', size='x-small')
    ge.set_plot(ax,
                xlabel=r'$\,_{CB1 \rightarrow L4}$ $p_{conn}$ factor',
                ylabel='psyn $\,_{CB1->L23}$')

    ge.show()
    
elif sys.argv[-1]=='main-scan-plot':

    Model2 = {'data_folder': './data/', 'zip_filename':'data/main-space-scan.zip'}
    Model2, PARAMS_SCAN, DATA = ntwk.scan.get(Model2)

    sumup = {'rate':[]}
    for data in DATA:
        sumup['rate'].append(ntwk.analysis.get_mean_pop_act(data, pop='L23Exc',
                                                            tdiscard=200))

    fig, ax, cb = ge.twoD_plot(np.array(PARAMS_SCAN['p_CB1Inh_L4Exc'])/Model['p_CB1Inh_L4Exc'],
                               np.array(PARAMS_SCAN['psyn_CB1Inh_L23Exc']),
                               np.array(sumup['rate']),
                               bar_legend_args={'label':'L23 PN rate (Hz)'})

    for x, y, label, color in zip([1, 4, 4],
                                  [0.5, 0.5, 0.25],
                                  ['V1', 'V2M-CB1-KO', 'V2M'],
                                  ge.colors[5:]):
        ax.scatter([x], [y], s=20, color='r', facecolor='none')
        ge.annotate(ax, label+'\n', (x, y), xycoords='data', ha='center', va='center', color='k', size='x-small')
    ge.set_plot(ax,
                xlabel=r'$\,_{CB1 \rightarrow L4}$ $p_{conn}$ factor',
                ylabel='psyn $\,_{CB1->L23}$')

    ge.show()
