import sys, os

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

def running_sim_func(Model):
    run_single_sim(Model,
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=0,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])

    
def input_output_analysis(data):

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    output = []
    for et in data['event_times']:
        cond = (t>(et-50)) & (t<(et+50))
        output.append(np.mean(data['POP_ACT_L23Exc'][cond]))

    fig, ax = ge.plot(data['event_amplitudes'], output,
                      xlabel='input to L4 (Hz)', ylabel='L23 PN rate (Hz)')
    return fig, ax 



if sys.argv[-1]=='bg-analysis':
    # means scan

    Model2 = {'data_folder': './data/', 'zip_filename':'data/L4-bg-scan.zip'}
    Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2, filenames_only=True)

    current_residual, current_params = np.inf, {}
    for filename in PARAMS_SCAN['FILENAMES']:
        data = ntwk.recording.load_dict_from_hdf5(filename)
        print('- p_AffExcBG_L4Exc=%.3f: L4=%.2fHz, L23=%.2fHz' % (data['p_AffExcBG_L4Exc'],
                                                                  data['rate_L4Exc'],data['rate_L23Exc']))

elif 'bg' in sys.argv[-1]:
    # means scan, possible options:
    # "bg", "bg-fix-missing", "bg-with-repeat" or "bg-fix-missing-with-repeat"
    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/L4-bg-scan.zip'

    Model['tstop'] = 1000 # 1s and we discard the first 200ms

    def running_sim_func(Model):
        run_single_sim(Model,
                       REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                       AFF_POPS=['AffExcBG'],
                       build_pops_args=dict(with_raster=False,
                                            with_Vm=0,
                                            with_pop_act=True,
                                            verbose=False),
                       filename=Model['filename'])

    
    pconn_values = np.linspace(0.01, 0.1, 16)

    ntwk.scan.run(Model, ['p_AffExcBG_L4Exc'], [pconn_values], running_sim_func,
                  fix_missing_only=('fix-missing' in sys.argv[-1]),
                  parallelize=True)

    if 'with-repeat' in sys.argv[-1]:
        # "scan-with-repeat" or "scan-fix-missing-with-repeat"
        for i in range(5):
            ntwk.scan.run(Model, ['p_AffExcBG_L4Exc'], [pconn_values], running_sim_func,
                          fix_missing_only=True, parallelize=True)


elif sys.argv[-1]=='test-run':

    Model['p_L4Exc_L23Exc'] = 0.15
    Model['p_L4Exc_Inh'] = 0.1
    
    Model['event_amplitudes'] = np.linspace(0, 10, 10)
    Model['event_width'] = 200
    Model['event_times'] = 200+1000*np.arange(10)
    Model['tstop'] = Model['event_times'][-1]+3*Model['event_width']

    run_single_sim(Model,
                   REC_POPS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                   AFF_POPS=['AffExcBG', 'AffExcTV'],
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=0,
                                        with_pop_act=True,
                                        verbose=False),
                   filename='data/L4-circuit.h5')
    
elif sys.argv[-1]=='test-analysis':

    data = ntwk.recording.load_dict_from_hdf5('data/L4-circuit.h5')

    fig, ax = input_output_analysis(data)
    fig.savefig('fig.png')
    
    fig, _ = ntwk.plots.activity_plots(data, subsampling=20)
    fig.savefig('fig_raw.png')

else:
    print('need args')
