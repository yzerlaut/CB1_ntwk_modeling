import sys, os

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path += ['./datavyz', './neural_network_dynamics', './src']
from datavyz import graph_env_manuscript as ge
from Model import *
import ntwk

def running_sim_func(Model,
                     max_input=10.,
                     rec_pops=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                     aff_pops=['AffExcBG', 'AffExcTV']):

    # adding input
    Model['event_amplitudes'] = np.linspace(0, max_input, 40)
    Model['event_width'] = 200
    Model['event_times'] = 200+1000*np.arange(len(Model['event_amplitudes']))
    Model['tstop'] = Model['event_times'][-1]+3*Model['event_width']
    
    run_single_sim(Model,
                   REC_POPS=rec_pops,
                   AFF_POPS=aff_pops,
                   build_pops_args=dict(with_raster=True,
                                        with_Vm=0,
                                        with_pop_act=True,
                                        verbose=False),
                   filename=Model['filename'])

    
def input_output_analysis(data,
                          pop='L23Exc',
                          smoothing=2):

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']

    output = []
    for et in data['event_times']:
        cond = (t>(et-data['event_width']/4.)) & (t<(et+data['event_width']/4.))
        output.append(np.mean(data['POP_ACT_%s'%pop][cond]))
    # smoothing
    output = gaussian_filter1d(output, smoothing)

    return data['event_amplitudes'], output

if sys.argv[1]=='L23-ntwk':

    if sys.argv[-1]=='analysis':
        fig, ax = ge.figure(figsize=(.95,1.))
        for m in ['V1', 'V2']:
            data = ntwk.recording.load_dict_from_hdf5('data/gain-%s-%s.zip' % (sys.argv[1], m))
            x, y = input_output_analysis(data)
            ax.plot(x, y, label=m)
        ge.legend(ax, size='x-small', frameon=False)
        ge.set_plot(ax, xlabel='L23PN input (Hz)  ', ylabel='L23PN output (Hz)    ', title='L23 network', yticks=[0,2,4])
        fig.savefig('doc/L23-ntwk-gain.svg')
        ge.show()
    else:
        Model = update_model(Model, sys.argv[2])
        Model['filename'] = 'data/gain-%s-%s.zip' % (sys.argv[1], sys.argv[2])
        running_sim_func(Model,
                     rec_pops=['L23Exc', 'PvInh', 'CB1Inh'],
                     aff_pops=['AffExcBG', 'L4Exc'])

elif sys.argv[1]=='L4L23-ntwk':

    if sys.argv[-1]=='analysis':
        fig, ax = ge.figure(figsize=(.95,1.))
        fig2, ax2 = ge.figure(figsize=(.7,1.)) 
        for i, m in enumerate(['V1', 'V2', 'V2-CB1-KO']):
            data = ntwk.recording.load_dict_from_hdf5('data/gain-%s-%s.zip' % (sys.argv[1], m))
            x, y = input_output_analysis(data)
            ax.plot(x, y, label=m)
            # relative gain plot
            ax2.bar([i], [np.polyfit(x, y, 1)[0]/y[0]], color=ge.green)
        ge.legend(ax, size='x-small', frameon=False)
        ge.set_plot(ax, xlabel='L4PN input (Hz)  ', ylabel='L23PN output (Hz)    ', title='L4-L23 network')
        ge.set_plot(ax2, xticks=range(3), xticks_labels=['V1', 'V2', 'V2-CB1-KO'], xticks_rotation=70, ylabel='rel. gain (Hz$^{-1}$)\n(gain/spont-level)')
        fig2.savefig('doc/L4L23-rel-gain.svg')
        fig.savefig('doc/L4L23-ntwk-gain.svg')
        ge.show()
    else:
        Model = update_model(Model, sys.argv[2])
        Model['filename'] = 'data/gain-%s-%s.zip' % (sys.argv[1], sys.argv[2])
        running_sim_func(Model,
                         #max_input=10 if sys.argv[2]!='V2' else 8,
                         max_input=8,
                         rec_pops=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                         aff_pops=['AffExcBG', 'AffExcTV'])

elif sys.argv[1]=='L4-ntwk':

    if sys.argv[-1]=='analysis':
        fig, ax = ge.figure(figsize=(.95,1.))
        for m in ['V1', 'V2', 'V2-CB1-KO']:
            data = ntwk.recording.load_dict_from_hdf5('data/gain-L4L23-ntwk-%s.zip' % m)
            x, y = input_output_analysis(data, pop='L4Exc')
            ax.plot(x, y, label=m)
        ge.legend(ax, size='x-small', frameon=False)
        ge.set_plot(ax, xlabel='L4PN input (Hz)  ', ylabel='L4PN output (Hz)    ', title='L4-L23 network')
        fig.savefig('doc/L4-ntwk-gain.svg')
        ge.show()


elif sys.argv[1]=='analysis':

    fig, AX = ge.figure(axes=(8,1),
                        figsize=(.8,.8), wspace=1.2, left=1.4, bottom=1.1, top=1.5)
    
    for c, cond, color in zip(range(3), ['V1', 'V2', 'V2-CB1-KO'], [ge.blue, ge.red, ge.green]):
        Model2 = {'data_folder': './data/', 'zip_filename':'data/gain-scan-%s.zip' % cond}
        Model2, PARAMS_SCAN, _ = ntwk.scan.get(Model2, filenames_only=True)

        ge.annotate(AX[0], c*20*' '+cond, (.2,.99), xycoords='figure fraction', va='top', color=color)
        for i, filename in enumerate(PARAMS_SCAN['FILENAMES']):
            data = ntwk.recording.load_dict_from_hdf5(filename)
            x, y = input_output_analysis(data)
            
            # AX[i%8][int(i/8)].plot(x, y, '-', lw=1, color=color)
            AX[i].plot(x, y, '-', lw=1, color=color)
            if cond=='V2-CB1-KO':
                # ge.set_plot(AX[i%8][int(i/8)])
                # ge.annotate(AX[i%8][int(i/8)],
                ge.set_plot(AX[i])
                ge.annotate(AX[i],
                            'p$_{L4-L23PN}$=%.2f\np$_{L4-Inh}$=%.2f' % (data['p_L4Exc_L23Exc'], data['p_L4Exc_Inh']),
                            (1,1), ha='right', va='top', size='xx-small')
                
    # for i in range(8):
    #     ge.set_plot(AX[i][0], ylabel='$\delta$ rate (Hz)')
    #     ge.set_plot(AX[7][i], xlabel='input (Hz)')
    ge.set_plot(AX[0], xlabel='input (Hz)', ylabel='$\delta$ rate (Hz)')

    # fig.savefig('doc/gain-comparison-various-psyn.png')
    ge.show()

        
    
elif sys.argv[1] in ['V1', 'V2', 'V2-CB1-KO']:

    # means scan, possible options:
    # "", "fix-missing", "with-repeat" or "fix-missing-with-repeat"

    Model = update_model(Model, sys.argv[1])

    Model['data_folder'] = './data/'
    Model['zip_filename'] = 'data/gain-scan-%s.zip' % sys.argv[1]
    
    KEYS = ['p_L4Exc_L23Exc', 'p_L4Exc_Inh']
    pconn = np.linspace(0.005, 0.15, 8)
    
    VALUES = [pconn, [0]]
    ntwk.scan.run(Model, KEYS, VALUES, running_sim_func,
                  fix_missing_only=('fix-missing' in sys.argv[2]),
                  parallelize=True)

    if 'with-repeat' in sys.argv[2]:
        # "scan-with-repeat" or "scan-fix-missing-with-repeat"
        for i in range(5):
            ntwk.scan.run(Model,  KEYS, VALUES, running_sim_func,
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
                   filename='data/L4-gain.h5')
    
elif sys.argv[-1]=='test-analysis':

    data = ntwk.recording.load_dict_from_hdf5('data/L4-gain.h5')

    x, y = input_output_analysis(data)
    fig, ax = ge.plot(x, y, xlabel='input to L4 (Hz)', ylabel='L23 PN rate (Hz)')
    fig.savefig('fig.png')
    
    fig, _ = ntwk.plots.activity_plots(data, subsampling=20)
    fig.savefig('fig_raw.png')
    
    ge.show()

else:
    print('need args')    
