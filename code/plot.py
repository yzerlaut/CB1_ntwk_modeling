import sys

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
import ntwk

def raw_data_fig_multiple_sim(FILES,
                              LABELS=None,
                              POP_KEYS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                              POP_COLORS=[ge.blue, ge.green, ge.red, ge.orange],
                              tzoom=[200,7000],
                              subsampling=1,
                              with_log_scale_for_act=False,
                              verbose=False):

    POPS = ntwk.plots.find_pop_keys(ntwk.recording.load_dict_from_hdf5(FILES[0]))
    data0 = ntwk.recording.load_dict_from_hdf5(FILES[0])

    tzoom = [np.max([tzoom[0], 0]), np.min([data0['tstop'], tzoom[1]])]
    
    if len(POP_KEYS)!=len(POPS):
        POP_KEYS = POPS
        POP_COLORS = ge.colors[:len(POPS)]
        print('forcing pops to: ', POPS)
    
    ROW_LENGTHS = [1,3]+list(np.ones(len(POPS), dtype=int))+[1]
    ROW_LABELS = ['input (Hz)', 'spike raster']+POPS+['pop. act. (Hz)']
    AXES_EXTENTS = []

    LABELS = ['input (Hz)', 'spike raster']+['(mV)' for p in POP_KEYS]+['rate (Hz)']
    for row_length in [1, 3]+[1 for p in POP_KEYS]+[2]:
        AXES_EXTENTS.append([[1, row_length] for i in range(len(FILES))])

    fig, AX = ge.figure(axes_extents=AXES_EXTENTS,
                        figsize=(2,.8), wspace=0.1, hspace=0.1, left=0.7, reshape_axes=False)

    for i, f in enumerate(FILES):
        data = ntwk.recording.load_dict_from_hdf5(f)

        ntwk.plots.input_rate_subplot(data, AX[0][i],
                                      ['AffExcTV', 'AffExcBG'],
                                      [ge.brown, 'k'], tzoom, ge)
        
        ntwk.plots.raster_subplot(data, AX[1][i], POP_KEYS, POP_COLORS, tzoom, ge, subsampling=subsampling)

        
        ntwk.plots.population_activity_subplot(data, AX[-1][i], POP_KEYS, POP_COLORS, tzoom, ge,
                                               with_smoothing=10, with_log_scale=with_log_scale_for_act)
        if verbose:
            t = np.arange(int(data['tstop']/data['dt']))*data['dt']
            print(' ---- mean firing rates ----')
            t_cond = (t>tzoom[0]) # discarding transient period for the dynamics
            for key in POP_KEYS:
                print(' - %s: %.2fHz' % (key, np.mean(data['POP_ACT_%s'%key][t_cond])))

        ntwk.plots.Vm_subplots_mean_with_single_trace(data,
                                                      [AX[p][i] for p in range(2, len(POP_KEYS)+2)],
                                                      POP_KEYS, POP_COLORS, tzoom, ge, clip_spikes=True)

    for a in range(len(AX)):
        ylim = [np.inf, -np.inf]
        for i in range(len(FILES)):
            ylim[0] = np.min([ylim[0], AX[a][i].get_ylim()[0]])
            ylim[1] = np.max([ylim[1], AX[a][i].get_ylim()[1]])
        for i in range(len(FILES)):
            if i==0 and a==(len(AX)-1):
                ge.set_plot(AX[a][i], (['left', 'bottom'] if '(' in LABELS[a] else ['bottom']),
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom, xlabel='time (s)', yscale=('log' if with_log_scale_for_act else 'lin'))
            elif i==0:
                ge.set_plot(AX[a][i], (['left'] if '(' in LABELS[a] else []),
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom)
            else:
                ge.set_plot(AX[a][i], [], ylim=ylim, xlim=tzoom)
                
    return fig, AX


if __name__=='__main__':


    FILES = ['data/model-with-thal-V1.h5',
             'data/model-with-thal-V2.h5',
             'data/model-with-thal-V2-CB1-KO.h5']


    if sys.argv[-1]=='raw':
        fig, AX = raw_data_fig_multiple_sim(FILES)
        ge.save_on_desktop(fig, 'fig.png')
        
    elif sys.argv[-1] in ['syn', 'connec', 'matrix']:
        for i, f in enumerate(FILES[:1]):
            data = ntwk.recording.load_dict_from_hdf5(f)
            fig, _, _ = ntwk.plots.connectivity_matrix(data,
                                                       REC_POPS=['L23Exc', 'PvInh', 'CB1Inh', 'L4Exc'],
                                                       AFF_POPS=['AffExcBG', 'AffExcTV'],
                                                       blank_zero=True,
                                                       graph_env=ge)
            # ge.save_on_desktop(fig, 'fig.png')
            ge.show()
    else:
        FILES = sys.argv[1:]
        fig, AX = raw_data_fig_multiple_sim(FILES)
        ge.show()
        
