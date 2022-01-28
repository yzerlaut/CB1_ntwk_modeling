import sys

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
import ntwk

COLORS = [ge.green, ge.red, ge.orange, ge.blue]
POPS = ['L23Exc', 'PvInh', 'CB1Inh', 'L4Exc']

def raw_data_fig_multiple_sim(FILES,
                              LABELS=None,
                              POP_KEYS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                              POP_COLORS=[ge.blue, ge.green, ge.red, ge.orange],
                              tzoom=[0,7000],
                              subsampling=1):

    # POPS = ntwk.plots.find_pop_keys(ntwk.recording.load_dict_from_hdf5(FILES[0]))
    # print(POPS)
    
    ROW_LENGTHS = [1,3]+list(np.ones(len(POPS), dtype=int))+[1]
    ROW_LABELS = ['input (Hz)', 'spike raster']+POPS+['pop. act. (Hz)']
    AXES_EXTENTS = []

    for row_length in ROW_LENGTHS:
        AXES_EXTENTS.append([[1, row_length] for i in range(len(FILES))])
        
    fig, AX = ge.figure(axes_extents=AXES_EXTENTS,
                        figsize=(2,.8), wspace=0.1, hspace=0.1, left=0.7)

    for i in range(len(FILES)):
        data = ntwk.recording.load_dict_from_hdf5(FILES[i])

        ntwk.plots.Vm_subplots_mean_with_single_trace(data, [AX[j][i] for j in range(2,6)],
                                                      POPS,
                                                      COLORS, tzoom, ge)
    
    for ir, row_label in enumerate(ROW_LABELS):
        if 'Hz' in row_label:
            ge.set_plot(AX[ir][0], ['left'], ylabel=row_label)
        # else:
        #     ge.set_plot(AX[ir][0], [])
        # for i in range(len(FILES))[1:]:
        #     ge.set_plot(AX[ir][i], [])

    LABELS = ['input (Hz)', 'spike raster']+['(mV)' for p in POP_KEYS]+['rate (Hz)']
    for row_length in [1, 3]+[1 for p in POP_KEYS]+[2]:
        AXES_EXTENTS.append([[1, row_length] for i in range(len(FILES))])

    fig, AX = ge.figure(axes_extents=AXES_EXTENTS,
                        figsize=(2,.8), wspace=0.1, hspace=0.1, left=0.7)

    for i, f in enumerate(FILES):
        data = ntwk.recording.load_dict_from_hdf5(f)

        ntwk.plots.input_rate_subplot(data, AX[0][i],
                                      ['AffExcTV', 'AffExcBG'],
                                      [ge.brown, 'k'], tzoom, ge)
        
        ntwk.plots.raster_subplot(data, AX[1][i], POP_KEYS, POP_COLORS, tzoom, ge)
        

        
        ntwk.plots.population_activity_subplot(data, AX[-1][i], POP_KEYS, POP_COLORS, tzoom, ge,
                                               with_smoothing=10)


        ntwk.plots.Vm_subplots_mean_with_single_trace(data,
                                                      [AX[p][i] for p in range(2, len(POP_KEYS)+2)],
                                                      POP_KEYS, POP_COLORS, tzoom, ge)

    for a in range(len(AX)):
        ylim = [np.inf, -np.inf]
        for i in range(len(FILES)):
            ylim[0] = np.min([ylim[0], AX[a][i].get_ylim()[0]])
            ylim[1] = np.max([ylim[1], AX[a][i].get_ylim()[1]])
        for i in range(len(FILES)):
            if i==0:
                ge.set_plot(AX[a][i], (['left'] if '(' in LABELS[a] else []),
                            ylim=ylim, ylabel=LABELS[a])
            else:
                ge.set_plot(AX[a][i], [], ylim=ylim)
                
    # for ir, row_label in enumerate(['input (Hz)', 'spike raster', '$Vm$', 'pop. act. (Hz)']):
    #     ge.set_plot(AX[ir][0], ['left'], ylabel=row_label)
    #     ge.set_plot(AX[ir][1], [])
    #     ge.set_plot(AX[ir][2], [])

        
    
    return fig, AX

if __name__=='__main__':


    fig, AX = raw_data_fig_3_sim(['data/model-with-thal-V1.h5',
                                  'data/model-with-thal-V2.h5'])

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
