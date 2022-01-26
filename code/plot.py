import sys

import numpy as np

sys.path += ['./datavyz', './neural_network_dynamics', './code']
from datavyz import graph_env_manuscript as ge
import ntwk

COLORS = [ge.green, ge.red, ge.orange, ge.blue]
POPS = ['L23Exc', 'PvInh', 'CB1Inh', 'L4Exc']

def raw_data_fig_3_sim(FILES,
                       tzoom=[0,2000]):

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
        
    
    return fig, AX

if __name__=='__main__':

    fig, AX = raw_data_fig_3_sim(['data/model-with-thal-V1.h5',
                                  'data/model-with-thal-V2.h5'])

    ge.show()
