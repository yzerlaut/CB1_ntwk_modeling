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
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom,
                            xlabel='time (s)', yscale=('log' if with_log_scale_for_act else 'lin'))
            elif i==0:
                ge.set_plot(AX[a][i], (['left'] if '(' in LABELS[a] else []),
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom)
            else:
                ge.set_plot(AX[a][i], [], ylim=ylim, xlim=tzoom)
                
    return fig, AX

def raw_data_fig_multiple_sim_with_zoom(FILES,
                                        LABELS=None,
                                        POP_KEYS=['L4Exc', 'L23Exc', 'PvInh', 'CB1Inh'],
                                        POP_COLORS=[ge.blue, ge.green, ge.red, ge.orange],
                                        tzoom=[200,7000],
                                        tzoom2=[200,300],
                                        subsampling=1,
                                        with_log_scale_for_act=True,
                                        min_pop_act_for_log=0.,
                                        verbose=False):

    POPS = ntwk.plots.find_pop_keys(ntwk.recording.load_dict_from_hdf5(FILES[0]))
    data0 = ntwk.recording.load_dict_from_hdf5(FILES[0])

    tzoom = [np.max([tzoom[0], 0]), np.min([data0['tstop'], tzoom[1]])]
    
    if len(POP_KEYS)!=len(POPS):
        POP_KEYS = POPS
        POP_COLORS = ge.colors[:len(POPS)]
        print('forcing pops to: ', POPS)
    
    ROW_LENGTHS = [1, 2, 3, 2]
    ROW_LABELS = ['input (Hz)', 'spike raster', '', 'pop. act. (Hz)']
    AXES_EXTENTS = []

    LABELS = ['input (Hz)', 'spike raster', '$V_m$', 'rate (Hz)']
    for row_length in ROW_LENGTHS:
        axes_per_column = []
        for i in range(len(FILES)):
            axes_per_column.append([10, row_length])
            axes_per_column.append([1, row_length])
            axes_per_column.append([12, row_length])
            axes_per_column.append([3, row_length])
                       
        AXES_EXTENTS.append(axes_per_column)

    fig, AX = ge.figure(axes_extents=AXES_EXTENTS,
                        figsize=(.5,.8), wspace=0.1, hspace=0.1, left=0.7, reshape_axes=False)


    for i, f in enumerate(FILES):
        
        data = ntwk.recording.load_dict_from_hdf5(f)

        if min_pop_act_for_log>0:
            for key in POP_KEYS:
                data['POP_ACT_%s'%key] += min_pop_act_for_log
        
        # -- full view
        ntwk.plots.input_rate_subplot(data, AX[0][4*i+2],
                                      ['AffExcTV', 'AffExcBG'],
                                      [ge.brown, 'k'], tzoom, ge)
        
        ntwk.plots.raster_subplot(data, AX[1][4*i+2], POP_KEYS, POP_COLORS, tzoom, ge, subsampling=subsampling)


        ntwk.plots.few_Vm_plot(data, ax=AX[2][4*i+2],
                               POP_KEYS=POP_KEYS[::-1],
                               COLORS=POP_COLORS[::-1], graph_env=ge,
                               tzoom=tzoom, shift=25, vpeak=-45)
        
        
        ntwk.plots.population_activity_subplot(data, AX[3][4*i+2], POP_KEYS, POP_COLORS, tzoom, ge,
                                               with_smoothing=10, with_log_scale=with_log_scale_for_act)
        

        # -- zoomed view
        ntwk.plots.input_rate_subplot(data, AX[0][4*i],
                                      ['AffExcTV', 'AffExcBG'],
                                      [ge.brown, 'k'], tzoom2, ge)
        
        ntwk.plots.raster_subplot(data, AX[1][4*i], POP_KEYS, POP_COLORS, tzoom2, ge, subsampling=subsampling)


        ntwk.plots.few_Vm_plot(data, ax=AX[2][4*i],
                               POP_KEYS=POP_KEYS[::-1],
                               COLORS=POP_COLORS[::-1], graph_env=ge,
                               tzoom=tzoom2, shift=25, vpeak=-45)
        
        
        ntwk.plots.population_activity_subplot(data, AX[3][4*i], POP_KEYS, POP_COLORS, tzoom2, ge,
                                               with_smoothing=10, with_log_scale=with_log_scale_for_act)

        for j in [1,3]:
            for a in range(len(AX)):
                AX[a][4*i+j].axis('off')
                

    for a in range(len(AX)):
        ylim = [np.inf, -np.inf]
        for i in range(len(FILES)):
            ylim[0] = np.min([ylim[0], AX[a][4*i].get_ylim()[0], AX[a][4*i+2].get_ylim()[0]])
            ylim[1] = np.max([ylim[1], AX[a][4*i].get_ylim()[1], AX[a][4*i+2].get_ylim()[1]])
        for i in range(len(FILES)):
            if i==0 and a==(len(AX)-1):
                ge.set_plot(AX[a][4*i], ['left'],
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom2,
                            yscale=('log' if with_log_scale_for_act else 'lin'))
            elif i==0:
                ge.set_plot(AX[a][4*i], (['left'] if '(' in LABELS[a] else []),
                            ylim=ylim, ylabel=LABELS[a], xlim=tzoom2)
            else:
                ge.set_plot(AX[a][4*i], [], ylim=ylim, xlim=tzoom2)
            ge.set_plot(AX[a][4*i+2], [], ylim=ylim, xlim=tzoom)
            
            if a==0:
                AX[a][4*i].fill_between(tzoom2, ylim[0]*np.ones(2), ylim[1]*np.ones(2), color='gray', alpha=.2, lw=0)
                AX[a][4*i+2].fill_between(tzoom2, ylim[0]*np.ones(2), ylim[1]*np.ones(2), color='gray', alpha=.2, lw=0)
            
    ge.show()
    return fig, AX


if __name__=='__main__':

    import os
    FILES = [('data/CB1_ntwk_model-%s.h5' % cond) for cond in ['V1','V2']\
             if os.path.isfile('data/CB1_ntwk_model-%s.h5' % cond)]
    
    if sys.argv[-1]=='raw':
        fig, AX = raw_data_fig_multiple_sim_with_zoom(FILES,
                                                      min_pop_act_for_log=0.1)
        # ge.save_on_desktop(fig, 'fig.png')
        
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
        
