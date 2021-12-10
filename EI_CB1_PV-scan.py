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

        synch, meanFR, stdFR = [np.zeros(len(DATA)) for i in range(3)]
        for i, data in enumerate(DATA):
            meanFR[i] = ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)
            stdFR[i] = ntwk.analysis.get_std_pop_act(data, pop='Exc', tdiscard=200)
            synch[i] = ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                              Tbin=100, Nmax_pairs=2000)

        Naff = len(np.unique(PARAMS_SCAN['F_AffExc'])]
        
        fig, AX = ge.figure(axes=(3, Naff), top=3)
        
        
        ge.twoD_plot(PARAMS_SCAN['CB1_PV_ratio'], PARAMS_SCAN['inh_exc_ratio'], meanFR,
                     ax=AX[0], vmin=meanFR.min(), vmax=meanFR.max())
        ticks = [meanFR.min(), .5*(meanFR.min()+meanFR.max()), meanFR.max()]
        ge.bar_legend(AX[0], continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                      ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                      bounds=[meanFR.min(), meanFR.max()],
                      label='exc. rate (Hz)')

        ge.twoD_plot(PARAMS_SCAN['CB1_PV_ratio'], PARAMS_SCAN['inh_exc_ratio'], stdFR,
                     ax=AX[1], vmin=stdFR.min(), vmax=stdFR.max())
        ticks = [stdFR.min(), .5*(stdFR.min()+stdFR.max()), stdFR.max()]
        ge.bar_legend(AX[1], continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                      ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                      bounds=[stdFR.min(), stdFR.max()],
                      label='exc. rate std (Hz)')
        
        ge.twoD_plot(PARAMS_SCAN['CB1_PV_ratio'], PARAMS_SCAN['inh_exc_ratio'], synch,
                     ax=AX[2], vmin=synch.min(), vmax=synch.max())
        ticks = 100*np.array([synch.min(), .5*(synch.min()+synch.max()), synch.max()])

        ge.bar_legend(AX[2], continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                      ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                      bounds=[synch.min(), synch.max()],
                      label='synch (proba x100)')

        for ax in AX:
            ge.set_plot(ax, xlabel='CB1/PV ratio', ylabel='Inh/Exc ratio')
        ge.show()
        
    else:

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/exc-inh-CB1-PV-scan.zip'

        ntwk.scan.run(Model,
                      # ['inh_exc_ratio', 'CB1_PV_ratio'],
                      ['F_AffExc', 'inh_exc_ratio', 'CB1_PV_ratio'],
                      [np.linspace(4., 7., 3), np.linspace(0.2,0.4,3), np.linspace(0.05,0.95,3)],
                      running_sim_func,
                      parallelize=False)
        

