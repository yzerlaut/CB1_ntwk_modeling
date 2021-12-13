from Model import *
from datavyz import ges as ge

def running_sim_func(Model, a=0):
    run_single_sim(Model, filename=Model['filename'])

if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        print('loading data [...]')
        Model = {'data_folder': 'data/', 'zip_filename':'data/exc-inh-CB1-PV-scan.zip'}
        Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model, verbose=False)

        print('analyzing data [...]')
        synch, meanFR, stdFR = [np.zeros(len(DATA)) for i in range(3)]
        for i, data in enumerate(DATA):
            meanFR[i] = ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)
            stdFR[i] = ntwk.analysis.get_std_pop_act(data, pop='Exc', tdiscard=200)
            synch[i] = ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                              Tbin=100, Nmax_pairs=2000)
        print('plotting data [...]')

        Naff = len(np.unique(PARAMS_SCAN['F_AffExc']))
        
        fig, AX = ge.figure(axes=(3, Naff), top=1, hspace=3.)

        for ia, faff in enumerate(np.unique(PARAMS_SCAN['F_AffExc'])):
            ge.annotate(AX[ia][0], '$\\nu_{aff}$=%.1fHz' % faff, (-.65,0), rotation=90)
            
            aff_cond = (PARAMS_SCAN['F_AffExc']==faff)

            for x, label, ax in zip([meanFR, stdFR, synch],
                                    ['exc. rate (Hz)', 'exc. rate std (Hz)', 'synch.'],
                                    AX[ia]):
                
                ge.twoD_plot(PARAMS_SCAN['CB1_PV_ratio'][aff_cond], PARAMS_SCAN['inh_exc_ratio'][aff_cond], x[aff_cond],
                             ax=ax, vmin=x[aff_cond].min(), vmax=x[aff_cond].max())
                ticks = [x[aff_cond].min(), .5*(x[aff_cond].min()+x[aff_cond].max()), x[aff_cond].max()]
                ge.bar_legend(ax, continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                              ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                              bounds=[x[aff_cond].min(), x[aff_cond].max()],
                              label=label)
            
            for ax in AX[ia][:]:
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
        

