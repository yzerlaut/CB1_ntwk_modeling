from Model import *
from datavyz import ge

Model['tstop'] = 10000

def running_sim_func(Model, a=0):
    run_single_sim(Model, filename=Model['filename'])

def data_analysis(zip_filename, data_folder):

    Model = {'data_folder': data_folder, 'zip_filename':os.path.join(data_folder, zip_filename)}

    Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model, verbose=False)

    sttc, meanFR, stdFR = [np.zeros(len(DATA)) for i in range(3)]
    for i, data in enumerate(DATA):
        meanFR[i] = ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)
        stdFR[i] = ntwk.analysis.get_std_pop_act(data, pop='Exc', tdiscard=200)
        sttc[i] = ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                          method='STTC',
                                                          Tbin=300, Nmax_pairs=2000)

    data = dict(PARAMS_SCAN=PARAMS_SCAN, sttc=sttc, meanFR=meanFR, stdFR=stdFR)

    np.save(os.path.join(data_folder, zip_filename.replace('.zip', '.npy')), data)
    
if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        print('loading data [...]')

        zip_filename = os.path.join('exc-inh-CB1-PV-Aff-scan.zip')
        data = np.load(os.path.join('data', zip_filename.replace('.zip', '.npy')), allow_pickle=True).item()
        
        print('plotting data [...]')

        Naff = len(np.unique(data['PARAMS_SCAN']['F_AffExcBG']))
        
        fig, AX = ge.figure(axes=(3, Naff), hspace=3., left=2., top=2)

        for ia, faff in enumerate(np.unique(data['PARAMS_SCAN']['F_AffExcBG'])):
            print(faff)
            ge.annotate(AX[ia][0], '$\\nu_{aff}$=%.1fHz' % faff, (-1,0), rotation=90)
            
            aff_cond = (data['PARAMS_SCAN']['F_AffExcBG']==faff)

            for x, label, ax in zip([data['meanFR'], data['stdFR'], data['sttc']],
                                    ['exc. rate (Hz)', 'exc. rate std (Hz)', 'STTC'],
                                    AX[ia]):
                
                ge.twoD_plot(data['PARAMS_SCAN']['CB1_PV_ratio'][aff_cond], data['PARAMS_SCAN']['inh_exc_ratio'][aff_cond], x[aff_cond],
                             ax=ax, vmin=x[aff_cond].min(), vmax=x[aff_cond].max())
                ticks = [x[aff_cond].min(), .5*(x[aff_cond].min()+x[aff_cond].max()), x[aff_cond].max()]
                ge.bar_legend(ax, continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                              ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                              bounds=[x[aff_cond].min(), x[aff_cond].max()],
                              label=label)
            
            for ax in AX[ia][:]:
                ge.set_plot(ax, xlabel='CB1/PV ratio', ylabel='Inh/Exc ratio')

        ge.show()
        
    elif sys.argv[-1]=='plot-vthre':
        
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        print('loading data [...]')

        zip_filename = os.path.join('exc-inh-CB1-PV-VthreInh-scan.zip')
        data = np.load(os.path.join('data', zip_filename.replace('.zip', '.npy')), allow_pickle=True).item()
        
        print('plotting data [...]')

        Nvthre = len(np.unique(data['PARAMS_SCAN']['common_Vthre_Inh']))
        
        fig, AX = ge.figure(axes=(3, Nvthre), hspace=3., left=2., top=2)

        for ia, faff in enumerate(np.unique(data['PARAMS_SCAN']['common_Vthre_Inh'])):
            print(faff)
            ge.annotate(AX[ia][0], '$V_{thre}^{inh}$=%.1fmV' % faff, (-1,0), rotation=90)
            
            aff_cond = (data['PARAMS_SCAN']['common_Vthre_Inh']==faff)

            for x, label, ax in zip([data['meanFR'], data['stdFR'], data['sttc']],
                                    ['exc. rate (Hz)', 'exc. rate std (Hz)', 'STTC'],
                                    AX[ia]):
                
                ge.twoD_plot(data['PARAMS_SCAN']['CB1_PV_ratio'][aff_cond], data['PARAMS_SCAN']['inh_exc_ratio'][aff_cond], x[aff_cond],
                             ax=ax, vmin=x[aff_cond].min(), vmax=x[aff_cond].max())
                ticks = [x[aff_cond].min(), .5*(x[aff_cond].min()+x[aff_cond].max()), x[aff_cond].max()]
                ge.bar_legend(ax, continuous=True, colorbar_inset=dict(rect=[.1,1.3,.8,.12], facecolor=None), colormap=ge.viridis,
                              ticks = ticks, ticks_labels = ['%.2f' % t for t in ticks], orientation='horizontal', labelpad=4.,
                              bounds=[x[aff_cond].min(), x[aff_cond].max()],
                              label=label)
            
            for ax in AX[ia][:]:
                ge.set_plot(ax, xlabel='CB1/PV ratio', ylabel='Inh/Exc ratio')

        ge.show()
        

    elif sys.argv[-1]=='analysis':

        print('analyzing data [...]')
        data_analysis('exc-inh-CB1-PV-VthreInh-scan.zip', 'data')
        # data_analysis('exc-inh-CB1-PV-Aff-scan.zip', 'data')
        
    elif ('aff' in sys.argv[-1]) or ('Aff' in sys.argv[-1]):

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/exc-inh-CB1-PV-Aff-scan.zip'

        print('running simulation [...]')

        ntwk.scan.run(Model,
                      ['F_AffExcBG', 'inh_exc_ratio', 'CB1_PV_ratio'],
                      [np.linspace(4., 15., 4), np.linspace(0.1,0.5,8), np.linspace(0.05,0.95,8)],
                      running_sim_func,
                      parallelize=True)

    elif ('vthre' in sys.argv[-1]) or ('VthreInh' in sys.argv[-1]):

        Model['data_folder'] = './data/'
        Model['zip_filename'] = 'data/exc-inh-CB1-PV-VthreInh-scan.zip'

        ntwk.scan.run(Model,
                      # ['inh_exc_ratio', 'CB1_PV_ratio'],
                      ['common_Vthre_Inh', 'inh_exc_ratio', 'CB1_PV_ratio'],
                      [np.linspace(-53, -40, 5), np.linspace(0.2,0.4,8), np.linspace(0.05,0.95,8)],
                      running_sim_func,
                      parallelize=True)
        
    else:
        print('\n Need to provide instruction ...\n')
