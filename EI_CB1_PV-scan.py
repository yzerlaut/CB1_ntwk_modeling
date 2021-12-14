from Model import *
from datavyz import ge

def running_sim_func(Model, a=0):
    run_single_sim(Model, filename=Model['filename'])

def data_analysis(zip_filename, data_folder):

    Model = {'data_folder': data_folder, 'zip_filename':zip_filename}

    Model, PARAMS_SCAN, DATA = ntwk.scan.get(Model, verbose=False)

    synch, meanFR, stdFR = [np.zeros(len(DATA)) for i in range(3)]
    for i, data in enumerate(DATA):
        meanFR[i] = ntwk.analysis.get_mean_pop_act(data, pop='Exc', tdiscard=200)
        stdFR[i] = ntwk.analysis.get_std_pop_act(data, pop='Exc', tdiscard=200)
        synch[i] = ntwk.analysis.get_synchrony_of_spiking(data, pop='Exc',
                                                          method='STTC',
                                                          Tbin=300, Nmax_pairs=2000)

    data = dict(PARAMS_SCAN=PARAMS_SCAN, synch=synch, meanFR=meanFR, stdFR=stdFR)

    np.save(os.path.join(data_folder, zip_filename.replace('.zip', '.npy')), data)
    
if __name__=='__main__':
    
    if sys.argv[-1]=='plot':
        
        # ######################
        # ## ----- Plot ----- ##
        # ######################

        print('loading data [...]')

        zip_filename = os.path.join('exc-inh-CB1-PV-scan.zip')
        data = np.load(os.path.join('data', zip_filename.replace('.zip', '.npy')), allow_pickle=True).item()
        
        print('plotting data [...]')

        Naff = len(np.unique(data['PARAMS_SCAN']['F_AffExc']))
        
        fig, AX = ge.figure(axes=(3, Naff), top=1, hspace=3.)

        for ia, faff in enumerate(np.unique(data['PARAMS_SCAN']['F_AffExc'])):
            ge.annotate(AX[ia][0], '$\\nu_{aff}$=%.1fHz' % faff, (-.65,0), rotation=90)
            
            aff_cond = (data['PARAMS_SCAN']['F_AffExc']==faff)

            for x, label, ax in zip([data['meanFR'], data['stdFR'], data['synch']],
                                    ['exc. rate (Hz)', 'exc. rate std (Hz)', 'synch.'],
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
        data_analysis('exc-inh-CB1-PV-scan.zip', 'data')
        
    elif sys.argv[-1]=='simulation':

        Model['data_folder'] = 'data/'
        Model['zip_filename'] = 'exc-inh-CB1-PV-scan.zip'

        print('running simulation [...]')
        
        ntwk.scan.run(Model,
                      ['F_AffExc', 'inh_exc_ratio', 'CB1_PV_ratio'],
                      [np.linspace(4., 15., 3), np.linspace(0.1,0.5,3), np.linspace(0.05,0.95,3)],
                      running_sim_func,
                      parallelize=True)
        
    else:
        print('\n Need to provide instruction ...\n')
