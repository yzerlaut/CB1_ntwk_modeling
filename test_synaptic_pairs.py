from Model import *
from datavyz import ges as ge


def get_evoked_depol(window, Vevoked):
    return np.min(np.mean(Vevoked, axis=0))-np.mean(np.mean(Vevoked, axis=0)[window<0])

def record_pairs(NTWK, Model,
                 Npairs=2):

    # we manually add the recording of a few connection pairs for the two inhibitory to excitatory connections
    if not 'VMS' in NTWK:
        NTWK['VMS'] = []

    exc_rec, pv_rec, cb1_rec, n = [], [], [], 0
    while (len(exc_rec)!=Npairs) and (n<1000) and (NTWK['connections'][1,0] is not None):
        # PV -> Exc
        index = np.random.choice(len(NTWK['connections'][1,0]['j']),1)[0]
        if (NTWK['connections'][1,0]['j'][index] not in exc_rec) and (NTWK['connections'][1,0]['i'][index] not in pv_rec):
            pv_rec.append(NTWK['connections'][1,0]['i'][index])
            exc_rec.append(NTWK['connections'][1,0]['j'][index])
        n+=1

    n=0
    while len(exc_rec)!=(2*Npairs) and (n<1000) and (NTWK['connections'][2,0] is not None):
        # CB1 -> Exc
        index = np.random.choice(len(NTWK['connections'][2,0]['j']),1)[0]
        if (NTWK['connections'][2,0]['j'][index] not in exc_rec) and (NTWK['connections'][2,0]['i'][index] not in cb1_rec):
            cb1_rec.append(NTWK['connections'][2,0]['i'][index])
            exc_rec.append(NTWK['connections'][2,0]['j'][index])
        n+=1
        
    for pop, irecs in zip(NTWK['POPS'], [exc_rec, pv_rec, cb1_rec]):
        NTWK['VMS'].append(ntwk.StateMonitor(pop, 'V', record=irecs))

def plot_Vm_pairs(data,
                  POP_KEYS = None, COLORS=None, NVMS=None,
                  tzoom=[0, np.inf], subsampling=1,
                  vpeak=-45, lw=1, exc_magnification=1.):

    fig, AX = ge.figure(axes=(2, max([len(data['VMS_PvInh']), len(data['VMS_CB1Inh'])])), figsize=(2,1), reshape_axes=False, right=.1, left=.5, wspace=.5, bottom=0.2, top=0.5)

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    window = -20+np.arange(int(50/data['dt']))*data['dt']

    for i in range(len(data['VMS_PvInh'])):

        VmI = data['VMS_PvInh'][i].flatten()
        VmE = data['VMS_Exc'][i].flatten()

        AX[i][0].plot(t[::subsampling], VmI[::subsampling], color=plt.cm.tab10(3))

        # adding spikes
        tspikes, threshold, Vevoked = *ntwk.plots.find_spikes_from_Vm(t, VmI, data, 'PvInh'), []
        for ts in tspikes:
            AX[i][0].plot([ts, ts], np.array([threshold, vpeak]), '--', color=plt.cm.tab10(3), lw=lw)
            new_V = VmE[t>ts+window[0]][:len(window)]
            if len(new_V)==len(window):
                Vevoked.append(new_V)
        inset = AX[i][0].inset_axes([-0.1, 0.5, 0.1, 0.4])
        ge.annotate(AX[i][0], 'PSP=%.2fmV' % get_evoked_depol(window, Vevoked), (-0.18,0), size='small')
        
        inset.axis('off')
        if len(Vevoked)>0:
            ge.plot(window, np.mean(Vevoked, axis=0), sy=np.std(Vevoked, axis=0), ax=inset)
            ge.draw_bar_scales(inset, Xbar=10, Xbar_label='10ms', Ybar=1, Ybar_label='1mV')
            inset.plot([0,0], inset.get_ylim(), '--', color=plt.cm.tab10(3), lw=0.5)
        # add exc.
        AX[i][0].plot(t[::subsampling], threshold+exc_magnification*(VmE[::subsampling]-VmE.min()), color=plt.cm.tab10(2))
        ge.set_plot(AX[i][0], [], xlim=[t[0], t[-1]])

    for i in range(len(data['VMS_CB1Inh'])):

        VmI = data['VMS_CB1Inh'][i].flatten()
        VmE = data['VMS_Exc'][i+len(data['VMS_PvInh'])].flatten()

        AX[i][1].plot(t[::subsampling], VmI[::subsampling], color=plt.cm.tab10(1))
        
        # adding spikes
        tspikes, threshold, Vevoked = *ntwk.plots.find_spikes_from_Vm(t, VmI, data, 'CB1Inh'), []
        for ts in tspikes:
            AX[i][1].plot([ts, ts], np.array([threshold, vpeak]), '--', color=plt.cm.tab10(1), lw=lw)
            new_V = VmE[t>ts+window[0]][:len(window)]
            if len(new_V)==len(window):
                Vevoked.append(new_V)
        ge.annotate(AX[i][1], 'PSP=%.2fmV' % get_evoked_depol(window, Vevoked), (-0.2,0), size='small')
        inset = AX[i][1].inset_axes([-0.1, 0.5, 0.1, 0.4])
        inset.axis('off')
        if len(Vevoked)>0:
            ge.plot(window, np.mean(Vevoked, axis=0), sy=np.std(Vevoked, axis=0), ax=inset)
            ge.draw_bar_scales(inset, Xbar=10, Xbar_label='10ms', Ybar=0.2, Ybar_label='0.2mV')
            inset.plot([0,0], inset.get_ylim(), '--', color=plt.cm.tab10(1), lw=0.5)
        # add exc.
        AX[i][1].plot(t[::subsampling], threshold+exc_magnification*(VmE[::subsampling]-VmE.min()), color=plt.cm.tab10(2))
        ge.set_plot(AX[i][1], [], xlim=[t[0], t[-1]])
            
    return fig, AX

def reset_connections(Model):
    for key in Model:
        if 'p_' in key:
            Model[key] = 0 # all connections to Zero !
    
if __name__=='__main__':

    if sys.argv[-1]=='plot':

        ###############################
        ###### CB1 only pairs #########
        ###############################
        data = ntwk.recording.load_dict_from_hdf5('./data/pairs_CB1_only.h5')
        fig, AX = plot_Vm_pairs(data, exc_magnification=3.)
        for i in range(np.array(AX, dtype=object).shape[0]):
            AX[i][0].axis('off')
        ge.annotate(AX[0][0], 'no PV inhibition', (0.5,0.5), va='center', ha='center')

        ###############################
        ###### PV only pairs #########
        ###############################
        data = ntwk.recording.load_dict_from_hdf5('./data/pairs_PV_only.h5')
        fig, AX = plot_Vm_pairs(data, exc_magnification=3.)
        for i in range(np.array(AX, dtype=object).shape[0]):
            AX[i][1].axis('off')
        ge.annotate(AX[0][1], 'no CB1 inhibition', (0.5,0.5), va='center', ha='center')
        
        #################################
        ###### full network act #########
        #################################
        data = ntwk.recording.load_dict_from_hdf5('./data/checking_pairs.h5')
        fig, AX = plot_Vm_pairs(data, subsampling=10)
        
        ge.show()
        
    elif sys.argv[-1]=='full':
        
        Model['tstop'] = 30000
        Model['F_AffExc'] = 3.5 # low level of afferent drive
        
        run_single_sim(Model, filename='./data/checking_pairs.h5',
                       specific_record_function=record_pairs, srf_args=dict(Npairs=5))
        
        
    else:
        
        Model['tstop'] = 10000

        ###############################
        ###### CB1 only pairs #########
        ###############################
        print('- CB1 only sim. ')
        Model['N_Exc'], Model['N_PvInh'], Model['N_CB1Inh'], Model['N_AffExc'] = 5, 2, 5, 20
        reset_connections(Model) # reset connections !
        Model['p_CB1Inh_Exc'], Model['p_PvInh_Exc'] = 0.2, 0 # except the 2 Inhibitions on Excitation
        Model['p_AffExc_CB1Inh'], Model['p_AffExc_PvInh'] = 1, 1 # + driving the 2 Inhibitions with Afferent input
        
        run_single_sim(Model, filename='./data/pairs_CB1_only.h5',
                       specific_record_function=record_pairs, srf_args=dict(Npairs=5))

        ###############################
        ###### PV only pairs #########
        ###############################
        print('- PV only sim. ')
        Model['N_Exc'], Model['N_PvInh'], Model['N_CB1Inh'], Model['N_AffExc'] = 5, 5, 2, 20
        reset_connections(Model) # reset connections !
        Model['p_CB1Inh_Exc'], Model['p_PvInh_Exc'] = 0, 0.2 # except the 2 Inhibitions on Excitation
        Model['p_AffExc_CB1Inh'], Model['p_AffExc_PvInh'] = 1, 1 # + driving the 2 Inhibitions with Afferent input
        
        run_single_sim(Model, filename='./data/pairs_PV_only.h5',
                       specific_record_function=record_pairs, srf_args=dict(Npairs=5))
        
        print('Results of the simulation are stored as:', './data/checking_pairs.h5')
        print('--> Run \"python test_synaptic_pairs.py plot\" to plot the results')

    
