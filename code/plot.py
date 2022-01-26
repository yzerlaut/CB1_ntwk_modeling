from datavyz import graph_env_manuscript as ge

def raw_data_fig_3_sim():

    AXES_EXTENTS = []
    for row_length in [1,3,4,1]:
        AXES_EXTENTS.append([[1, row_length] for i in range(3)])
    fig, AX = ge.figure(axes_extents=AXES_EXTENTS,
                        figsize=(1.5,.4), wspace=0.1, hspace=0.1, left=0.7)

    for ir, row_label in enumerate(['input (Hz)', 'spike raster', '$Vm$', 'pop. act. (Hz)']):
        ge.set_plot(AX[ir][0], ['left'], ylabel=row_label)
        ge.set_plot(AX[ir][1], [])
        ge.set_plot(AX[ir][2], [])
        
    
    return fig, AX

if __name__=='__main__':

    fig, AX = raw_data_fig_3_sim()

    ge.show()
