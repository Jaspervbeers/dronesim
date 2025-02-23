
# Plot results
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D as mlines
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import scipy.stats as spStats
import os
import numpy as np
from funcs.angleFuncs import QuatRot, Eul2Quat

from animation import animate as viz
from animation import drone

def prettifyAxis(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')


def show():
    plt.show()


def addVLINE(ax, x, ymin, ymax, **kwargs):
    ylim = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ylim)


def addXVSPAN(ax, xmin, xmax, **kwargs):
    xlim = ax.get_xlim()
    ax.axvspan(xmin, xmax, **kwargs)
    ax.set_xlim(xlim)


def addLegendPatch(handles, **patchKwargs):
    handles.append(mpatches.Patch(**patchKwargs))


def addLegendLine(handles, **lineKwargs):
    handles.append(mlines([], [], **lineKwargs))


def makeFig(nrows = 1, ncolumns = 1, returnGridSpec = False, **kwargs):
    fig = plt.figure()
    if returnGridSpec:
        gs = fig.add_gridspec(nrows = nrows, ncols = ncolumns)
        return fig, gs
    else:
        return fig


def addXHSPAN(ax, ymin, ymax, **kwargs):
    ylim = ax.get_ylim()
    ax.axhspan(ymin, ymax, **kwargs)
    ax.set_ylim(ylim)

def addHLINE(ax, y, xmin, xmax, **kwargs):
    xlim = ax.get_xlim()
    ax.hlines(y, xmin, xmax, **kwargs)
    ax.set_xlim(xlim)


def addBoxAroundPoint(ax, point, deltaX = None, deltaY = None, boxAnnotation = None, annotationXY = None, **kwargs):
    if deltaX is None:
        deltaX = 0.25*point[0]
    if deltaY is None:
        deltaY = 0.25*point[1]
    ax.add_patch(mpatches.Rectangle((point[0]-deltaX, point[1]-deltaY), 2*deltaX, 2*deltaY, **kwargs))
    if boxAnnotation is not None:
        if annotationXY is None:
            annotationXY = (point[0] - deltaX, point[1] + deltaY)
        ax.annotate(boxAnnotation, annotationXY, fontsize = 12, verticalalignment = 'top')




def plotEWSTime(simulator, savePath):
    simVars = simulator.simVars
    time = simulator.time
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, simVars['EWS'][:, 0, :], color = '#008bb4')
    ax.set_ylabel(r'$\mathbf{EWS}$ [eRPM]')
    prettifyAxis(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = fig.add_subplot(gs[1, 0], sharex = ax)
    ax.plot(time, simVars['detrended'], color = 'mediumaquamarine')
    ax.set_ylabel(r'$\mathbf{Detrended}$ $\mathbf{EWS}$ [eRPM]')
    prettifyAxis(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = fig.add_subplot(gs[2, 0], sharex = ax)
    ax.plot(time, simVars['AR'], color = '#ffbe3c', label = 'AR')
    ax.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ (Lag = 1 sample) [-]')
    ax.set_xlabel(r'$\mathbf{Time}$ [s]')
    plt.tight_layout()

    fig.savefig(os.path.join(savePath, 'EWS.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'EWS.pdf'))

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, simVars['AR'], color = '#ffbe3c', label = 'AR')
    ax.set_ylabel(r'$\mathbf{Autocorrelation}$ [-]')
    prettifyAxis(ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax = fig.add_subplot(gs[1:, 0], sharex = ax)
    # ax.plot(time, simVars['AR'], color = 'dimgrey', label = 'AR')
    ax.plot(time, simVars['P(LOC | EWS)'], color = 'k', label = 'P(LOC | EWS)', linewidth = 2)
    ax.plot(time, simVars['P(LOC)'], color = '#e67d0a', alpha = 0.7, label = 'P(LOC)')
    ax.plot(time, simVars['P(EWS | LOC)'], color = 'firebrick', alpha = 0.7, label = 'P(EWS | LOC)')
    ax.plot(time, simVars['P(EWS | not LOC)'], color = 'mediumseagreen', alpha = 0.7, label = 'P(EWS | not LOC)')
    ax.set_ylabel(r'$\mathbf{Probability}$ [-]')
    ax.set_xlabel(r'$\mathbf{Time}$ [s]')
    ax.legend()
    prettifyAxis(ax)
    plt.tight_layout()

    fig.savefig(os.path.join(savePath, 'P_LOC_EWS.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'P_LOC_EWS.pdf'))

    return None


def show_Posteriors(simulator, BF, PBNA_total):
    simVars = simulator.simVars
    time = simulator.time
    dt = simVars['dt']
    detectionWindow = simVars['detectionWindow']

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.plot(time, simVars['AR'], color = 'dimgrey')
    ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
    ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
    prettifyAxis(ax1)
    l3, = ax1.plot([], [], color = '#ffbe3c')

    ax2 = fig.add_subplot(gs[:, 2], sharey = ax1)
    mesh = np.linspace(-1, 1, 1000)
    ax2.set_xlim([0, 15])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
    prettifyAxis(ax2)
    kde1 = spStats.gaussian_kde(BF._PBA.trueData, bw_method = 0.25)
    l1, = ax2.plot(kde1(mesh), mesh, color = 'firebrick', label = 'P(EWS | LOC)')
    l2, = ax2.plot(mesh*0, mesh, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
    l4, = ax2.plot(mesh*0, mesh, color = '#ffbe3c')
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.0001)
    plt.show(block=False)
    step = int((1/dt)/60)
    start = simVars['minSamples'] + 1
    for i, (t, PBNAt) in enumerate(zip(time[start::step], PBNA_total[::step])):
        runningT = time[start - detectionWindow:i*step + start]
        runningAR = simVars['AR'].reshape(-1)[start-detectionWindow:i*step + start]
        l3.set_xdata(runningT)
        l3.set_ydata(runningAR)
        kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
        l4.set_xdata(kdeAR(mesh))
        kde2 = spStats.gaussian_kde(PBNAt, bw_method = 0.25)
        l2.set_xdata(kde2(mesh))
        handles = []
        handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(t)))
        handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
        handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
        handles.append(mlines([], [], color = '#ffbe3c', label = 'AR'))
        ax2.legend(handles = handles)
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.0001)

    return None


def plot_Posteriors(simulator, savePath, BF, PBNA_total):

    def _update(frame, time, PBNA_total):
        # NOTE: START IS NOT SCALED BY STEP YET!!!
        ax1.clear()
        ax2.clear()
        runningT = time[start - detectionWindow:frame*step + start]
        runningAR = simVars['AR'].reshape(-1)[start-detectionWindow:frame*step + start]
        ax1.plot(time, simVars['AR'], color = 'dimgrey')
        ax1.plot(runningT, runningAR, color = '#ffbe3c')
        kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
        ax2.plot(kdeAR(mesh), mesh, color = '#ffbe3c')
        ax2.fill_betweenx(mesh, mesh*0, kdeAR(mesh), color = '#ffbe3c', alpha = 0.2)
        kde2 = spStats.gaussian_kde(PBNA_total[frame*step], bw_method = 0.25)
        ax2.plot(kde2(mesh), mesh, color = 'mediumaquamarine')
        ax2.fill_betweenx(mesh, mesh*0, kde1(mesh), color = 'firebrick', alpha = 0.2)
        ax2.plot(kde1(mesh), mesh, color = 'firebrick', label = 'P(EWS | LOC)')
        ax2.fill_betweenx(mesh, mesh*0, kde2(mesh), color = 'mediumaquamarine', alpha = 0.2)
        plt.setp(ax2.get_yticklabels(), visible = False)
        handles = []
        handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(time[start + frame*step])))
        handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
        handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
        handles.append(mlines([], [], color = '#ffbe3c', label = 'AR Distribution'))
        ax2.legend(handles = handles)
        prettifyAxis(ax1)
        prettifyAxis(ax2)
        ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
        ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
        ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)

    simVars = simulator.simVars
    time = simulator.time
    dt = simVars['dt']
    detectionWindow = simVars['detectionWindow']

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.plot(time, simVars['AR'], color = 'dimgrey')
    ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
    ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
    prettifyAxis(ax1)
    ax1.plot([], [], color = '#ffbe3c')

    ax2 = fig.add_subplot(gs[:, 2], sharey = ax1)
    mesh = np.linspace(-1, 1, 1000)
    ax2.set_xlim([0, 15])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
    prettifyAxis(ax2)
    kde1 = spStats.gaussian_kde(BF._PBA.trueData, bw_method = 0.25)
    ax2.plot(kde1(mesh), mesh, color = 'firebrick', label = 'P(EWS | LOC)')
    ax2.plot(mesh*0, mesh, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
    ax2.plot(mesh*0, mesh, color = '#ffbe3c')

    fps = 30
    step = int((1/dt)/fps)
    start = simVars['minSamples'] + 1
    N = len(time) - start

    ani = FuncAnimation(fig, _update, fargs=(time, PBNA_total), 
                    frames = int(N/step), 
                    save_count=int(N/step), repeat = False)
    # plt.show(block=False)
    ani.save(os.path.join(savePath, 'posterior_Evolution.mp4'), writer = 'ffmpeg', fps = fps/2, dpi = 300)

    return None


def show_LocalBF(simulator, PBA_local, PBNA_local):
    simVars = simulator.simVars
    time = simulator.time
    dt = simVars['dt']
    detectionWindow = simVars['detectionWindow']

    fig = plt.figure(figsize=(10, 5.5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.plot(time, simVars['AR'], color = 'dimgrey')
    ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
    ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
    prettifyAxis(ax1)
    l3, = ax1.plot([], [], color = '#ffbe3c')

    ax2 = fig.add_subplot(gs[:, 2], sharey = ax1)
    mesh = np.linspace(np.nanmin(simVars['AR']), 1, 1000)
    ax2.set_xlim([0, 30])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
    prettifyAxis(ax2)
    l1, = ax2.plot(mesh*0, mesh, color = 'firebrick', label = 'P(EWS | LOC)')
    l2, = ax2.plot(mesh*0, mesh, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
    l4, = ax2.plot(mesh*0, mesh, color = '#ffbe3c')
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.0001)
    plt.show(block=False)
    step = int((1/dt)/60)
    start = simVars['minSamples'] + 1
    for i, (t, PBAn, PBNAn) in enumerate(zip(time[start::step], PBA_local[::step], PBNA_local[::step])):
        runningT = time[start + i*step - detectionWindow:i*step + start]
        runningAR = simVars['AR'].reshape(-1)[start + i*step - detectionWindow:i*step + start]
        l3.set_xdata(runningT)
        l3.set_ydata(runningAR)
        kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
        l4.set_xdata(kdeAR(mesh))
        if len(PBAn) >= 5:
            kde1 = spStats.gaussian_kde(PBAn, bw_method = 0.25)
            l1.set_xdata(kde1(mesh))
        else:
            l1.set_xdata(mesh*0)
        if len(PBNAn) >= 5:
            kde2 = spStats.gaussian_kde(PBNAn, bw_method = 0.25)
            l2.set_xdata(kde2(mesh))
        else:
            l2.set_xdata(mesh*0)
        handles = []
        handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(t)))
        handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
        handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
        handles.append(mlines([], [], color = '#ffbe3c', label = 'AR'))
        ax2.legend(handles = handles)
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.0001)


def plot_LocalBF(simulator, savePath, PBA_local, PBNA_local):

    def update(frame):
        ax1.clear()
        ax2.clear()
        PBAn = PBA_local[frame*step]
        PBNAn = PBNA_local[frame*step]
        runningT = time[start + frame*step - detectionWindow:frame*step + start]
        runningAR = simVars['AR'].reshape(-1)[start + frame*step - detectionWindow:frame*step + start]
        ax1.plot(time, simVars['AR'], color = 'dimgrey')
        ax1.plot(runningT, runningAR, color = '#ffbe3c')
        kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
        ax2.plot(kdeAR(mesh), mesh, color = '#ffbe3c')
        ax2.fill_betweenx(mesh, mesh*0, kdeAR(mesh), color = '#ffbe3c', alpha = 0.2)
        if len(PBAn) >= inferenceMinSamples:
            kde1 = spStats.gaussian_kde(PBAn, bw_method = 0.25)
            ax2.plot(kde1(mesh), mesh, color = 'firebrick')
            ax2.fill_betweenx(mesh, mesh*0, kde1(mesh), color = 'firebrick', alpha = 0.2)
        else:
            ax2.plot(mesh*0, mesh, color = 'firebrick')
        if len(PBNAn) >= inferenceMinSamples:
            kde2 = spStats.gaussian_kde(PBNAn, bw_method = 0.25)
            ax2.plot(kde2(mesh), mesh, color = 'mediumaquamarine')
            ax2.fill_betweenx(mesh, mesh*0, kde2(mesh), color = 'mediumaquamarine', alpha = 0.2)
        else:
            ax2.plot(mesh*0, mesh, color = 'mediumaquamarine')
        plt.setp(ax2.get_yticklabels(), visible = False)
        handles = []
        handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(time[start + frame*step])))
        handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
        handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
        handles.append(mlines([], [], color = '#ffbe3c', label = 'AR Distribution'))
        ax2.legend(handles = handles)
        prettifyAxis(ax1)
        prettifyAxis(ax2)
        ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
        ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
        ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
        ax2.set_xlim([0, 30])
        plt.tight_layout()

    simVars = simulator.simVars
    time = simulator.time
    dt = simVars['dt']
    detectionWindow = simVars['detectionWindow']
    inferenceMinSamples = simVars['inferenceMinSamples']

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[:, :2])
    ax1.plot(time, simVars['AR'], color = 'dimgrey')
    ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
    ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
    prettifyAxis(ax1)
    ax1.plot([], [], color = '#ffbe3c')

    ax2 = fig.add_subplot(gs[:, 2], sharey = ax1)
    mesh = np.linspace(np.nanmin(simVars['AR']), 1, 1000)
    ax2.set_xlim([0, 30])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
    prettifyAxis(ax2)
    ax2.plot(mesh*0, mesh, color = 'firebrick', label = 'P(EWS | LOC)')
    ax2.plot(mesh*0, mesh, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
    ax2.plot(mesh*0, mesh, color = '#ffbe3c')
    plt.tight_layout()

    fps = 30
    step = int((1/dt)/fps)
    start = simVars['minSamples'] + 1
    N = len(time) - start

    ani = FuncAnimation(fig, update, 
                    frames = int(N/step),
                    save_count=int(N/step), repeat = False)
    # plt.show(block=False)
    ani.save(os.path.join(savePath, 'local_evaluation.mp4'), writer = 'ffmpeg', fps = fps/2, dpi = 300)

    return None


def plotResults(simulator, savePath):
    # Extract results
    state = simulator.state
    stateDerivative = simulator.stateDerivative
    forces = simulator.forces
    moments = simulator.moments
    cmdRotorSpeeds = simulator.inputs_CMD
    rotorSpeeds = simulator.inputs
    time = simulator.time
    simVars = simulator.simVars
    reference = simVars['reference']

    # Commanded rotor speeds 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, rotorSpeeds[:, 0, 0], label = 'Rotor 1')
    ax.plot(time, rotorSpeeds[:, 0, 1], label = 'Rotor 2')
    ax.plot(time, rotorSpeeds[:, 0, 2], label = 'Rotor 3')
    ax.plot(time, rotorSpeeds[:, 0, 3], label = 'Rotor 4')
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Rotor \quad speed} \quad [eRPM]$', fontsize=16)
    prettifyAxis(ax)
    ax.legend(loc = 'upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'rotorSpeeds.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'rotorSpeeds.pdf'))

    # Attitude
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, np.unwrap(state[:, 0, 0]), label = r'$True \quad \phi$', color = 'royalblue')
    ax.plot(time, np.unwrap(state[:, 0, 1]), label = r'$True \quad \theta$', color = 'mediumvioletred')
    ax.plot(time, np.unwrap(state[:, 0, 2]), label = r'$True \quad \psi$', color = 'mediumorchid')
    ax.plot(time, reference[:, 0, 0], label = r'$Reference \quad \phi$', color = 'royalblue', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 1], label = r'$Reference \quad \theta$', color = 'mediumvioletred', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 2], label = r'$Reference \quad \psi$', color = 'mediumorchid', linestyle = '--', alpha = 0.8)
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Attitude} \quad [rad]$', fontsize=16)
    prettifyAxis(ax)
    handles = []
    handles.append(mlines([], [], label = 'True attitude'))
    handles.append(mlines([], [], label = 'Reference attitude', linestyle = '--', alpha = 0.8))
    handles.append(mpatches.Patch(color = 'royalblue', label = r'$\phi$'))
    handles.append(mpatches.Patch(color = 'mediumvioletred', label = r'$\theta$'))
    handles.append(mpatches.Patch(color = 'mediumorchid', label = r'$\psi$'))
    ax.legend(handles = handles)
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'attitude.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'attitude.pdf'))


    # Rotational rates
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, state[:, 0, 6], label = 'p', color = 'royalblue')
    ax.plot(time, state[:, 0, 7], label = 'q', color = 'mediumvioletred')
    ax.plot(time, state[:, 0, 8], label = 'r', color = 'mediumorchid')
    ax.plot(time, reference[:, 0, 6], label = r'$Reference \quad p$', color = 'royalblue', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 7], label = r'$Reference \quad q$', color = 'mediumvioletred', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 8], label = r'$Reference \quad r$', color = 'mediumorchid', linestyle = '--', alpha = 0.8)
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Rotational \quad rate} \quad [rad\cdot s^{-1}]$', fontsize=16)
    prettifyAxis(ax)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'rates.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'rates.pdf'))



    # Position
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, state[:, 0, 9], label = r'$True \quad x$', color = 'royalblue')
    ax.plot(time, state[:, 0, 10], label = r'$True \quad y$', color = 'mediumvioletred')
    ax.plot(time, state[:, 0, 11], label = r'$True \quad z$', color = 'mediumorchid')
    ax.plot(time, reference[:, 0, 9], label = r'$Reference \quad x$', color = 'royalblue', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 10], label = r'$Reference \quad y$', color = 'mediumvioletred', linestyle = '--', alpha = 0.8)
    ax.plot(time, reference[:, 0, 11], label = r'$Reference \quad z$', color = 'mediumorchid', linestyle = '--', alpha = 0.8)
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Position} \quad [m]$', fontsize=16)
    prettifyAxis(ax)
    handles = []
    handles.append(mlines([], [], label = 'True attitude'))
    handles.append(mlines([], [], label = 'Reference attitude', linestyle = '--', alpha = 0.8))
    handles.append(mpatches.Patch(color = 'royalblue', label = r'$x$'))
    handles.append(mpatches.Patch(color = 'mediumvioletred', label = r'$y$'))
    handles.append(mpatches.Patch(color = 'mediumorchid', label = r'$z$'))
    ax.legend(handles = handles)
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'position.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'position.pdf'))



    # Velocity (EARTH FRAME!)
    Att_quat = Eul2Quat(state[:, 0, 0:3])
    v_E = QuatRot(Att_quat, state[:, 0, 3:6], rot = 'B2E')
    ref_E = QuatRot(Att_quat, reference[:, 0, 3:6], rot = 'B2E')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, v_E[:, 0], label = r'$u_{E}$', color = 'royalblue')
    ax.plot(time, v_E[:, 1], label = r'$v_{E}$', color = 'mediumvioletred')
    ax.plot(time, v_E[:, 2], label = r'$w_{E}$', color = 'mediumorchid')
    ax.plot(time, ref_E[:, 0], label = r'$Reference \quad u_{E}$', color = 'royalblue', linestyle = '--', alpha = 0.8)
    ax.plot(time, ref_E[:, 1], label = r'$Reference \quad v_{E}$', color = 'mediumvioletred', linestyle = '--', alpha = 0.8)
    ax.plot(time, ref_E[:, 2], label = r'$Reference \quad w_{E}$', color = 'mediumorchid', linestyle = '--', alpha = 0.8)
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Velocity \quad (E-frame)} \quad [m\cdot s^{-1}]$', fontsize=16)
    prettifyAxis(ax)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'velocities.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'velocities.pdf'))


    # Forces
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, forces[:, 0, 0], label = r'$F_{x}$', color = 'royalblue')
    ax.plot(time, forces[:, 0, 1], label = r'$F_{y}$', color = 'mediumvioletred')
    ax.plot(time, forces[:, 0, 2], label = r'$F_{z}$', color = 'mediumorchid')
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Force} \quad [N]$', fontsize=16)
    prettifyAxis(ax)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'forces.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'forces.pdf'))


    # Moments
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(time, moments[:, 0, 0], label = r'$M_{x}$', color = 'royalblue')
    ax.plot(time, moments[:, 0, 1], label = r'$M_{y}$', color = 'mediumvioletred')
    ax.plot(time, moments[:, 0, 2], label = r'$M_{z}$', color = 'mediumorchid')
    ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize=16)
    ax.set_ylabel(r'$\mathbf{Moment} \quad [Nm]$', fontsize=16)
    prettifyAxis(ax)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savePath, 'moments.png'), dpi = 600)
    fig.savefig(os.path.join(savePath, 'moments.pdf'))


def plot_LocalBFWithDrone(simulator, savePath, PBA_local, PBNA_local):

    def update(frame):
        idx = int(frame * step + start)
        ax.clear()
        for actr in actors.values():
            sv = objectsPose[actr.name]
            xyz = sv['state'][idx][:, 9:12]
            q = sv['quat'][idx]
            omega = sv['inputs'][idx]
            if 'ActorColorHistory' in sv.keys():
                actr.actor._setColor(sv['ActorColorHistory'][idx])
            actr.drawBrute(xyz, q, ax, omega = omega)
            h = animator.positionHistory(idx)
            ax.plot(sv['state'][(idx - h):idx, 0, 9], sv['state'][(idx - h):idx, 0, 10], sv['state'][(idx - h):idx, 0, 11], color = '#e67d0a', linewidth = 1)

        ax.set_xlim(*np.sort([xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]))
        ax.set_ylim(*np.sort([xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]))
        ax.set_zlim(*np.sort([xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]))
        ax.set_xlabel(r'$\mathbf{x}$ [m]')
        ax.set_ylabel(r'$\mathbf{y}$ [m]')
        ax.set_zlabel(r'$\mathbf{z}$ [m]')  
        ax.invert_zaxis()
        # self.ax.invert_yaxis()
        # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
        # Below is a work around
        ylim = ax.get_ylim()
        ax.set_yticks(ax.get_yticks())
        ax.set_ylim(ylim[::-1])
        handles = []
        handles.append(mlines([], [], linestyle = 'None', label = 't = {:.4f} [s]'.format(sv["time"][idx])))
        ax.legend(handles = handles, loc = 'upper right')

        ax1.clear()
        ax2.clear()
        PBAn = PBA_local[frame*step]
        PBNAn = PBNA_local[frame*step]
        runningT = time[start + frame*step - detectionWindow:frame*step + start]
        runningAR = simVars['AR'].reshape(-1)[start + frame*step - detectionWindow:frame*step + start]
        ax1.plot(time, simVars['AR'], color = 'dimgrey')
        ax1.plot(runningT, runningAR, color = '#ffbe3c')
        kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
        ax2.plot(kdeAR(mesh), mesh, color = '#ffbe3c')
        ax2.fill_betweenx(mesh, mesh*0, kdeAR(mesh), color = '#ffbe3c', alpha = 0.2)
        if len(PBAn) >= inferenceMinSamples:
            kde1 = spStats.gaussian_kde(PBAn, bw_method = 0.25)
            ax2.plot(kde1(mesh), mesh, color = 'firebrick')
            ax2.fill_betweenx(mesh, mesh*0, kde1(mesh), color = 'firebrick', alpha = 0.2)
        else:
            ax2.plot(mesh*0, mesh, color = 'firebrick')
        if len(PBNAn) >= inferenceMinSamples:
            kde2 = spStats.gaussian_kde(PBNAn, bw_method = 0.25)
            ax2.plot(kde2(mesh), mesh, color = 'mediumaquamarine')
            ax2.fill_betweenx(mesh, mesh*0, kde2(mesh), color = 'mediumaquamarine', alpha = 0.2)
        else:
            ax2.plot(mesh*0, mesh, color = 'mediumaquamarine')
        plt.setp(ax2.get_yticklabels(), visible = False)
        handles = []
        handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(time[start + frame*step])))
        handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
        handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
        handles.append(mlines([], [], color = '#ffbe3c', label = 'AR Distribution'))
        ax2.legend(handles = handles)
        prettifyAxis(ax1)
        prettifyAxis(ax2)
        ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
        ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
        ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
        ax2.set_xlim([0, 30])
        # plt.tight_layout()
        # fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = None, hspace = None)

    simVars = simulator.simVars
    time = simulator.time
    dt = simVars['dt']
    detectionWindow = simVars['detectionWindow']
    objectsPose = {simulator.model.modelID:simulator.simVars}
    actors = simulator.animator.actors
    animator = simulator.animator
    inferenceMinSamples = simVars['inferenceMinSamples']

    # fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    fig = plt.figure(figsize=(16, 9))
    # gs = fig.add_gridspec(1, 10)
    # ax = fig.add_subplot(gs[:, :1], projection = '3d')
    ax = fig.add_subplot(133, projection = '3d')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharey = ax1)

    # ax1 = fig.add_subplot(gs[:, 2:8])
    ax1.plot(time, simVars['AR'], color = 'dimgrey')
    ax1.set_xlabel(r'$\mathbf{Time}$ [s]', fontsize = 16)
    ax1.set_ylabel(r'$\mathbf{Autocorrelation}$ $\mathbf{EWS}$ [-]', fontsize = 16)
    prettifyAxis(ax1)
    ax1.plot([], [], color = '#ffbe3c')

    # ax2 = fig.add_subplot(gs[:, 8:], sharey = ax1)
    mesh = np.linspace(np.nanmin(simVars['AR']), 1, 1000)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel(r'$\mathbf{Density}$ [-]', fontsize = 16)
    prettifyAxis(ax2)
    ax2.plot(mesh*0, mesh, color = 'firebrick', label = 'P(EWS | LOC)')
    ax2.plot(mesh*0, mesh, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
    ax2.plot(mesh*0, mesh, color = '#ffbe3c')

    fps = 60
    step = int((1/dt)/fps)
    start = simVars['minSamples'] + 1
    N = len(time) - start

    ani = FuncAnimation(fig, update, 
                    frames = int(N/step),
                    save_count=int(N/step), repeat = False)
    # plt.show(block=False)
    ani.save(os.path.join(savePath, 'local_evaluation_withDrone.mp4'), writer = 'ffmpeg', fps = fps/2, dpi = 300)

    return None


def plot_ProbabilityBreakdownWithOrientation(simulator, BF, savePath, tProbe1 = None, tProbe2 = None, facecolor1 = "gold", facecolor2 = "gold", ax1YLabel = r'$\mathbf{AR(U_{q})}$'):
    simVars = simulator.simVars

    t = simulator.time
    pos = simulator.state[:, :, 9:]
    quat = simulator.quat
    omega = simulator.inputs

    EWS = simVars['AR']
    _probability = simVars['P(LOC | EWS)']
    _PA = simVars['P(LOC)']
    _PBA = simVars['P(EWS | LOC)']
    _PBNA = simVars['P(EWS | not LOC)']

    if tProbe1 is None:
        tProbe1 = t[int(len(t)*0.1)]
    if tProbe2 is None:
        hits = np.where(_probability >= BF.threshold)[0]
        if len(hits):
            tProbe2 = t[hits[0]-10]
            facecolor2 = 'firebrick'
        else:
            tProbe2 = t[int(len(t)*0.9)]

    anim = viz.animation()
    droneBody = drone.body(model = simulator.model, origin = pos[0, 0, :], rpy = simulator.state[0, 0, :3])
    droneBody.R = 0.1
    droneBody.b = 0.2
    droneBody.rotorArms = []
    droneBody._initRotorArms() # Re-draw
    anim.addActor(droneBody, name = simulator.model.modelID)

    # Compile data into pose object
    myDronePose = {
        'time':t,
        'position':pos,
        'rotation_q':quat,
        'inputs':omega
    }
    objectPoses = {f'{simulator.model.modelID}':myDronePose}

    Im = plt.figure(figsize = (12, 7))
    AspectR = Im.bbox_inches.bounds[2]/Im.bbox_inches.bounds[3]
    gsIm = Im.add_gridspec(4, 3)
    ax1 = Im.add_subplot(gsIm[:2, :2])
    ax2 = Im.add_subplot(gsIm[2:, :2], sharex = ax1)

    ax3 = Im.add_subplot(gsIm[:2, 2], projection = '3d')

    addBoxAroundPoint(ax1, (tProbe1, EWS[np.where(t>=tProbe1)[0][0]]), deltaX = 0.2*AspectR, deltaY = 0.1, boxAnnotation = 'A', facecolor = facecolor1, alpha = 0.25, linewidth = 2)
    addBoxAroundPoint(ax1, (tProbe2, EWS[np.where(t>=tProbe2)[0][0]]), deltaX = 0.2*AspectR, deltaY = 0.1, boxAnnotation = 'B', facecolor = facecolor2, alpha = 0.25, linewidth = 2)
    ax1.plot(t, EWS, color = 'k')
    prettifyAxis(ax1)
    plt.setp(ax1.get_xticklabels(), visible = False)
    ax1.set_ylabel(ax1YLabel, fontsize = 14)
    # Axis 2
    ax2.plot(t, _probability, color = 'firebrick', linewidth = 4, label = 'P(LOC|EWS) (Aggregate)', zorder = 0)
    ax2.plot(t, _PA, color = '#e67d0a', alpha = 0.8, label = 'P(LOC)', linewidth = 2.5, zorder = 2)
    ax2.plot(t, _PBA, color = 'royalblue', alpha = 0.8, label = 'P(EWS|LOC)', linewidth = 2.5, zorder = 3)
    ax2.plot(t, _PBNA, color = 'seagreen', alpha = 0.8, label = 'P(EWS|not LOC)', linewidth = 4, zorder = 1)
    ax2.legend(loc = (0.01, 0.5), fontsize = 9)
    ax2.set_ylabel(r'$\mathbf{Probability}$', fontsize = 14)
    ax2.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 14)
    prettifyAxis(ax2)
    plt.tight_layout() 
    #Axix 3
    probePos = pos[[np.where(t>=tProbe1)[0][0]]]
    axLims = {'x':[-0.25 + probePos[0, 0, 0], 0.25 + probePos[0, 0, 0]], 'y':[-0.25 + probePos[0, 0, 1], 0.25 + probePos[0, 0, 1]], 'z':[-0.25 + probePos[0, 0, 2], 0.25 + probePos[0, 0, 2]]}
    Im = anim.asImage(objectPoses, [tProbe1], parentFig = Im, axisLims = axLims)
    ax3.set_axis_off()
    ax3.annotate('A', (0, 1), xycoords = 'axes fraction', fontsize = 'xx-large', verticalalignment = 'top')
    ax3.set_facecolor(facecolor1)
    ax3.patch.set_alpha(0.1)
    # ax3.set_title('Snapshot A')

    ax4 = Im.add_subplot(gsIm[2:, 2], projection = '3d')
    probePos = pos[[np.where(t>=tProbe2)[0][0]]]
    axLims = {'x':[-0.25 + probePos[0, 0, 0], 0.25 + probePos[0, 0, 0]], 'y':[-0.25 + probePos[0, 0, 1], 0.25 + probePos[0, 0, 1]], 'z':[-0.25 + probePos[0, 0, 2], 0.25 + probePos[0, 0, 2]]}
    Im = anim.asImage(objectPoses, [tProbe2], parentFig = Im, axisLims = axLims)
    ax4.set_axis_off()
    ax4.annotate('B', (0, 1), xycoords = 'axes fraction', fontsize = 'xx-large', verticalalignment = 'top')
    ax4.set_facecolor(facecolor2)
    ax4.patch.set_alpha(0.1)
    # ax4.set_title('Snapshot B', y = -0.01)

    plt.tight_layout()

    Im.savefig(os.path.join(savePath, 'probabilityBreakdownWithOrientation.png'), dpi = 600)
    Im.savefig(os.path.join(savePath, 'probabilityBreakdownWithOrientation.pdf'))

    return None




def plot_ProbabilityBreakdownWithDrone(simulator, savePath, ax1YLabel = r'$\mathbf{AR(U_{q})}$', fps = 30, axLims = None, fixedAxis = False, LoCThreshold = 0.95):
    simVars = simulator.simVars

    t = simulator.time
    pos = simulator.state[:, :, 9:]
    quat = simulator.quat
    omega = simulator.inputs

    ref = simulator.reference

    EWS = simVars['AR']
    _probability = simVars['P(LOC | EWS)']
    _PA = simVars['P(LOC)']
    _PBA = simVars['P(EWS | LOC)']
    _PBNA = simVars['P(EWS | not LOC)']

    if axLims is None:
        if not fixedAxis:
            axLims = {'x':[-2, 2], 'y':[-2, 2], 'z':[-2, 2]}
        else:
            _axLims = [np.nanmin(pos[:, 0, :]), np.nanmax(pos[:, 0, :])]
            axLims = {'x':_axLims, 'y':_axLims, 'z':_axLims}

    anim = viz.animation()
    droneBody = drone.body(model = simulator.model, origin = pos[0, 0, :], rpy = simulator.state[0, 0, :3])
    droneBody.R = 0.1
    droneBody.b = 0.2
    droneBody.rotorArms = []
    droneBody._initRotorArms() # Re-draw
    anim.addActor(droneBody, name = simulator.model.modelID)

    # Compile data into pose object
    myDronePose = {
        'time':t,
        'position':pos,
        'rotation_q':quat,
        'inputs':omega
    }
    objectPoses = {f'{simulator.model.modelID}':myDronePose}

    def animWrapper(frames, objectsPoses, wrapperKwargs):
        # Necessary elements of wrapperKwargs
        AnimationUpdateFactor = wrapperKwargs['AnimationUpdateFactor'] # Added to kwargs by droneviz
        idx = int(frames * AnimationUpdateFactor) # Running index of our data, if we want to plot a single point use: myPoint = myDataArray[idx]
        axes = wrapperKwargs['axes']
        anim = wrapperKwargs['anim']
        axAnim = axes[wrapperKwargs['axAnim_index']]
        if idx > 0:
            # Unpack data
            t = wrapperKwargs['time'][:idx]
            EWS = wrapperKwargs['EWS'][:idx]
            probabilities = wrapperKwargs['probability'][:idx]
            PA = wrapperKwargs['PA'][:idx]
            PBA = wrapperKwargs['PBA'][:idx]
            PBNA = wrapperKwargs['PBNA'][:idx]
            pos = objectsPoses[simulator.model.modelID]['position'][idx]
            if not fixedAxis:
                axLims.update({'x':[axLimsBasis['x'][0]+pos[0, 0], axLimsBasis['x'][1]+pos[0, 0]]})
                axLims.update({'y':[axLimsBasis['y'][0]+pos[0, 1], axLimsBasis['y'][1]+pos[0, 1]]})
                axLims.update({'z':[axLimsBasis['z'][0]+pos[0, 2], axLimsBasis['z'][1]+pos[0, 2]]})
            # Axis 1
            ax1 = axes[0]
            ax1.clear()
            ax1.plot(wrapperKwargs['time'], wrapperKwargs['EWS'], color = 'gainsboro')
            ax1.plot(t, EWS, color = 'k')
            ax1.scatter(t[-1], EWS[-1], color = 'k', marker = 'o')
            prettifyAxis(ax1)
            plt.setp(ax1.get_xticklabels(), visible = False)
            ax1.set_ylabel(ax1YLabel, fontsize = 14)
            # Axis 2
            ax2 = axes[1]
            ax2.clear()
            ax2.plot(t, probabilities, color = 'k', linewidth = 4, label = 'P(LOC|EWS) (Aggregate)', zorder = 0)
            ax2.scatter(t[-1], probabilities[-1], color = 'k', marker = 'o')
            if probabilities[-1] >= LoCThreshold:
                droneBody._setColor('r')
            else:
                droneBody._setColor('k')
            ax2.plot(t, PA, color = '#e67d0a', alpha = 0.8, label = 'P(LOC)', linewidth = 2.5, zorder = 2)
            ax2.scatter(t[-1], PA[-1], color = '#e67d0a', marker = 'o')
            ax2.plot(t, PBA, color = 'royalblue', alpha = 0.8, label = 'P(EWS|LOC)', linewidth = 2.5, zorder = 3)
            ax2.scatter(t[-1], PBA[-1], color = 'royalblue', marker = 'o')
            ax2.plot(t, PBNA, color = 'seagreen', alpha = 0.8, label = 'P(EWS|not LOC)', linewidth = 4, zorder = 1)
            ax2.scatter(t[-1], PBNA[-1], color = 'seagreen', marker = 'o')
            ax2.legend(loc = (0.01, 0.5), fontsize = 9)
            ax2.set_ylabel(r'$\mathbf{Probability}$', fontsize = 14)
            ax2.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 14)
            prettifyAxis(ax2)
            plt.tight_layout() 
            fig.subplots_adjust(right = 0.960)
        # Activate axAnim for animation object. This sets the active axis of the animation object to our desired axis
        anim.ax = axAnim

    # Animation
    fig = plt.figure(figsize = (12, 7))
    gs = fig.add_gridspec(4, 3)
    ax1 = fig.add_subplot(gs[:2, :2])
    ax2 = fig.add_subplot(gs[2:, :2], sharex = ax1)
    ax3 = fig.add_subplot(gs[:, 2], projection = '3d')

    ax1.plot(t, EWS, color = 'gainsboro')

    prettifyAxis(ax1)
    plt.setp(ax1.get_xticklabels(), visible = False)
    ax1.set_ylabel(r'$\mathbf{AR(U_{q})}$', fontsize = 14)
    ax2.set_ylabel(r'$\mathbf{Probability}$', fontsize = 14)
    ax2.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 14)
    prettifyAxis(ax2)
    plt.tight_layout()
    fig.subplots_adjust(right = 0.960)

    wrapperKwargs = {
        'axes':fig.axes,
        'anim':anim,
        'axAnim_index': -1,
        'time':t,
        'probability':_probability,
        'PA':_PA,
        'PBA':_PBA,
        'PBNA':_PBNA,
        'EWS':EWS
    }
    axLimsBasis = {k:v for k,v in axLims.items()}
    animation = anim.animate(t, objectPoses, figure = fig, wrapper = animWrapper, wrapperKwargs = wrapperKwargs, axisLims=axLims)
    anim.saveAnimation(os.path.join(savePath, 'ProbabilityBreakdownWithDrone.mp4'), fps = fps)

    return None

# # Look at regressor contributions 
# modelInputs = model.FzModel.droneGetModelInput(state[:, 0, :], rotorSpeeds[:, 0, :])
# _m = model.FzModel
# # _m = model.FxModel
# # _m = model.FyModel
# # _m = model.MzModel
# allRegressors = [r for r in _m.regressors]
# allCoefficients = [c for c in _m.coefficients.__array__()[:, 0]]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# ax.plot(np.ones(modelInputs.shape[0])*_m.coefficients[0].__array__()[0][0], label = 'bias') # Bias contribution
# for r in range(len(allRegressors)):
#     _m.regressors = allRegressors[:r+1]
#     _m.coefficients = np.matrix(allCoefficients[:r+2]).T
#     ax.plot(_m.predict(modelInputs), label = _m.polynomial[r+1])

# ax.legend()
# ax.set_xlabel(r'$\mathbf{Time}$ [s]')
# ax.set_ylabel(r'$\mathbf{Force}$ $\mathbf{(Moment)}$ [N] ([Nm])')
# # Reset regressors
# _m.regressors = allRegressors
# _m.coefficients = np.matrix(allCoefficients).T
# plt.show()




'''
Old plots
'''

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.plot(time, simVars['AR'], color = 'dimgrey')
# l3, = ax1.plot([], [], color = '#ffbe3c')

# ax2 = fig.add_subplot(212)
# ax2.set_xlim([-1, 1])
# ax2.set_ylim([-0.01, 10])
# mesh = np.linspace(-1, 1, 1000)
# l1, = ax2.plot(mesh, mesh*0, color = 'firebrick', label = 'P(EWS | LOC)')
# l2, = ax2.plot(mesh, mesh*0, color = 'mediumaquamarine', label = 'P(EWS | not LOC)')
# l4, = ax2.plot(mesh, mesh*0, color = '#ffbe3c')
# fig.canvas.draw_idle()
# fig.canvas.start_event_loop(0.0001)
# plt.show(block=False)
# step = int((1/dt)/60)
# start = simVars['minSamples'] + 1
# for i, (t, PBAn, PBNAn) in enumerate(zip(time[start::step], PBA_local[::step], PBNA_local[::step])):
#     runningT = time[start + i*step - detectionWindow:i*step + start]
#     runningAR = simVars['AR'].reshape(-1)[start + i*step - detectionWindow:i*step + start]
#     l3.set_xdata(runningT)
#     l3.set_ydata(runningAR)
#     kdeAR = spStats.gaussian_kde(runningAR, bw_method = 0.25)
#     l4.set_ydata(kdeAR(mesh))
#     if len(PBAn) >= 5:
#         kde1 = spStats.gaussian_kde(PBAn, bw_method = 0.25)
#         l1.set_ydata(kde1(mesh))
#     else:
#         l1.set_ydata(mesh*0)
#     if len(PBNAn) >= 5:
#         kde2 = spStats.gaussian_kde(PBNAn, bw_method = 0.25)
#         l2.set_ydata(kde2(mesh))
#     else:
#         l2.set_ydata(mesh*0)
#     handles = []
#     handles.append(mlines([], [], linestyle='None', label = 't = {:.4f} [s]'.format(t)))
#     handles.append(mlines([], [], color = 'firebrick', label = 'P(EWS | LOC)'))
#     handles.append(mlines([], [], color = 'mediumaquamarine', label = 'P(EWS | not LOC)'))
#     handles.append(mlines([], [], color = '#ffbe3c', label = 'AR'))
#     ax2.legend(handles = handles)
#     fig.canvas.draw_idle()
#     fig.canvas.start_event_loop(0.0001)