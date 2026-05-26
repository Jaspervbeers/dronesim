
# Plot results
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D as mlines
import matplotlib.patches as mpatches
import os
import numpy as np
from funcs.angleFuncs import QuatRot, Eul2Quat

from animation import animate as viz
from animation import drone

# Color palate:
myOrange = '#e67d0a'
myBlue = '#008bb4'
myGreen = 'mediumseagreen'
myYellow = '#ffbe3c'
myRed = 'firebrick'
myGrey = 'gainsboro'
myVelvet = 'mediumvioletred'
myOrangeRed = '#E5340B'
myPurple = 'mediumorchid'


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

def makeBoldLabel(label):
    '''
    Make a label bold
    '''
    split = label.split(' ')
    boldSplit = []
    for s in split:
        boldSplit.append(r'$\mathbf{' + s + r'}$')
    boldLabel = ' '.join(boldSplit)
    return boldLabel


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


def plot_iod(model, simulator, f2iod = None, m2iod = None, convex_iod = False):
    # Forces
    skip = False
    if f2iod is None:
        f2iod = {}
        fmax = 0
        for i, (key, mdl) in enumerate({'Fx':model.FxModel, 
                                        'Fy':model.FyModel, 
                                        'Fz':model.FzModel}.items()):
            iod = model._get_iod(simulator.forces[:, 0, :][:, i], 
                                simulator.state[:, 0, :], 
                                simulator.inputs[:, 0, :], 
                                mdl, convex = convex_iod)
            if iod is None:
                skip = True
                break
            fmax = len(iod) if len(iod) > fmax else fmax
            f2iod.update({key:iod})
    else:
        fmax = np.max([len(iod) for iod in f2iod.values()])
    # Figure
    if not skip:
        figf = plt.figure(figsize=(10, 7))
        gs = figf.add_gridspec(nrows = 3, ncols = fmax)
        axshadow = figf.add_subplot(gs[0, :])
        handles = []
        addLegendPatch(handles, color = myGreen, label = 'Model response', alpha = 0.8)
        addLegendLine(handles, color = myGrey, label = 'Zero-line')
        addLegendPatch(handles, facecolor = 'none', edgecolor = 'k', label = 'iod contour', linewidth = 2)
        lgd = axshadow.legend(handles=handles, fontsize = 14,
                            bbox_to_anchor=(0, 1.2, 1, 0.2), loc="lower left", mode="expand", ncol=2,  labelspacing = 0.8,  borderaxespad=0, handlelength = 3, columnspacing=1)
        lgd._legend_box.align = "left"
        axshadow.axis('off')
        for i, (_MDL, iod_data) in enumerate(f2iod.items()):
            Regressors = iod_data.keys()
            for j, reg in enumerate(Regressors):
                if j == 0:
                    ax = figf.add_subplot(gs[i, j])
                    axRef = ax
                    ax.set_ylabel(makeBoldLabel(f'{_MDL[0]}_{_MDL[1]}'), fontsize = 14)
                else:
                    ax = figf.add_subplot(gs[i, j], sharey = axRef)
                    plt.setp(ax.get_yticklabels(), visible = False)
                ax.scatter(iod_data[reg]['points'][:, 0], iod_data[reg]['points'][:, 1], color = myGreen, alpha = 0.1, s = 1, zorder = 20, rasterized = True)
                # ax.plot(iod_data[reg]['points'][:, 0], iod_data[reg]['points'][:, 1], color = myGreen, alpha = 0.7, zorder = 20)
                ax.plot(iod_data[reg]['hull'][:, 0], iod_data[reg]['hull'][:, 1], color = 'k', linewidth = 2)
                (xmin, xmax) = ax.get_xlim()
                ax.set_xticks([xmin, xmax])
                (ymin, ymax) = ax.get_ylim()
                addVLINE(ax, 0, ymin-10, ymax+10, color = myGrey, zorder = 1)
                if i == 0:
                    ax.set_title(makeBoldLabel(f'Regressor {j+1}'))
                # plotter.prettifyAxis(ax)
                # ax.legend(loc = 'best', labels = [reg])
                hndls = []
                addLegendPatch(hndls, facecolor = 'none', edgecolor = 'none', label = lmapper(reg))
                ax.legend(loc = 'best', handles = hndls)
        figf.supxlabel(makeBoldLabel('Regressor contribution') + ', N', fontsize = 14)
        plt.tight_layout()
    else:
        figf = None
        print('Force IODs are None. Cannot plot.')
    # Moments
    skip = False
    if m2iod is None:
        m2iod = {}
        mmax = 0
        for i, (key, mdl) in enumerate({'Mx':model.MxModel, 
                                        'My':model.MyModel, 
                                        'Mz':model.MzModel}.items()):
            iod = model._get_iod(simulator.moments[:, 0, :][:, i], 
                                simulator.state[:, 0, :], 
                                simulator.inputs[:, 0, :], 
                                mdl, convex = convex_iod)
            if iod is None:
                skip = True
                break
            mmax = len(iod) if len(iod) > mmax else mmax
            m2iod.update({key:iod})
    else:
        mmax = np.max([len(iod) for iod in m2iod.values()])
    if not skip:
        # Figure 
        figm = plt.figure(figsize=(10, 7))
        gs = figm.add_gridspec(nrows = 3, ncols = mmax)
        axshadow = figm.add_subplot(gs[0, :])
        handles = []
        addLegendPatch(handles, color = myGreen, label = 'Model response', alpha = 0.8)
        addLegendLine(handles, color = myGrey, label = 'Zero-line')
        addLegendPatch(handles, facecolor = 'none', edgecolor = 'k', label = 'iod contour', linewidth = 2)
        lgd = axshadow.legend(handles=handles, fontsize = 14,
                            bbox_to_anchor=(0, 1.2, 1, 0.2), loc="lower left", mode="expand", ncol=2,  labelspacing = 0.8,  borderaxespad=0, handlelength = 3, columnspacing=1)
        lgd._legend_box.align = "left"
        axshadow.axis('off')
        for i, (_MDL, iod_data) in enumerate(m2iod.items()):
            Regressors = iod_data.keys()
            for j, reg in enumerate(Regressors):
                if j == 0:
                    ax = figm.add_subplot(gs[i, j])
                    axRef = ax
                    ax.set_ylabel(makeBoldLabel(f'{_MDL[0]}_{_MDL[1]}'), fontsize = 14)
                else:
                    ax = figm.add_subplot(gs[i, j], sharey = axRef)
                    plt.setp(ax.get_yticklabels(), visible = False)
                ax.scatter(iod_data[reg]['points'][:, 0], iod_data[reg]['points'][:, 1], color = myGreen, alpha = 0.1, s = 1, zorder = 20, rasterized = True)
                # ax.plot(iod_data[reg]['points'][:, 0], iod_data[reg]['points'][:, 1], color = myGreen, alpha = 0.7, zorder = 20)
                ax.plot(iod_data[reg]['hull'][:, 0], iod_data[reg]['hull'][:, 1], color = 'k', linewidth = 2)
                (xmin, xmax) = ax.get_xlim()
                ax.set_xticks([xmin, xmax])
                (ymin, ymax) = ax.get_ylim()
                addVLINE(ax, 0, ymin-10, ymax+10, color = myGrey, zorder = 1)
                if i == 0:
                    ax.set_title(makeBoldLabel(f'Regressor {j+1}'))
                # plotter.prettifyAxis(ax)
                hndls = []
                addLegendPatch(hndls, facecolor = 'none', edgecolor = 'none', label = lmapper(reg))
                ax.legend(loc = 'best', handles = hndls)
        figm.supxlabel(makeBoldLabel('Regressor contribution') + ', N', fontsize = 14)
        plt.tight_layout()
    else:
        figm = None
        print('Moment IODs are None. Cannot plot.')
    return figf, figm

def lmapper(reg):
    lreg = reg.replace('(w2_1 + w2_2 + w2_3 + w2_4)', r'\sum_{i=1}^{4}\omega^{2}_{i}')
    lreg = lreg.replace('((|u| + |v|)^(2))*(w^(1.0))', r'{[|u| + |v|]}^{2}*w')
    lreg = lreg.replace(' ', '')
    lreg = lreg.replace('(', '{').replace(')', "}")
    lreg = lreg.replace('w_tot', r'\omega_{tot}')
    lreg = lreg.replace('.0', '')
    lreg = lreg.replace('*', r' \cdot ')
    lreg = lreg.replace('[', '(').replace(']', ')')
    return makeBoldLabel(lreg)


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

    # iod plots
    figf, figm = plot_iod(simulator.simVars['model'], simulator)
    figf.savefig(os.path.join(savePath, 'iod_force.png'), dpi = 600)
    figf.savefig(os.path.join(savePath, 'iod_force.pdf'))
    figm.savefig(os.path.join(savePath, 'iod_moment.png'), dpi = 600)
    figm.savefig(os.path.join(savePath, 'iod_moment.pdf'))