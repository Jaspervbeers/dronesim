
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