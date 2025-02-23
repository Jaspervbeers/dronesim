from models import droneModel
from controllers import dronePIDController, droneUpsetController_PID, droneNDI, droneNDI_Sandbox, droneINDI_Sandbox, droneINDI_nDes, droneVPV, droneVPV2, droneNDI_V_stability
from actuators import droneRotors
from funcs import droneEOM, integrators, plotting, controlTools
from noiseDisturbance import droneSensorNoise
from animation import animate, drone
import numpy as np
import sim
import os
import control as c

import sys
import matplotlib.pyplot as plt
import dill as pkl


simulatorInstance = 'simResults/simulatorInstance.pkl'

# simulatorInstance = '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_UnityGains/NoNoise/simulatorInstance.pkl'
# simulatorInstance = '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_TunedGains/NoNoise/simulatorInstance.pkl'
# simulatorInstance = '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_OptimizedGains/NoNoise/simulatorInstance.pkl'

# simulatorInstance = '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/ContinuousAdaption/HDBeetle_Hover_UnityGains/NoNoise/simulatorInstance.pkl'
with open(simulatorInstance, 'rb') as f:
    simulator = pkl.load(f)


# Unpack certain elements of the simulator
simVars = simulator.simVars
model = simulator.model
controller = simulator.controller
plt.close('all')


def findBodeCross(x, val):
    valArr = np.zeros(x.shape) + val
    idxClose = np.where(np.isclose(x[1:-1], valArr[1:-1], atol = 5))[0] + 1
    xUpper = x[idxClose + 1]
    xLower = x[idxClose - 1]
    idxCross_1 = np.where((xUpper < val) & (xLower > val))[0]
    idxCross_2 = np.where((xUpper > val) & (xLower < val))[0]
    idxCross = np.hstack((idxClose[idxCross_1], idxClose[idxCross_2]))
    if len(idxCross):
        return True, np.nanmax(idxCross)
    else:
        return False, -1
    


def _plotBode(ax1, ax2, omega, magnitude, phase, showMargins = False, plotKwargs = {}):
    ax1.semilogx(omega, c.mag2db(magnitude), **plotKwargs)
    ax2.semilogx(omega, phase*180/np.pi, **plotKwargs)
    if showMargins:
        cond, idx = findBodeCross(c.mag2db(magnitude), 0)
        if cond:
            ax2.scatter(omega[idx], phase[idx]*180/np.pi, zorder = 100, color = plotKwargs['color'] if 'color' in plotKwargs.keys() else 'gainsboro')
            ax2.vlines(omega[idx], -180, phase[idx]*180/np.pi, linestyle = '--', color = 'k', alpha = 0.7)
        cond, idx = findBodeCross(phase*180/np.pi, -180)
        if cond:
            ax1.scatter(omega[idx], c.mag2db(magnitude[idx]), zorder = 100, color = plotKwargs['color'] if 'color' in plotKwargs.keys() else 'gainsboro')
            ax1.vlines(omega[idx], 0, c.mag2db(magnitude[idx]), linestyle = '--', color = 'k', alpha = 0.7)


PMbased = True
badChannels_in = np.sum(np.abs(np.array(controller.phase_margins_in).reshape(-1, 9)) < simulator.simVars['PM_min'], axis =0)
badChannels_out = np.sum(np.abs(np.array(controller.phase_margins_out).reshape(-1, 9)) < simulator.simVars['PM_min'], axis =0)
# No phase margins violated, so now check S(dc) magnitude
if np.sum([badChannels_in, badChannels_out]) == 0:
    PMbased = False
    badChannels_in = np.sum(np.array(controller.magSi_dc).reshape(-1, 9) > -30, axis =0)
    badChannels_out = np.sum(np.array(controller.magSo_dc).reshape(-1, 9) > -30, axis =0)

badChannelsRanked = np.argsort(badChannels_in + badChannels_out)[::-1]
badInOut = {1:[0, 0], 2:[1, 1], 3:[2, 2]}
badIn = {1:[0, 0], 2:[1, 1], 3:[2, 2]}
badOut = {1:[0, 0], 2:[1, 1], 3:[2, 2]}
for __p in badInOut.keys():
    _idx = badChannelsRanked[__p-1]
    _inOut = [int(_idx/3), _idx % 3]
    badInOut.update({__p:_inOut})
    _idx_i = np.argsort(badChannels_in)[::-1][__p-1]
    _in = [int(_idx_i/3), _idx_i % 3]
    badIn.update({__p:_in})
    _idx_o = np.argsort(badChannels_out)[::-1][__p-1]
    _out = [int(_idx_o/3), _idx_o % 3]
    badOut.update({__p:_out})

MinPhaseMargin = simulator.simVars['PM_min']

labelMapIn = {0:r'$u_{p}$', 1:r'$u_{q}$', 2:r'$u_{r}$'}
labelMapOut = {0:r'$p$', 1:r'$q$', 2:r'$r$'}

if PMbased:
    print('[ INFO ] Identifying worst-case system (based on overall phase margins)')
    idxWorst_i = np.argmin(np.abs(np.array(controller.phase_margins_in))[:, badIn[1][0], badIn[1][1]])
    idxWorst_o = np.argmin(np.abs(np.array(controller.phase_margins_out))[:, badOut[1][0], badOut[1][1]])
else:
    print('[ INFO ] Identifying worst-case system (based on sensitivity DC gain)')
    idxWorst_i = np.argmax(np.array(simulator.controller.magSi_dc)[:, badIn[1][0], badIn[1][1]])
    idxWorst_o = np.argmax(np.array(simulator.controller.magSo_dc)[:, badOut[1][0], badOut[1][1]])

if idxWorst_i == 0:
    idxWorst_i = 1

if idxWorst_o == 0:
    idxWorst_o = 1


resLi = controlTools.diskmargin_siso(controller.Li[idxWorst_i][int(badIn[1][0]), int(badIn[1][1])], res = 10000, plot = True)
plotting.addHLINE(resLi['figures'][1].axes[1], MinPhaseMargin, 1e-6, 1e8, color = 'firebrick', linestyle = '--')
plotting.addXHSPAN(resLi['figures'][1].axes[1], -10, MinPhaseMargin, color = 'firebrick', alpha = 0.1, hatch = '/')

resLo = controlTools.diskmargin_siso(controller.Lo[idxWorst_o][int(badOut[1][0]), int(badOut[1][1])], res = 10000, plot = True)
plotting.addHLINE(resLo['figures'][1].axes[1], MinPhaseMargin, 1e-6, 1e8, color = 'firebrick', linestyle = '--')
plotting.addXHSPAN(resLo['figures'][1].axes[1], -10, MinPhaseMargin, color = 'firebrick', alpha = 0.1, hatch = '/')

plt.show()


# time = simulator.time
# Nshots = 5
# if Nshots % 2 != 0:
#     Nshots = Nshots + 1

# timesl = np.linspace(0, time[int(len(time)*0.6)], int(Nshots/2))
# timesu = np.linspace(time[int(len(time)*0.6)], time[int(len(time)*0.75)], int(Nshots/2))
# times = np.hstack((timesl[:-1], timesu))
# fig = simulator.animator.snapShot({simulator.model.modelID:simulator.simVars}, times)
# fig.axes[0].azim = 195.106
# fig.axes[0].elev = 16.404
# plotting.show()



time = simulator.time
Nshots = 8
if Nshots % 2 != 0:
    Nshots = Nshots + 1

timesl = np.linspace(0, time[int(len(time)*0.5)], int(Nshots/2))
timesu = np.linspace(time[int(len(time)*0.5)], time[int(len(time)*1)-1], int(Nshots/2))
times = np.hstack((timesl[:-1], timesu))
fig = simulator.animator.snapShot({simulator.model.modelID:simulator.simVars}, times)
fig.axes[0].azim = 321.53
fig.axes[0].elev = 18.4
plotting.show()


colors = ('tab:blue', 'tab:orange', 'tab:green')
labelMapIn = {0:r'$\mathbf{u_{p}}$', 1:r'$\mathbf{u_{q}}$', 2:r'$\mathbf{u_{r}}$'}
labelMapOut = {0:r'$\mathbf{p}$', 1:r'$\mathbf{q}$', 2:r'$\mathbf{r}$'}
fig = plt.figure(figsize = (12, 7))
axp = fig.add_subplot(311)
axq = fig.add_subplot(312, sharex = axp)
axr = fig.add_subplot(313, sharex = axp)

axMap = {0:axp, 1:axq, 2:axr}
GainHistory = np.array(controller.gains_history)[:, :3, :3]
for _o in range(GainHistory.shape[1]):
    for _i in range(GainHistory.shape[2]):
        linewidth = 1
        alpha_background = 0.7
        if _o == _i:
            linewidth = 3
            alpha_background = 1
        axMap[_o].plot(time[2:], GainHistory[:, _o, _i], color = colors[_i], label = labelMapIn[_i] + '-' + labelMapOut[_o], alpha = alpha_background, linewidth = linewidth)

axr.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 14)
axq.set_ylabel(r'$\mathbf{Gain}$, -', fontsize = 14)
axp.legend()
axp.grid('on')
axq.legend()
axq.grid('on')
axr.legend()
axr.grid('on')
plotting.prettifyAxis(axp)
plotting.prettifyAxis(axq)
plotting.prettifyAxis(axr)
plt.tight_layout()















# '''
# STEP RESPONSES
# '''

# directories = [
#     '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_UnityGains/NoNoise/STEP_RESPONSE',
#     '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_TunedGains/NoNoise/STEP_RESPONSE',
#     '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_OptimizedGains/NoNoise/STEP_RESPONSE',
#     '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/ContinuousAdaption/HDBeetle_Hover_UnityGains/NoNoise/STEP_RESPONSE',
#     '/data/AE-PhD/Results/Margin_stability_analysis/DiskMarginBased_Optimization/DiskMargin_PMGM/SimulOpt-InAndOut_Opt-SimulRobust-Perf_SensitivityBased/NoAdaption/HDBeetle_Hover_AggressiveOptimizedGains/NoNoise/STEP_RESPONSE'
# ]

# labels = [
#     r'$\mathbf{K_{base}}$',
#     r'$\mathbf{K_{tuned}}$',
#     r'$\mathbf{K_{opt}}$',
#     r'$\mathbf{K_{adapt}}$',
#     r'$\mathbf{K_{opt,agg}}$'
# ]
# colors = [
#     'tab:blue',
#     'tab:orange',
#     'tab:green',
#     'tab:red',
#     'tab:green'
# ]

# linestyles = [
#     'solid',
#     'solid',
#     'solid',
#     'solid',
#     'dotted'
# ]

# fig = plt.figure(figsize=(12, 7))
# axp = fig.add_subplot(311)
# axq = fig.add_subplot(312, sharex = axp)
# axr = fig.add_subplot(313, sharex = axp)
# axp.set_ylabel(r'$\mathbf{p}$, rad/s', fontsize = 14)
# axq.set_ylabel(r'$\mathbf{q}$, rad/s', fontsize = 14)
# axr.set_ylabel(r'$\mathbf{r}$, rad/s', fontsize = 14)
# axp.set_ylim([-0.2, 1.5])
# axq.set_ylim([-0.2, 1.5])
# axr.set_ylim([-0.2, 1.5])

# for d, clr, ls in zip(directories, colors, linestyles):
#     # ROLL
#     path = d + '/ROLL'
#     with open(path + '/simulatorInstance.pkl', 'rb') as f:
#         simulator = pkl.load(f)
#     axp.plot(simulator.time, simulator.state[:, 0, 6], color = clr, linewidth = 2, alpha = 0.8, zorder = 5, linestyle = ls)
#     # PITCH
#     path = d + '/PITCH'
#     with open(path + '/simulatorInstance.pkl', 'rb') as f:
#         simulator = pkl.load(f)
#     axq.plot(simulator.time, simulator.state[:, 0, 7], color = clr, linewidth = 2, alpha = 0.8, zorder = 5, linestyle = ls)
#     # YAW
#     path = d + '/YAW'
#     with open(path + '/simulatorInstance.pkl', 'rb') as f:
#         simulator = pkl.load(f)
#     axr.plot(simulator.time, simulator.state[:, 0, 8], color = clr, linewidth = 2, alpha = 0.8, zorder = 5, linestyle = ls)


# tArr = [simulator.time[0], 0.1, 0.1004, simulator.time[-1]]
# stepRef = [0, 0, 1, 1]
# axp.plot(tArr, stepRef, color = 'k', linestyle = '--', alpha = 0.8)
# axq.plot(tArr, stepRef, color = 'k', linestyle = '--', alpha = 0.8)
# axr.plot(tArr, stepRef, color = 'k', linestyle = '--', alpha = 0.8)
# axr.set_xlabel(r'$\mathbf{Time}$, s', fontsize = 14)
# handles = []
# plotting.addLegendLine(handles, color = 'k', linestyle = '--', label = 'Reference')
# for lbl, clr, ls in zip(labels, colors, linestyles):
#     plotting.addLegendLine(handles, color = clr, label = lbl, linestyle = ls)


# axp.legend(handles = handles)
# axp.grid('on')
# axq.grid('on')
# axr.grid('on')
# plotting.prettifyAxis(axp)
# plotting.prettifyAxis(axq)
# plotting.prettifyAxis(axr)
# fig.tight_layout()
# plt.show()









'''
G Responses
'''
G_As = np.array(controller.PIPE['G_A'])
G_Bs = np.array(controller.PIPE['G_B'])
G_Cs = np.array(controller.PIPE['G_C'])
G_Ds = np.array(controller.PIPE['G_D'])

magGs = []
for (A, B, C, D) in zip(G_As, G_Bs, G_Cs, G_Ds):
    Gs = c.ss2tf(c.ss(A, B, C, D))
    resp = c.freqresp(Gs, 0.1)
    magGs.append(resp.magnitude*np.sign(resp.fresp))

magGs = np.array(magGs).reshape(-1, 3, 3)

linestyles = ('solid', 'dashed', 'dotted')
colors = ('tab:blue', 'tab:orange', 'tab:green')
labelMapOut = {0:r'$\mathbf{p}$', 1:r'$\mathbf{q}$', 2:r'$\mathbf{r}$'}


fig = plt.figure()
axp = fig.add_subplot(311)
axq = fig.add_subplot(312, sharex = axp)
axr = fig.add_subplot(313, sharex = axp)
axp.set_ylim([-0.0006, 0.0006])
axq.set_ylim([-0.001, 0.001])
axr.set_ylim([-0.0005, 0.0005])
axMap = {0:axp, 1:axq, 2:axr}
for _o in range(3):
    for _i in range(3):
        linewidth = 1
        alpha_background = 0.7
        if _o == _i:
            linewidth = 3
            alpha_background = 1
        axMap[_o].plot(magGs[:, _o, _i], linestyle = linestyles[_i], color = colors[_i], label = labelMapOut[_i] + '-' + labelMapOut[_o], alpha = alpha_background, linewidth = linewidth)

for ax in axMap.values():
    idxs = np.argwhere(np.abs(np.array(controller.phase_margins_in)).reshape(-1, 9) < 15)[:, 0]
    for idx in idxs:
        plotting.addVLINE(ax, idx, -10, 10, linestyle = '--', color = 'firebrick')
    idxs = np.argwhere(np.abs(np.array(controller.phase_margins_out)).reshape(-1, 9) < 15)[:, 0]
    for idx in idxs:
        plotting.addVLINE(ax, idx, -10, 10, linestyle = 'dotted', color = 'firebrick')

plotting.addHLINE(axp, 0, -len(magGs), 2*len(magGs), color = 'k', alpha = 0.2)
plotting.addHLINE(axq, 0, -len(magGs), 2*len(magGs), color = 'k', alpha = 0.2)
plotting.addHLINE(axr, 0, -len(magGs), 2*len(magGs), color = 'k', alpha = 0.2)
axr.set_xlabel(r'$\mathbf{Sample}$, -', fontsize = 14)
axq.set_ylabel(r'$\mathbf{|G|}$, -', fontsize = 14)
axp.legend()
axq.legend()
axr.legend()
plotting.prettifyAxis(axp)
plotting.prettifyAxis(axq)
plotting.prettifyAxis(axr)
plt.show()