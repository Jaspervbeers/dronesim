import numpy as np
import matplotlib.pyplot as plt
import control as c
import os
import sys

# Add parent level directory to path
sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))

import controlTools
import plotting


'''
Example MIMO sensitivity (2x2)

Verified with MATLAB results
      - controlTools.getAllOpenLoopTransfer_ss
      - controlTools.sensitivity
TODO:
      - controlTools.getLoopTransferFunction_at_ij (Only works for diagonal)
'''
# Spinning satellite from matlab: https://nl.mathworks.com/help/robust/ref/dynamicsystem.diskmargin.html (Accessed 21-08-2023)
# In MATLAB define P and K as below:
# Then do: 
#   minreal(1/tf(eye(2) - P*K)) for So
# and: 
#   minreal(1/tf(eye(2) - K*P)) for Si
# Can also examine bode plots
# Example below computes the response both in time-domain (using state space manipulation)
# and in frequency domain directly. 
# As we are taking inverse of MIMO transfer functions, we see numerical issues for frequency
# based approach (compare resultant transfer functions) 
#   -> Nonetheless, the resultant frequency responses can remain the same if minreal is used appropriately. 
G_A = np.array([[0, 10], [-10, 0]])
G_B = np.eye(2)
G_C = np.array([[1, 10], [-10, 1]])
G_D = np.zeros((2, 2))
K_A = np.zeros(G_A.shape)
K_B = np.zeros(G_B.shape)
K_C = np.zeros(G_D.shape)
K_D = np.array([[1, -2], [0, 1]])

P_ss = c.ss(G_A, G_B, G_C, G_D)
K_ss = c.ss(K_A, K_B, K_C, -1*K_D) #-ve feedback 
P_tf = c.ss2tf(P_ss)
K_tf = c.ss2tf(K_ss)

Lo_tf = controlTools.multiply(controlTools.removeNearZero(P_tf), K_tf)
Li_tf = controlTools.multiply(K_tf, controlTools.removeNearZero(P_tf))
Lo_ss = controlTools.getAllOpenLoopTransfer_ss(K_ss, P_ss, at='output')
Li_ss = controlTools.getAllOpenLoopTransfer_ss(K_ss, P_ss, at='input')

So_tf = controlTools.sensitivity(Lo_tf.minreal(), method = 'laplace_domain')
Si_tf = controlTools.sensitivity(Li_tf.minreal(), method = 'laplace_domain')
So_ss = controlTools.sensitivity(Lo_ss, method = 'time_domain')
Si_ss = controlTools.sensitivity(Li_ss, method = 'time_domain')

# Show results
print('RESULTS SENSITIVITY FUNCTION & MIMO INVERSION')
print('Output sensitivity, S_o (MATLAB):')
print('\tFrom input 1 to output...'+
      '\n\t' + '\t'     + 's^2 - 21 s + 180'+
      '\n\t' + '[1]\t'  + '----------------'+
      '\n\t' + '\t'     + 's^2 - 22 s + 381'+
      '\n' + 
      '\n\t' + '\t'     + '   -10 s - 10   '+
      '\n\t' + '[2]\t'  + '----------------'+
      '\n\t' + '\t'     + 's^2 - 22 s + 381'
      )
print('\tFrom input 2 to output...'+
      '\n\t' + '\t'     + '    8 s + 210   '+
      '\n\t' + '[1]\t'  + '----------------'+
      '\n\t' + '\t'     + 's^2 - 22 s + 381'+
      '\n' + 
      '\n\t' + '\t'     + '  s^2 - s + 200 '+
      '\n\t' + '[2]\t'  + '----------------'+
      '\n\t' + '\t'     + 's^2 - 22 s + 381'
      )
print('Output sensitivity, S_o (StateSpace based):')
print(c.ss2tf(So_ss))
print('Output sensitivity, S_o (TransferFunction based):')
print(So_tf)
print('\n\n')

# L00 = controlTools.getLoopTransferFunction_at_ij(-1*K_ss, P_ss, 0, 0, at = 'output')
# S00 = controlTools.sensitivity(L00)
# S00_tf = c.tf(S00)
# plt.figure()
# c.bode(S00_tf, dB = True)
# c.bode(So_ss[0, 0], dB = True)

# L01 = controlTools.getLoopTransferFunction_at_ij(-1*K_ss, P_ss, 0, 1, at = 'output')
# S01 = controlTools.sensitivity(L01)
# S01_tf = c.tf(S01)
# plt.figure()
# c.bode(S01_tf, dB = True)
# c.bode(So_ss[0, 1], dB = True)

# L10 = controlTools.getLoopTransferFunction_at_ij(-1*K_ss, P_ss, 1, 0, at = 'output')
# S10 = controlTools.sensitivity(L10)
# S10_tf = c.tf(S10)
# plt.figure()
# c.bode(S10_tf, dB = True)
# c.bode(So_ss[1, 0], dB = True)

# L11 = controlTools.getLoopTransferFunction_at_ij(-1*K_ss, P_ss, 1, 1, at = 'output')
# S11 = controlTools.sensitivity(L11)
# S11_tf = c.tf(S11)
# plt.figure()
# c.bode(S11_tf, dB = True)
# c.bode(So_ss[1, 1], dB = True)

plt.show()



'''
MIMO sensitivity example 3x3:

Verified with MATLAB 
'''
G_A = np.array([[0, 10, 0], [-10, 0, 0], [1, 1, -3]])
G_B = np.eye(3)
G_C = np.array([[1, 10, 0], [-10, 1, 0], [1, 1, 1]])
G_D = np.zeros((3, 3))
K_A = np.zeros(G_A.shape)
K_B = np.zeros(G_B.shape)
K_C = np.zeros(G_D.shape)
K_D = np.array([[1, -2, 0], [0, 1, 0], [0, 0, 1]])

P_ss = c.ss(G_A, G_B, G_C, G_D)
K_ss = c.ss(K_A, K_B, K_C, -1*K_D) #-ve feedback 
P_tf = c.ss2tf(P_ss)
K_tf = c.ss2tf(K_ss)

Lo_tf = controlTools.multiply(controlTools.removeNearZero(P_tf), K_tf)
Li_tf = controlTools.multiply(K_tf, controlTools.removeNearZero(P_tf))
Lo_ss = controlTools.getAllOpenLoopTransfer_ss(K_ss, P_ss, at='output')
Li_ss = controlTools.getAllOpenLoopTransfer_ss(K_ss, P_ss, at='input')

So_tf = controlTools.sensitivity(Lo_tf.minreal(), method = 'laplace_domain')
Si_tf = controlTools.sensitivity(Li_tf.minreal(), method = 'laplace_domain')
So_ss = controlTools.sensitivity(Lo_ss, method = 'time_domain')
Si_ss = controlTools.sensitivity(Li_ss, method = 'time_domain')




# General bode plots
magSi, phaseSi, omega, figSi = controlTools.bode(Si_ss, returnFig=True, color = 'tab:blue')
figSi.suptitle(r'$\mathbf{Sensitivity}$ at $\mathbf{input}$, $\mathbf{S_{i}}$')
magSi, phaseSi, omega, figSi = controlTools.bode(Si_tf.minreal(), returnFig=True, color = 'tab:orange', parentFig = figSi)
handles = []
plotting.addLegendLine(handles, color = 'tab:blue', label = 'StateSpace')
plotting.addLegendLine(handles, color = 'tab:orange', label = 'TransferFunction')
figSi.axes[0].legend(handles = handles)
magSo, phaseSo, omega, figSo = controlTools.bode(So_ss, returnFig=True, color = 'tab:blue')
figSo.suptitle(r'$\mathbf{Sensitivity}$ at $\mathbf{output}$, $\mathbf{S_{o}}$')
magSo, phaseSo, omega, figSo = controlTools.bode(So_tf.minreal(), returnFig=True, color = 'tab:orange', parentFig = figSo)
handles = []
plotting.addLegendLine(handles, color = 'tab:blue', label = 'StateSpace')
plotting.addLegendLine(handles, color = 'tab:orange', label = 'TransferFunction')
figSo.axes[0].legend(handles = handles)

plt.show()



# EXAMPLE 1
# Dummy system; from https://arxiv.org/pdf/2003.04771.pdf (Accessed 02/08/2023; August; DOI: 10.1109/MCS.2020.3005277)
s = c.tf('s')
L1 = 25/(s**3 + 10*s**2 + 10*s + 10)

# Expect:
# For skew = 0
#   alpha_max = 0.46
#   gamma_min = 0.64
#   gamma_max = 1.59
#   PM = 25.8
DMs_L1 = controlTools.diskmargin_siso(L1, skew = 0, plot = True)
print('EXAMPLE 1')
print('Broken-loop system: ')
print(L1)
print(f'\talpha_max = {DMs_L1["DM"]}')
print(f'\t[gamma_min, gamma_max] = {DMs_L1["GM"]}')
print(f'\tPM = {DMs_L1["PM"]}')
print('\n\n')
plt.show()

# For skew < 0, disk shifts left, for skew > 0 disk shift right
# Also, since alpha_max is not being adjusted, we will have different sizes of disks
DMs_L1_minus1 = controlTools.diskmargin_siso(L1, skew = -1)
DMs_L1_plus1 = controlTools.diskmargin_siso(L1, skew = 1)

fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot(111)
ax.set_ylim([0, 40])
controlTools._plotDiskMarginVariation(DMs_L1['gammas'], DMs_L1['thetas'], parentAx=ax, color = 'tab:blue')
controlTools._plotDiskMarginVariation(DMs_L1_minus1['gammas'], DMs_L1_minus1['thetas'], parentAx=ax, color = 'tab:orange')
controlTools._plotDiskMarginVariation(DMs_L1_plus1['gammas'], DMs_L1_plus1['thetas'], parentAx=ax, color = 'tab:green')
handles = []
plotting.addLegendPatch(handles, color = 'tab:blue', label = r'$\sigma = 0$')
plotting.addLegendPatch(handles, color = 'tab:orange', label = r'$\sigma = -1$')
plotting.addLegendPatch(handles, color = 'tab:green', label = r'$\sigma = 1$')
ax.legend(handles = handles)
plt.show()



# EXAMPLE 2
# Verify gain-phase variation plots for alpha_max = 0.75, skew = [-1, 0, 1]
[gamma_min_0, gamma_max_0], PM_max_0, _, _ = controlTools._getDiskParams(0.75, skew = 0)
gammas_0 = np.logspace(np.log10(gamma_min_0), np.log10(gamma_max_0), 1000)
thetas_0 = controlTools._thetaFromGamma(gammas_0, gamma_min_0, gamma_max_0)

[gamma_min_m1, gamma_max_m1], PM_max_m1, _, _ = controlTools._getDiskParams(0.75, skew = -1)
gammas_m1 = np.logspace(np.log10(gamma_min_m1), np.log10(gamma_max_m1), 1000)
thetas_m1 = controlTools._thetaFromGamma(gammas_m1, gamma_min_m1, gamma_max_m1)

[gamma_min_p1, gamma_max_p1], PM_max_p1, _, _ = controlTools._getDiskParams(0.75, skew = 1)
gammas_p1 = np.logspace(np.log10(gamma_min_p1), np.log10(gamma_max_p1), 1000)
thetas_p1 = controlTools._thetaFromGamma(gammas_p1, gamma_min_p1, gamma_max_p1)

fig = plt.figure(figsize = (9, 5))
ax = fig.add_subplot(111)
ax.set_ylim([0, 50])
controlTools._plotDiskMarginVariation(gammas_0, thetas_0, parentAx=ax, color = 'tab:blue')
controlTools._plotDiskMarginVariation(gammas_m1, thetas_m1, parentAx=ax, color = 'tab:orange')
controlTools._plotDiskMarginVariation(gammas_p1, thetas_p1, parentAx=ax, color = 'tab:green')

handles = []
plotting.addLegendPatch(handles, color = 'tab:blue', label = r'$\sigma = 0$')
plotting.addLegendPatch(handles, color = 'tab:orange', label = r'$\sigma = -1$')
plotting.addLegendPatch(handles, color = 'tab:green', label = r'$\sigma = 1$')
ax.legend(handles = handles)
plt.show()



# EXAMPLE 3
L2 = (6.25*(s+3)*(s+5))/(s*(s+1)**2 * (s**2 + 0.18*s + 100))

DMs_L2 = controlTools.diskmargin_siso(L2, plot = True)
plt.show()




# # # Example: 3x3 system
# # G11 = c.TransferFunction([1], [1, 1])
# # G12 = c.TransferFunction([1], [1, 2])
# # G13 = c.TransferFunction([1], [1, 3])
# # G21 = c.TransferFunction([1], [1, 4])
# # G22 = c.TransferFunction([1], [1, 5])
# # G23 = c.TransferFunction([1], [1, 6])
# # G31 = c.TransferFunction([1], [1, 7])
# # G32 = c.TransferFunction([1], [1, 8])
# # G33 = c.TransferFunction([1], [1, 9])

# # Example: 3x3 system
# G11 = c.TransferFunction([1], [1, 1])
# G12 = c.TransferFunction([1], [1, 3, 1])
# G13 = c.TransferFunction([3, 2], [1, 3, 4, 5])
# G21 = c.TransferFunction([0.5, 3, 1.2], [1, 1, 0.7, 3])
# G22 = c.TransferFunction([1], [1, 5])
# G23 = c.TransferFunction([10], [10, 1])
# G31 = c.TransferFunction([1, 2], [3, 2, 1])
# G32 = c.TransferFunction([0.1, 0.001], [1, 3, 8])
# G33 = c.TransferFunction([1], [1, 9])

# # G = [[G11, G12, G13], [G21, G22, G23], [G31, G32, G33]]
# G = [[G11, G12, c.tf('s')*0], [c.tf('s')*0, G22, G23], [c.tf('s')*0, c.tf('s')*0, G33]]

# G_inv = controlTools.invert_MIMO(G, method="laplace_domain")

# # G_inv = controlTools._invert_MIMO(G)
# # for row in G_inv:
# #     for elem in row:
# #         print(elem)


# # Verified with MATLAB! 

