'''Equations of Motion for the quadrotor. The runSim class requires F, M, x as inputs and outputs x_dot. 

As inputs
:param F: Forces (x, y, z) at time t
:param M: Moments (x, y, z) at time t
:param x: state vector (roll, pitch, yaw, u, v, w, p, q, r) at time t
:param droneParams: dictionary containing drone configuration and additional information. Required fields are drone mass, gravitational acceleration, and drone moment of inertia

As outputs
:param x_dot: State derivative 
'''

from numpy import array as np_array
from numpy import matmul as np_matmul
from numpy import cross as np_cross
from numpy.linalg import inv as np_linalg_inv

from funcs.angleFuncs import Eul2Quat, QuatRot, PQR_to_PhiThetaDot_Euler, PQR_to_PhiThetaDot_Quat


def EOM(simVars):
    step = simVars['currentTimeStep_index']
    # NOTE: step + 1 for forces and moments since this one-step-ahead info is available from model
    F = simVars['forces'][step+1]
    M = simVars['moments'][step+1]
    droneParams = simVars['model'].droneParams
    x = simVars['state'][step]
    quat = simVars['quat'][step]
    
    x_dot = _EOM(x, quat, F, M, droneParams = droneParams)

    return x_dot


def _EOM(x, quat,  F, M, droneParams = None):
    linVel = x[:, 3:6]
    rotVel = x[:, 6:9]

    g = np_array([0, 0, droneParams['g']])

    linAcc = F/droneParams['m'] + QuatRot(quat, g.reshape(1, -1), rot='E2B') - np_cross(rotVel, linVel)
    rotAcc = np_matmul(np_linalg_inv(droneParams['Iv']), M.T) - np_matmul(np_linalg_inv(droneParams['Iv']), np_cross(rotVel, np_matmul(droneParams['Iv'], rotVel.T).T).T)

    x_dot = x.copy()
    # x_dot[:, 0:3] = PQR_to_PhiThetaDot_Quat(att, rotVel)
    # x_dot[:, 0:3] = PQR_to_PhiThetaDot_Euler(att, rotVel)
    x_dot[:, 3:6] = linAcc
    x_dot[:, 6:9] = rotAcc.T
    x_dot[:, 9:12] = QuatRot(quat, linVel, rot = 'B2E') 
    return x_dot