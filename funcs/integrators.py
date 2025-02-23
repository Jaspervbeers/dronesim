from funcs import angleFuncs
from funcs import droneEOM
import numpy as np

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

def droneIntegrator_Euler(simVars):
    # Extract necessary variables 
    step = simVars['currentTimeStep_index']
    x = simVars['state'][step]
    # NOTE: we take step + 1 for state derivative since it is updated from EOM (i.e. latest available)
    x_dot = simVars['stateDerivative'][step+1]
    dt = simVars['dt']

    # Define new state vector
    newStates = x.copy()*0

    # Update body states: u, v, w, p, q, r, x, y, z
    newStates[:, 3:12] = x[:, 3:12] + x_dot[:, 3:12]*dt

    # Update earth states: roll, pitch, yaw
    # -> Note, at small angles DM method can have issues with yaw. 
    # newAngles = _integrateQuat_DM(simVars)
    newAngles = _integrateQuat_PCDM(simVars)
    # Need to enforce angle continuity since quaternions map to [-pi, pi)
    if step > 5:
        newAngles = np.unwrap(np.vstack((simVars['state'][step-5:step, 0, :3], newAngles)).T).T[-1]
    newStates[:, :3] = newAngles

    # Update state dot for bookkeeping
    x_dot[:, :3] = (newStates[:, :3] - x[:, :3])/dt

    # import code
    # code.interact(local=locals())

    return newStates



def droneIntegrator_rk4(simVars):
    # Extract necessary variables 
    step = simVars['currentTimeStep_index']
    x = simVars['state'][step].copy()
    # NOTE: we take step + 1 for state derivative since it is updated from EOM (i.e. latest available)
    x_dot = simVars['stateDerivative'][step+1].copy()
    dt = simVars['dt']
    F = simVars['forces'][step + 1].copy()
    M = simVars['moments'][step + 1].copy()
    droneParams = simVars['model'].droneParams
    omega = simVars['inputs'][step + 1]

    newStates = x.copy()*0

    # xNew = rk4(droneEOM._EOM, x, F, M, dt, droneParams = droneParams)
    xNew = rk4(_EoMModelFunc, x, omega, None, dt, simVars = simVars)

    newStates[:, 3:12] = xNew[:, 3:12]

    # Update earth states: roll, pitch, yaw
    # newAngles = _integrateQuat_DM(simVars)
    newAngles = _integrateQuat_PCDM(simVars)    
    if step > 5:
        newAngles = np.unwrap(np.vstack((simVars['state'][step-5:step, 0, :3], newAngles)).T).T[-1]
    newStates[:, :3] = newAngles

    # Update state dot for bookkeeping
    x_dot[:, :3] = (newStates[:, :3] - x[:, :3])/dt

    return newStates


def _EoMModelFunc(x, u, *args, simVars = None, **kwargs):
    model = simVars['model']
    step = simVars['currentTimeStep_index']
    quat = simVars['quat'][step]
    droneParams = model.droneParams
    
    # Prepare inputs
    modelInputs = model.FxModel.droneGetModelInput(x, u)

    # Make predictions
    Fx = model.FxModel.predict(modelInputs).__array__()[0][0] - model.Fx0
    Fy = model.FyModel.predict(modelInputs).__array__()[0][0] - model.Fy0
    Fz = model.FzModel.predict(modelInputs).__array__()[0][0] - model.Fz0
    Mx = model.MxModel.predict(modelInputs).__array__()[0][0] - model.Mx0
    My = model.MyModel.predict(modelInputs).__array__()[0][0] - model.My0
    Mz = model.MzModel.predict(modelInputs).__array__()[0][0] - model.Mz0

    if model.isNormalized:
        # Extract normalizing factors
        F_den = np.array(modelInputs['F_den']).reshape(-1)
        M_den = np.array(modelInputs['M_den']).reshape(-1)
    else:
        F_den, M_den = 1, 1

    # Build force and moment vectors
    F = (np.array([Fx, Fy, Fz])*F_den).reshape(1, -1)
    M = (np.array([Mx, My, Mz])*M_den).reshape(1, -1)

    x_dot = droneEOM._EOM(x, quat, F, M, droneParams = droneParams)

    return x_dot


def rk4(func, x, u, t, dt, **kwargs):
    ''' Fourth order Runge-Kutta numerical integration scheme

    :param func: Derivative of unknown function to use for integration
    :param x: State vector at step n
    :param u: Input vector at step n
    :param t: Time at step n
    :param dt: Time step size
    :param kwargs: Additional keyword arguments required by <func>
    :return: State vector at step n + 1 and associated time (i.e. t + dt)
    '''
    k1 = dt * func(x, u, t, **kwargs)
    k2 = dt * func(x + k1/2, u, t, **kwargs)
    k3 = dt * func(x + k2/2, u, t, **kwargs)
    k4 = dt * func(x + k3, u, t, **kwargs)

    x_new = x + k1/6 + k2/3 + k3/3 + k4/6

    return x_new




def _integrateQuat_DM(simVars):
    # Direct multiplication method from:
    #   Betsch, P., Siebert, R.: Rigid body dynamics in terms of Quaternions: Hamiltonian formulation and conserving numericalintegration. Int. J. Numer. Methods Eng. 79(4), 444–473 (2009)
    # Also found in:
    #   https://link.springer.com/content/pdf/10.1007/s00707-013-0914-2.pdf
    #   Zhao, F., and B. G. M. Van Wachem. "A novel Quaternion integration approach for describing the behaviour of non-spherical particles." Acta Mechanica 224.12 (2013): 3091-3109.
    step = simVars['currentTimeStep_index']
    dt = simVars['dt']
    _rates = simVars['state'][step][:, 6:9]
    ratesDot = simVars['stateDerivative'][step + 1][:, 6:9]
    rates = _rates + ratesDot*dt 
    quat = simVars['quat'][step]

    # Get unit Quaternion of rotation, based on angular velocity
    n = np.linalg.norm(rates)
    # if np.isclose(n, 0):
    if n == 0:
        xyz = rates
    else:
        xyz = rates/n
    
    qw = np.array([np.cos(n*dt/2)])
    qx = np.sin(n/2*dt)*xyz[:, 0]
    qy = np.sin(n/2*dt)*xyz[:, 1]
    qz = np.sin(n/2*dt)*xyz[:, 2]

    q_tilde = np.array([qw, qx, qy, qz]).reshape(-1, 4)

    qNew = angleFuncs.QuatMul(q_tilde, quat)
    
    simVars['quat'][step + 1] = qNew

    angle = enforceAngleCont(simVars['state'][step][:, :3], angleFuncs.Quat2Eul(qNew))

    # import code
    # code.interact(local=locals())

    return angle


def _integrateQuat_PCDM(simVars):
    # Predictor-corrector direct multiplication method from:
    # https://link.springer.com/content/pdf/10.1007/s00707-013-0914-2.pdf
    # Zhao, F., and B. G. M. Van Wachem. "A novel Quaternion integration approach for describing the behaviour of non-spherical particles." Acta Mechanica 224.12 (2013): 3091-3109.
    step = simVars['currentTimeStep_index']
    dt = simVars['dt']
    rates = simVars['state'][step][:, 6:9]
    ratesDot = simVars['stateDerivative'][step + 1][:, 6:9]
    # quat = angleFuncs.Eul2Quat(simVars['state'][step][:, :3])
    quat = simVars['quat'][step]
    # quatDot = simVars['quatDot'][step + 1]

    # Get quarter and half predictions of (body) angular rate
    w_b_14 = rates + 0.25*ratesDot*dt
    w_b_12 = rates + 0.5*ratesDot*dt

    w_E_14 = angleFuncs.QuatRot(quat, w_b_14)

    n_prime_12 = np.linalg.norm(w_E_14)
    if np.isclose(n_prime_12, 0):
        xyz_prime_12 = w_E_14
    else:
        xyz_prime_12 = w_E_14/n_prime_12
    qw_prime_12 = np.cos([n_prime_12*dt*0.25])
    qx_prime_12 = np.sin(n_prime_12*dt*0.25)*xyz_prime_12[:, 0]
    qy_prime_12 = np.sin(n_prime_12*dt*0.25)*xyz_prime_12[:, 1]
    qz_prime_12 = np.sin(n_prime_12*dt*0.25)*xyz_prime_12[:, 2]

    _quat_prime_12 = np.array([qw_prime_12, qx_prime_12, qy_prime_12, qz_prime_12]).reshape(-1, 4)

    quat_prime_12 = angleFuncs.QuatMul(_quat_prime_12, quat)

    w_E_12 = angleFuncs.QuatRot(quat_prime_12, w_b_12)

    n = np.linalg.norm(w_E_12)
    if np.isclose(n, 0):
        xyz = w_E_12
    else:
        xyz = w_E_12/n
    qw = np.cos([n*dt*0.5])
    qx = np.sin(n*dt*0.5)*xyz[:, 0]
    qy = np.sin(n*dt*0.5)*xyz[:, 1]
    qz = np.sin(n*dt*0.5)*xyz[:, 2]

    _quatNew = np.array([qw, qx, qy, qz]).reshape(-1, 4)

    quatNew = angleFuncs.QuatMul(_quatNew, quat)

    simVars['quat'][step + 1] = quatNew

    # w_b_12_dot = np.matmul(np.linalg.inv(simVars['model'].droneParams['Iv']), simVars['moments'].T) - np.matmul(np.linalg.inv(simVars['model'].droneParams['Iv']), np.cross(w_b_12, np.matmul(simVars['model'].droneParams['Iv'], w_b_12.T).T).T)

    angle = enforceAngleCont(simVars['state'][step][:, :3], angleFuncs.Quat2Eul(quatNew))

    # import code
    # code.interact(local=locals())

    return angle


def enforceAngleCont(angleOld, angleNew, maxChange = np.pi/2):
    # perms = np.array([
    #     [np.pi, np.pi, np.pi],
    #     [-np.pi, np.pi, np.pi],
    #     [-np.pi, -np.pi, np.pi],
    #     [np.pi, np.pi, -np.pi],
    #     [np.pi, -np.pi, -np.pi],
    #     [np.pi, -np.pi, np.pi],
    #     [-np.pi, np.pi, -np.pi],
    #     [-np.pi, -np.pi, -np.pi],
    #     [0, 0, 0]
    # ])
    # test = np.zeros((len(perms), 3))
    # test[:, 0] = angleNew[:, 0] + perms[:, 0]
    # test[:, 1] = angleNew[:, 1] + perms[:, 1]
    # test[:, 2] = angleNew[:, 2] + perms[:, 2]

    test = _applyPerms(angleNew)
    
    quatTest = angleFuncs.Eul2Quat(test)
    quatTrue = angleFuncs.Eul2Quat(angleNew)

    viableTests = test[np.where(np.sum(np.isclose(quatTrue, quatTest), axis = 1) == 4), :].reshape(-1, 3)

    error = np.abs(wrap2Pi(angleOld) - viableTests)
    angleNewCorrected = viableTests[np.argmin(np.sum(error, axis = 1)), :]

    return angleNewCorrected


def _applyPerms(angle):
    # Remove 2pi multiples
    remainder = wrap2Pi(angle)
    equivalent = np.zeros(remainder.shape)
    equivalent[:, 0] = remainder[:, 0] - np.pi
    equivalent[:, 1] = -1*(remainder[:, 1] - np.pi)
    equivalent[:, 2] = remainder[:, 2] - np.pi
    return np.vstack((angle, equivalent))


def wrap2Pi(angle):
    return angle - (np.ceil((angle + np.pi)/(2*np.pi)) - 1)*(2*np.pi)