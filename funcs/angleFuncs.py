import numpy as np


def Eul2Quat(theta):
    '''Function to convert euler angles to their quaternion equivalents
    
    :param theta: Array of the euler angles with shape (N, 3) or (3, N) where N is the number of samples

    :return: Quaternion representation of euler angles
    '''

    # Reshape theta
    theta = theta.reshape(-1, 3)

    quat = np.zeros((len(theta), 4))

    cr = np.cos(theta[:, 0]*0.5)
    sr = np.sin(theta[:, 0]*0.5)
    cp = np.cos(theta[:, 1]*0.5)
    sp = np.sin(theta[:, 1]*0.5)
    cy = np.cos(theta[:, 2]*0.5)
    sy = np.sin(theta[:, 2]*0.5)

    quat[:, 0] = cr*cp*cy + sr*sp*sy
    quat[:, 1] = sr*cp*cy - cr*sp*sy
    quat[:, 2] = cr*sp*cy + sr*cp*sy
    quat[:, 3] = cr*cp*sy - sr*sp*cy

    return quat


def Quat2Eul(quat):
    '''Function to convert quaternion representation of orientation to euler angles
    
    :param quat: Quaternion representation, as array with shape (N, 4), where N is the number of samples
    
    :return: Corresponding euler angles
    '''
    quat = quat.reshape(-1, 4)
    eul = np.zeros((len(quat), 3))
    # Compute frequently used quantities
    sqx = quat[:, 0]*quat[:, 0]
    sqy = quat[:, 1]*quat[:, 1]
    sqz = quat[:, 2]*quat[:, 2]
    sqw = quat[:, 3]*quat[:, 3]

    # Check for singularities
    unit = sqw + sqx + sqy + sqz
    test = quat[:, 0]*quat[:, 2] - quat[:, 1]*quat[:, 3]
    # if test > 0.4995*unit:
    # if test > 0.49995*unit:
    if test > 0.499995*unit:
        # Singularity at pitch = pi/2
        eul[:, 0] = 2*np.arctan2(quat[:, 1], quat[:, 0])
        eul[:, 1] = np.pi/2
        eul[:, 2] = 0
    # elif test < -0.4995*unit:
    # elif test < -0.49995*unit:
    elif test < -0.499995*unit:
        # Singularity at pitch = -pi/2
        eul[:, 0] = -2*np.arctan2(quat[:, 1], quat[:, 0])
        eul[:, 1] = -np.pi/2
        eul[:, 2] = 0
    else:
        eul[:, 0] = np.arctan2(2*(quat[:, 0]*quat[:, 1] + quat[:, 2]*quat[:, 3]), 1 - 2*(sqy+sqz))
        # eul[:, 1] = np.arcsin(np.around(2*(quat[:, 0]*quat[:, 2] - quat[:, 3]*quat[:, 1]), 15))
        eul[:, 1] = np.arcsin(2*(quat[:, 0]*quat[:, 2] - quat[:, 3]*quat[:, 1]))
        eul[:, 2] = np.arctan2(2*(quat[:, 0]*quat[:, 3] + quat[:, 2]*quat[:, 1]), 1 - 2*(sqz+sqw))
    return eul



def QuatRot(q, x, rot='B2E'):
    '''Function to rotate a vector using its quaternion representation. 

    :param q: Quaternion signal, as array with shape [N, 4], where N is the number of samples
    :param x: Signal to rotate, as array with shape [N, 3]
    :param rot: String indicating the order of rotation; options are 'B2E' or 'E2B'. Default = 'B2E', indicating that the rotations are from the body frame to the earth frame. Conversely, 'E2B' denotes rotations from earth frame to body frame. 
    
    :return: Rotated x
    '''
    if rot == 'B2E':
        q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif rot == 'E2B':
        q0, q1, q2, q3 = q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]
    else:
        raise ValueError('specified rot is unknown. Use "B2E" or "E2B" for body to earth or earth to body rotations respectively')

    # Define rotation matrices for each axis 
    R_1 = np.array([(q0*q0 + q1*q1 - q2*q2 -q3*q3), (2*(q1*q2 - q0*q3)), (2*(q0*q2 + q1*q3))])
    R_2 = np.array([(2*(q1*q2 + q0*q3)), (q0*q0 - q1*q1 + q2*q2 - q3*q3), (2*(q2*q3 - q0*q1))])
    R_3 = np.array([(2*(q1*q3 - q0*q2)), (2*(q0*q1 + q2*q3)), (q0*q0 - q1*q1 - q2*q2 + q3*q3)])

    # Manipulate the indices of the rotation matrices above to get a vector of form
    # N x [3 x 3] such that each element corresponds to the rotation matrix for that
    # specific sample and can therefore be multiplied directly with the acceleration array
    R_1 = R_1.T
    R_2 = R_2.T
    R_3 = R_3.T
    R_stack = np.zeros((3*len(R_1), 3))
    R_stack[0:(3*len(R_1)):3] = R_1
    R_stack[1:(3*len(R_1)):3] = R_2
    R_stack[2:(3*len(R_1)):3] = R_3
    R = R_stack.reshape((len(R_1), 3, 3))
    
    x_rot = np.matmul(R, x.reshape((len(x), -1, 1)))

    return x_rot.reshape(x.shape)



def PQR_to_PhiThetaDot_Quat(att, rate):
    quatJPL = Eul2Quat_JPL(att)
    rateQ = quatJPL.copy()*0
    rateQ[:, :3] = rate

    q1 = quatJPL[:, 0]
    q2 = quatJPL[:, 1]
    q3 = quatJPL[:, 2]
    q4 = quatJPL[:, 3]

    A1 = np.array([q4, -q3, q2, q1])
    A2 = np.array([q3, q4, -q1, q2])
    A3 = np.array([-q2, q1, q4, q3])
    A4 = np.array([-q1, -q2, -q3, q4])

    A1 = A1.T
    A2 = A2.T
    A3 = A3.T
    A4 = A4.T
    A_stack = np.zeros((4*len(A1), 4))
    A_stack[0:(4*len(A1)):4] = A1
    A_stack[1:(4*len(A1)):4] = A2
    A_stack[2:(4*len(A1)):4] = A3
    A_stack[3:(4*len(A1)):4] = A4
    A = A_stack.reshape((len(A1), 4, 4))

    quatJPL_dot = 0.5*np.matmul(A, rateQ.reshape((len(rateQ), -1, 1)))
    q1_dot = quatJPL_dot[:, 0]
    q2_dot = quatJPL_dot[:, 1]
    q3_dot = quatJPL_dot[:, 2]
    q4_dot = quatJPL_dot[:, 3]
    
    phi_dot = d_arctan_quat(q4, q1, q2, q3, q4_dot, q1_dot, q2_dot, q3_dot)
    theta_dot = d_arcsin_quat(q4, q1, q2, q3, q4_dot, q1_dot, q2_dot, q3_dot)
    psi_dot = d_arctan_quat(q4, q3, q2, q1, q4_dot, q3_dot, q2_dot, q1_dot)

    # Attitude dot
    att_dot = np.array((phi_dot, theta_dot, psi_dot)).reshape(rate.shape)
    
    return att_dot


def d_arctan_quat(a, b, c, d, da, db, dc, dd):
    den1 = ((1-2*(b**2 + c**2)) * (1 + (4*(a*b + c*d)**2)/(1 - 2*(b**2 + c**2))**2))
    den2 = (1 + 4*b**4 + 4*c**4 + 4*b**2*(-1 + a**2 + 2*c**2) + 8*a*b*c*d + 4*c**2*(-1+d**2))
    dTda = 2*b*da / den1
    dTdb = 2*(a + 2*a*b**2 - 2*a*c**2 + 4*b*c*d)*db/ den2
    dTdc = 2*(4*a*b*c + d - 2*b**2*d + 2*c**2*d)*dc/ den2
    dTdd = 2*c*dd/ den1
    return dTda + dTdb + dTdc + dTdd


def d_arcsin_quat(a, b, c, d, da, db, dc, dd):
    den = np.sqrt((1-4*(a*c - b*d)**2))
    dSda = (2*c / den) * da
    dSdb = (-2*d / den) * db
    dSdc = (2*a / den) * dc
    dSdd = (-2*b / den) * dd
    return dSda + dSdb + dSdc + dSdd


# Below is incorrect since I need to convert phidot, thetadot, psidot to quat_dot. 
def PhiThetaDot_2_PQR_Quat(state, cmd):
    quatJPL = Eul2Quat_JPL(state[:, :3])
    rate = cmd
    rateQ = quatJPL.copy()*0
    rateQ[:, :3] = rate

    q1 = quatJPL[:, 0]
    q2 = quatJPL[:, 1]
    q3 = quatJPL[:, 2]
    q4 = quatJPL[:, 3]

    A1 = np.array([q4, q3, -q2, -q1])
    A2 = np.array([-q3, q4, q1, -q2])
    A3 = np.array([q2, -q1, q4, -q3])
    A4 = np.array([q1, q2, q3, q4])

    A1 = A1.T
    A2 = A2.T
    A3 = A3.T
    A4 = A4.T
    A_stack = np.zeros((4*len(A1), 4))
    # import code
    # code.interact(local=locals())
    A_stack[0:(4*len(A1)):4] = A1
    A_stack[1:(4*len(A1)):4] = A2
    A_stack[2:(4*len(A1)):4] = A3
    A_stack[3:(4*len(A1)):4] = A4
    A = A_stack.reshape((len(A1), 4, 4))

    pqr0 = 2*np.matmul(A, rateQ.reshape((len(rateQ), -1, 1)))
    
    pqr = pqr0[:, :3]

    return pqr



def Eul2Quat_JPL(theta):
    _quatHam = Eul2Quat(theta)
    _quatJPL = _quatHam.copy()
    _quatJPL[:, :3] = _quatHam[:, 1:]
    _quatJPL[:, 3] = _quatHam[:, 0]
    return _quatJPL



def Quat2Eul_JPL(quat):
    _quat = quat.copy()
    _quat[:, 0] = quat[:, 3]
    _quat[:, 1:] = quat[:, :3]
    return Quat2Eul(quat)


def PQR_to_PhiThetaDot_Euler(att, pqr):
    s_phi = np.sin(att[:, 0]).reshape(-1)
    c_phi = np.cos(att[:, 0]).reshape(-1)
    c_theta = np.cos(att[:, 1]).reshape(-1)
    t_theta = np.tan(att[:, 1]).reshape(-1)

    R_1 = np.array([np.ones(len(s_phi)), s_phi*t_theta, c_phi*t_theta])
    R_2 = np.array([np.zeros(len(s_phi)), c_phi, -1*s_phi])
    R_3 = np.array([np.zeros(len(s_phi)), s_phi/c_theta, c_phi/c_theta])

    R_1 = R_1.T
    R_2 = R_2.T
    R_3 = R_3.T
    R_stack = np.zeros((3*len(R_1), 3))
    R_stack[0:(3*len(R_1)):3] = R_1
    R_stack[1:(3*len(R_1)):3] = R_2
    R_stack[2:(3*len(R_1)):3] = R_3
    R = R_stack.reshape((len(R_1), 3, 3))
    
    att_dot = np.matmul(R, pqr.reshape(len(att), -1, 1))
    return att_dot.reshape(pqr.shape)



def PhiThetaDot_to_PQR_Euler(att, phithetadot):
    s_phi = np.sin(att[:, 0]).reshape(-1)
    c_phi = np.cos(att[:, 0]).reshape(-1)
    c_theta = np.cos(att[:, 1]).reshape(-1)
    s_theta = np.sin(att[:, 1]).reshape(-1)

    R_1 = np.array([np.ones(len(s_phi)), np.zeros(len(s_phi)), -1*s_theta])
    R_2 = np.array([np.zeros(len(s_phi)), c_phi, s_phi*c_theta])
    R_3 = np.array([np.zeros(len(s_phi)), -s_phi, c_phi*c_theta])

    R_1 = R_1.T
    R_2 = R_2.T
    R_3 = R_3.T
    R_stack = np.zeros((3*len(R_1), 3))
    R_stack[0:(3*len(R_1)):3] = R_1
    R_stack[1:(3*len(R_1)):3] = R_2
    R_stack[2:(3*len(R_1)):3] = R_3
    R = R_stack.reshape((len(R_1), 3, 3))
    
    pqr = np.matmul(R, phithetadot.reshape(len(att), -1, 1))
    return pqr.reshape(phithetadot.shape)



def QuatMul(Q1, Q2):
    '''Function to multiply two quaternion arrays. Q2 is applied first, then Q1 (i.e. Q2 in global frame and Q1 in local after Q2 is applied) 
    
    :param Q1: First quaternion, as array with shape [N, 4] where N is the number of samples
    :param Q2: Second quaternion, as array with shape [N, 4] where N is the number of samples

    :return: Product of Q1 and Q2
    '''
    Q_out = np.array([[Q1[:, 0]*Q2[:, 0] - Q1[:, 1]*Q2[:, 1] - Q1[:, 2]*Q2[:, 2] - Q1[:, 3]*Q2[:, 3]],
                    [Q1[:, 0]*Q2[:, 1] + Q1[:, 1]*Q2[:, 0] + Q1[:, 2]*Q2[:, 3] - Q1[:, 3]*Q2[:, 2]],
                    [Q1[:, 0]*Q2[:, 2] - Q1[:, 1]*Q2[:, 3] + Q1[:, 2]*Q2[:, 0] + Q1[:, 3]*Q2[:, 1]],
                    [Q1[:, 0]*Q2[:, 3] + Q1[:, 1]*Q2[:, 2] - Q1[:, 2]*Q2[:, 1] + Q1[:, 3]*Q2[:, 0]]])
    return Q_out.T.reshape(-1, 4)
