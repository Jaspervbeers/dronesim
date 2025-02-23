from controllers import PID as PIDController
import os
import json
import numpy as np
from funcs.angleFuncs import Eul2Quat, QuatRot, PhiThetaDot_to_PQR_Euler, Quat2Eul
from scipy.signal import butter, sosfiltfilt

class controller:

    def __init__(self, RatePIDSignMask, noiseBlock, PIDGainPath, PIDGainFile = None):

        self.name = 'droneUpsetController_PID'
        self.noiseBlock = noiseBlock

        if PIDGainFile is None:
            PIDGainFile = 'droneUpsetController_PIDGains.json'

        with open(os.path.join(PIDGainPath, PIDGainFile), 'rb') as f:
            gainDict = json.load(f)
        
        # Extract PID gains and store them in local variables
        ratePID = self.extractPIDsFromJSON(gainDict['RateToRotorSpeed'])
        # Need to modify the PID values of the rate controller based on the 
        # rotor configuration
        for i, PID in enumerate(ratePID):
            PID.P = (PID.P.reshape(-1) * RatePIDSignMask[:, i]).reshape(-1, 1)
            PID.I = (PID.I.reshape(-1) * RatePIDSignMask[:, i]).reshape(-1, 1)
            PID.D = (PID.D.reshape(-1) * RatePIDSignMask[:, i]).reshape(-1, 1)

        # RatePID maps rotational rates (p, q, r) to rotor speeds (w1, w2, w3, w4)
        self.ratePID = ratePID
        # attPID maps attitude (roll, pitch, yaw) to rates (p, q, r)
        self.attPID_PhiTheta = self.extractPIDsFromJSON(gainDict['AttToRate'])
        self.attPID_Psi = self.extractPIDsFromJSON(gainDict['AttToRate'])

        # velPID maps the velocity to accelerations
        self.velPID = self.extractPIDsFromJSON(gainDict['VelToAcc'])
        self.velNdes = self.extractPIDsFromJSON(gainDict['VelToN'])

        # posPID maps position (x, y, z) to velocity (u, v, w)
        self.posPID = self.extractPIDsFromJSON(gainDict['PosToVel'])

        self.xi_prev = 0
        self.n_prev = np.array([0, 0, -1]).reshape(-1, 3)
        self.principleAxisCorrectionMapper = {False:self._correctPrincipleAxis_init, True:self._correctPrincipleAxis}

        self.inRecovery = False
        self.xyz_prev = np.array([0, 0, 1]).reshape(-1, 3)
        
        self.enforceAngleContinuity = {False:self._enforceAngleContinuity_init, True:self._enforceAngleContinuity}
        self.deltaAtt_PhiTheta_prev = np.array([0, 0, 0]).reshape(-1, 3)

        self.n_des_history = []
        self.VE_history = []

        return None

    def extractPIDsFromJSON(self, PIDJSON):
        P = np.array(np.matrix(PIDJSON['P']))
        I = np.array(np.matrix(PIDJSON['I']))
        D = np.array(np.matrix(PIDJSON['D']))
        PID = []
        for col in range(P.shape[1]):
            PID.append(PIDController.PID(P[:, col].reshape(-1, 1), I[:, col].reshape(-1, 1), D[:, col].reshape(-1, 1)))
        return PID


    def control(self, simVars):
        # Unpack necessary variables
        step = simVars['currentTimeStep_index']
        self.addSensorNoise(simVars)                # state_noisy for the current step is updated here
        state = simVars['state_noisy'][step].copy()        # Use noisy state as this is what the controller 'sees'
        omega = simVars['inputs'][step].copy()
        reference = simVars['reference'][step]
        dt = simVars['dt']
        self.simVars = simVars
        self.step = step
        droneParams = simVars['model'].droneParams

        '''
        Norm-vector magnitude determination
        '''
        # Find desired velocity in E-frame from desired positions
        # NOTE: np.nan for uncontrolled positions
        V_ref_E = self._positionalControl(state, reference, dt)
        uncontrolledV = np.where(np.isnan(V_ref_E))[1]
        # If uncontrolled, set desired value to current state s.t. error = 0
        if len(uncontrolledV):
            V_Eframe = QuatRot(Eul2Quat(state[:, :3]), state[:, 3:6], rot = 'B2E')
            V_ref_E[:, uncontrolledV] = V_Eframe[:, uncontrolledV]

        # TODO: Split into individual xyz components such that individual components can be controlled while others not
        # Rely on velocity for n_des
        reference[:, 3:6] = V_ref_E
        V = reference[:, 3:6]

        # Velocity error in E-Frame
        velB_E = QuatRot(Eul2Quat(state[:, :3]), state[:, 3:6], rot = 'B2E')
        Ve = V - velB_E
        V_n_des = self._PIDLoop(V, velB_E, self.velNdes, dt)

        n_des = np.array([0, 0, -1]).reshape(-1, 3)*2 + V_n_des
        n_des_mag = np.sqrt(np.sum(n_des**2))

        '''
        Altitude control
        '''
        # Scale magnitude of n_des for altitude control 
        deltaAcc = -1*self._PIDLoop(QuatRot(Eul2Quat(state[:, :3]), V, rot = 'E2B'), state[:, 3:6], self.velPID, dt)
        deltaMag = np.sum(deltaAcc)

        PW = 10
        d_control_w = PW*deltaMag

        for i in range(len(omega[0, :])):
            omega[:, i] = np.max((droneParams['idle RPM'], np.min((droneParams['max RPM'], (omega[:, i] + d_control_w)[0]))))

        '''
        Attitude control
        '''
        n_des_norm = n_des/n_des_mag # Unit vector, in earth frame
        n_current_norm = QuatRot(Eul2Quat(state[:, :3]), np.array([0, 0, -1]).reshape(1, -1), rot = 'B2E') # Thurst vector, always pointing along -z in body -> transform to E-frame

        xi = self._getAngleBetweenTwoVectors(n_des_norm.reshape(-1), n_current_norm.reshape(-1))
        if np.abs(xi) > 0.9*np.pi/2:
            A = xi/2 # Get half angle to set as 'intermediate reference' This is to avoid issues when n_des_norm and n_current_norm are anti-parallel
        else:
            A = xi
        M = A

        # Convert to quaternions
        xyz = np.cross(n_des_norm, n_current_norm)

        # xyz = self.principleAxisCorrectionMapper[bool(step)](xi, n_des_norm, n_current_norm, self._validateXYZ(xi, xyz))
        xyz = self.principleAxisCorrectionMapper[bool(step)](xi, n_des_norm, n_current_norm, xyz)

        qw = np.array([np.cos(M/2)])
        qx = np.sin(M/2)*xyz[:, 0]
        qy = np.sin(M/2)*xyz[:, 1]
        qz = np.sin(M/2)*xyz[:, 2]

        quat = np.array((qw, qx, qy, qz))

        # Note, deltaAtt only controls roll and pitch, so yaw control needs to be separate 
        # _deltaAtt_PhiTheta = -1*Quat2Eul(quat)
        # deltaAtt_PhiTheta = self.enforceAngleContinuity[bool(step)](self.deltaAtt_PhiTheta_prev, _deltaAtt_PhiTheta)
        
        deltaAtt_PhiTheta = -1*Quat2Eul(quat)
        
        commandedRate_R_E = self._PIDLoop(reference[:, :3], state[:, :3], self.attPID_Psi, dt)
        commandedRate_R_B = PhiThetaDot_to_PQR_Euler(state[:, :3], commandedRate_R_E)
        # commandedRate_R_B = commandedRate_R_E
        if np.abs(xi) > 0.95*np.pi/2:
            commandedRate_R_B = commandedRate_R_E    
        # commandedRate_R_B = self._PIDLoop(reference[:, :3], state[:, :3], self.attPID, dt)

        commandedRate_PQ_B = self._PIDLoop(deltaAtt_PhiTheta, deltaAtt_PhiTheta*0, self.attPID_PhiTheta, dt)
        # commandedRate_B = np.vstack((
        #     commandedRate_PQ_B[:, 0],
        #     commandedRate_PQ_B[:, 1],
        #     commandedRate_R_B[:, 2])).T
        commandedRate_B = np.vstack((
            commandedRate_PQ_B[:, 0],
            commandedRate_PQ_B[:, 1],
            commandedRate_R_B[:, 2] + commandedRate_PQ_B[:, 2])).T
        simVars['reference'][step][:, 6:9] = commandedRate_B
        dControl_rate = self._PIDLoop(commandedRate_B, state[:, 6:9], self.ratePID, dt)
        controlInput = omega + dControl_rate

        # dummyCMD = commandedRate_B.copy()
        # dummyCMD[:, 2] = 0
        # dummyRate = state[:, 6:9].copy()
        # dummyRate[:, 2] = 0
        # dControl_rate = self._PIDLoop(dummyCMD, dummyRate, self.ratePID, dt)
        # controlInput = omega + dControl_rate

        # if step >= int(7/dt):
        #     print(f'Xi = {xi}')
        #     print(f'deltaAtt = {deltaAtt_PhiTheta}')
        #     print(f'Attitude = {state[:, :3]}')
        #     print(f'Commanded Rate = {commandedRate_B}')
        #     print(f'n_des_norm = {n_des_norm}')

        #     import code
        #     code.interact(local=locals())

        self.xi_prev = xi
        self.n_prev = n_current_norm.copy()
        self.xyz_prev = xyz.copy()
        self.deltaAtt_PhiTheta_prev = deltaAtt_PhiTheta.copy()

        self.n_des_history.append(n_des)
        self.VE_history.append(Ve)

        return controlInput


    def _validateXYZ(self, xi, xyz):
        # if np.abs(xi) > np.pi/2:
        if np.abs(xi) > 0.95*np.pi:
            if not self.inRecovery:
                xyz[:, 0] = 1/np.sqrt(2)
                xyz[:, 1] = 1/np.sqrt(2)
                # xyz[:, 1] = 1
                self.inRecovery = True
            else:
                xyz = self.xyz_prev
        else:
            self.inRecovery = False
        return xyz


    def _enforceAngleContinuity_init(self, angleOld, angleCurrent):
        return angleCurrent


    def _enforceAngleContinuity(self, angleOld, angleCurrent):
        outAngle = angleCurrent.copy()
        for i in range(angleCurrent.shape[1]):
            if np.sqrt(np.square(angleOld[:, i] - angleCurrent[:, i])) > np.sqrt(np.square(angleOld[:, i] + angleCurrent[:, i])) and np.abs(angleCurrent[:, i]) > 0.1:
                outAngle[:, i] = -1*angleCurrent[:, i]
        return outAngle


    def _correctPrincipleAxis(self, xi, n_des_norm, n_current_norm, xyz):
        # As the quadrotor rotates in space, its principle axis also rotates
        # This function updates it accordingly 

        # Find expected change in current norm, based on observed change in angle
        deltaXi = xi - self.xi_prev
        qw = np.array([np.cos(deltaXi/2)])
        qx = np.sin(deltaXi/2)*xyz[:, 0]
        qy = np.sin(deltaXi/2)*xyz[:, 1]
        qz = np.sin(deltaXi/2)*xyz[:, 2]
        quat = np.array((qw, qx, qy, qz))
        nExp = QuatRot(quat.reshape(-1, 4), self.n_prev)

        # Find projection of difference in x-y plane (E-frame), use this to correct for yaw
        a, b = np.zeros((1, 3)), np.zeros((1, 3))
        a[:, :2] = n_current_norm[:, :2]
        b[:, :2] = nExp[:, :2]
        dT = self._getAngleBetweenTwoVectors(a, b)

        cT = n_des_norm
        qwT = np.array([np.cos(dT/2)])
        qxT = np.sin(dT/2)*cT[:, 0]
        qyT = np.sin(dT/2)*cT[:, 1]
        qzT = np.sin(dT/2)*cT[:, 2]
        quatT = np.array((qwT, qxT, qyT, qzT))
        _xyz = QuatRot(quatT.reshape(-1, 4), xyz)

        # Re-normalize
        n_xyz = np.sqrt(np.sum(np.square(_xyz)))
        if n_xyz == 0:
            n_xyz = 1
        xyz_corrected = _xyz / n_xyz

        return xyz_corrected


    def _correctPrincipleAxis_init(self, xi, n_des_norm, n_current_norm, xyz):
        return xyz


    def _getAngleBetweenTwoVectors(self, a, b):
        normAB = np.linalg.norm(a)*b
        normBA = np.linalg.norm(b)*a
        num = np.linalg.norm(normAB - normBA)
        den = np.linalg.norm(normAB + normBA)
        return 2*np.arctan2(num, den)


    def _positionalControl(self, state, reference, dt):
        pos = state[:, 9:12]
        posRef = reference[:, 9:12]
        uncontrolledIdx = np.where(np.isnan(posRef))[1]
        velCMD_E = self._parseCombinedPID(posRef.copy(), pos.copy(), self.posPID, dt)
        velCMD_E[:, uncontrolledIdx] = np.nan
        return velCMD_E

    # def _smoothen_n_des(self, n_des, window):
    #     n_des_history = np.array(self.n_des_history)
    #     if len(n_des_history) >= window:
    #         # import code
    #         # code.interact(local=locals()) 
    #         n_curr = n_des_history[-window:]
    #         # weights = np.arange(1, window+2, 1)
    #         weights = np.ones(window+1)
    #         n_des_smooth = np.sum((weights[::-1].reshape(1, -1, 1)*np.hstack((n_des.reshape(1, 1, 3), n_curr.reshape(1, -1, 3)))).reshape(-1, 3), axis =0)/np.sum(weights)
    #     elif len(n_des_history) == 0:
    #         n_des_smooth = n_des
    #     else:
    #         n_curr = n_des_history
    #         # weights = np.arange(1, len(n_des_history)+2, 1)
    #         weights = np.ones(len(n_des_history) + 1)
    #         n_des_smooth = np.sum((weights[::-1].reshape(1, -1, 1)*np.hstack((n_des.reshape(1, 1, 3), n_curr.reshape(1, -1, 3)))).reshape(-1, 3), axis =0)/np.sum(weights)

    #     return n_des_smooth.reshape(1, 3)


    # def _smoothen_n_des(self, n_des, window):
    #     n_des_history = np.array(self.n_des_history)
    #     cutoff = 1/window
    #     Fs = 1/0.004
    #     if len(n_des_history) >= window:
    #         n_curr = n_des_history[-window:]
    #         n_des_smooth = self.passFilter(np.hstack((n_curr.reshape(1, -1, 3), n_des.reshape(1, 1, 3))).reshape(-1, 3).T, cutoff, Fs, 'low').T[-1]
    #     else:
    #         n_des_smooth = n_des
    #     # elif len(n_des_history) == 0:
    #     #     n_des_smooth = n_des
    #     # else:
    #     #     n_curr = n_des_history
    #     #     n_des_smooth = self.passFilter(np.hstack((n_des.reshape(1, 3), n_curr.reshape(-1, 3))), cutoff, Fs, 'low')[-1]
    #     return n_des_smooth.reshape(1, 3)


    def _smoothenSignal(self, signal, cutoff, dt):
        window = int((1/cutoff)/dt)
        Fs = 1/dt
        if len(signal) >= window:
            signal_smooth = self.passFilter(signal[-window:].T, cutoff, Fs, 'low').T
        else:
            signal_smooth = signal
        return signal_smooth


    def butterFilt(self, cutoff, fs, filtType, order = 4):
        sos = butter(order, cutoff, fs=fs, btype = filtType, output = 'sos')
        return sos


    def passFilter(self, data, cutoff, fs, filtType, order = 4):
        sos = self.butterFilt(cutoff, fs, filtType, order = order)
        # y_out = np.zeros(data.shape)
        # for i in range(data.shape[0]):
        #     y_out[i, :] = filtfilt(b, a, data[i, :])
        y_out = sosfiltfilt(sos, data)
        return y_out


    def _parseCombinedPID(self, combinedReference, state, PIDList, dt):
        command = np.zeros((1, PIDList[0].outDim))
        for i, PID in enumerate(PIDList):
            if not np.isnan(combinedReference[:, i]):
                command += PID.control(combinedReference[:, i] - state[:, i], dt)
        return command


    def _PIDLoop(self, ref, actual, PID, dt):
        dControlInput = self._parseCombinedPID(ref, actual, PID, dt)
        return dControlInput


    def addSensorNoise(self, simVars):
        self.noiseBlock.addStateNoise(simVars)