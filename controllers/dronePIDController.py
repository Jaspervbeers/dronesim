from controllers import PID as PIDController
import os
import json
import numpy as np
from funcs.angleFuncs import Eul2Quat, QuatRot, PhiThetaDot_to_PQR_Euler

class controller:

    def __init__(self, RatePIDSignMask, noiseBlock, PIDGainPath, PIDGainFile = None):

        self.name = 'drone_PID'
        self.noiseBlock = noiseBlock

        if PIDGainFile is None:
            PIDGainFile = 'dronePIDGains.json'
         
        if os.path.exists(os.path.join(PIDGainPath, PIDGainFile)):
            with open(os.path.join(PIDGainPath, PIDGainFile), 'rb') as f:
                gainDict = json.load(f)
        else:
            with open(f'controllers/{PIDGainFile}', 'rb') as f:
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
        self.attPID = self.extractPIDsFromJSON(gainDict['AttToRate'])
        # velPID_att maps lateral velocity (u, v) to attitude (roll, pitch)
        self.velPID_att = self.extractPIDsFromJSON(gainDict['LateralVelToAtt'])
        # velPID maps velocity (u, v, w) to rotor speeds (w1, w2, w3, w4) for thrust control 
        self.velPID = self.extractPIDsFromJSON(gainDict['VelToRotorSpeed'])
        # posPID maps position (x, y, z) to velocity (u, v, w)
        self.posPID = self.extractPIDsFromJSON(gainDict['PosToVel'])

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
        step = simVars['currentTimeStep_index']
        # state = simVars['state'][step]
        self.addSensorNoise(simVars)                # state_noisy for the current step is updated here
        state = simVars['state_noisy'][step]        # Use noisy state as this is what the controller 'sees'
        omega = simVars['inputs'][step]
        reference = simVars['reference'][step]
        dt = simVars['dt']

        # reference[:, 3:6] = self._positionalControl(state, reference, dt)
        vel_ref_E = self._positionalControl(state, reference, dt)
        reference[:, 3:6] = QuatRot(Eul2Quat(state[:, :3]), vel_ref_E, rot='E2B')
        dAttRef_vel, dControl_vel = self._velocityControl(state, reference, dt)
        attRef = reference[:, :3]
        attRef[:, :2] = attRef[:, :2] + dAttRef_vel
        attRef[:, :2] = self._sigmoidSaturate(attRef[:, :2])
        uncontrolledIdx = np.where(np.isnan(attRef))[1]
        # Turn attitude reference into a commanded rotational rate
        commandedRate_E = self._PIDLoop(attRef, state[:, :3], self.attPID, dt)
        commandedRate_B = PhiThetaDot_to_PQR_Euler(state[:, :3], commandedRate_E)
        # NOTE Should implement some sort of weighted average of commandedRate_E and commandedRate_B_true based on current theta to avoid singularities
        # commandedRate_B = commandedRate_E # Note, even though this is not exactly the same, PID works based on errors should give approximately the desired results.
        # Input commandedRate into rate PID to generate change in rotor speeds
        
        dummyRate = state[:, 6:9].copy()
        dummyRate[:, uncontrolledIdx] = 0
        commandedRate_B[:, uncontrolledIdx] = 0

        reference[:, 6:9] = commandedRate_B

        # # Do not control yaw
        # dummyRate[:, 2] = 0
        # commandedRate_B[:, 2] = 0

        # dummyRate = state[:, 6:9].copy()
        # dummyRate[:, 1:] = 0
        # commandedRate_B = dummyRate.copy()
        # commandedRate_B[:, 0] = 30
        # dControl_vel = dControl_vel*0

        # # Only control pitch rate
        # commandedRate_B = commandedRate_B.copy()*0
        # commandedRate_B[:, 1] = 100


        dControl_rate = self._PIDLoop(commandedRate_B, dummyRate, self.ratePID, dt)
        controlInput = omega + dControl_rate + dControl_vel
        return controlInput


    def _velocityControl(self, state, reference, dt):
        att = state[:, :3]
        velRef = reference[:, 3:6]
        quat = Eul2Quat(att)
        velE = QuatRot(quat, state[:, 3:6], rot = 'B2E')
        uncontrolledIdx = np.where(np.isnan(velRef[:, :2]))[1]
        dCommand_SINatt = self._parseCombinedPID(velRef[:, :2].copy(), velE[:, :2].copy(), self.velPID_att, dt)    
        dCommand_SINatt[:, uncontrolledIdx] = 0
        dCommand_w = self._parseCombinedPID(velRef, velE, self.velPID, dt)     
        return dCommand_SINatt, dCommand_w


    def _positionalControl(self, state, reference, dt):
        pos = state[:, 9:12]
        posRef = reference[:, 9:12]
        uncontrolledIdx = np.where(np.isnan(posRef))[1]
        velCMD_E = self._parseCombinedPID(posRef.copy(), pos.copy(), self.posPID, dt)
        velCMD_E[:, uncontrolledIdx] = np.nan
        return velCMD_E


    def _parseCombinedPID(self, combinedReference, state, PIDList, dt):
        command = np.zeros((1, PIDList[0].outDim))
        for i, PID in enumerate(PIDList):
            if not np.isnan(combinedReference[:, i]):
                command += PID.control(combinedReference[:, i] - state[:, i], dt)
        return command


    def _sigmoidSaturate(self, x, a = 0.95*np.pi, b = 1.5):
        c = -a/2
        y = a/(1+np.exp(-b*x)) + c
        return y 


    def _PIDLoop(self, ref, actual, PID, dt):
        dControlInput = self._parseCombinedPID(ref, actual, PID, dt)
        return dControlInput

    def addSensorNoise(self, simVars):
        self.noiseBlock.addStateNoise(simVars)