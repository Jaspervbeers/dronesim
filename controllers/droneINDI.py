'''
INDI controller scheme for quadrotor control

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 23-05-2023
'''

from controllers import PID as PIDController
import numpy as np
import os
import json

from funcs.angleFuncs import Eul2Quat, QuatRot, PhiThetaDot_to_PQR_Euler

class controller:

    def __init__(self, signMask, noiseBlock, PIDGainPath, PIDGainFile = None):
        '''Initialize the (INDI) controller object

        :param signMask: Mapping of rotor speed layout to control rolling, pitching, and yawing inputs - 3x4 np.array
        :param noiseBlock: Noise block object (see noiseDisturbance > droneSensorNoise.py)
        :param PIDGainPath: Directory of .json file containing the PID Gain values - string
        :param PIDGainFile: (Optional) name of .json file with PID Gain values - string or None. Default is None; default file name is used.

        :returns: None
        '''        
        self.name = 'drone_INDI'
        self.signMask = signMask
        self.noiseBlock = noiseBlock
        self.eRPM2Rads = 2*np.pi/60
        self.eRPM2Rads_2 = self.eRPM2Rads**2


        if PIDGainFile is None:
            PIDGainFile = 'droneINDIGains.json'
         
        if os.path.exists(os.path.join(PIDGainPath, PIDGainFile)):
            with open(os.path.join(PIDGainPath, PIDGainFile), 'rb') as f:
                gainDict = json.load(f)
        else:
            with open(f'controllers/{PIDGainFile}', 'rb') as f:
                gainDict = json.load(f)

        # attPID maps attitude (roll, pitch, yaw) to rates (p, q, r)
        self.attPID = self.extractPIDsFromJSON(gainDict['AttToRate'])
        # posPID maps position (x, y, z) to velocity (u, v, w)
        self.posPID = self.extractPIDsFromJSON(gainDict['PosToVel'])
        # velAccPID maps velocity to acceleration
        self.velAccPID = self.extractPIDsFromJSON(gainDict['VelToAcc'])

        self.gains_ATT = np.array(np.matrix(gainDict['VirtualControl_Rate']['P'])).reshape(-1)
        self.gains_ALT = np.array(np.matrix(gainDict['VirtualControl_Acc']['P'])).reshape(-1)

        self.commandedInput_ATT_prev = np.zeros(3)
        self.commandedInput_ALT_prev = np.zeros(3)
        self.commandedInput_prev = np.zeros(4)

        return None


    def extractPIDsFromJSON(self, PIDJSON):
        '''Utility function to extract gains from loaded .json file

        :param PIDJSON: PID mapping - Dictionary
        
        :returns: PID object (see controllers > PID.py)
        '''        
        P = np.array(np.matrix(PIDJSON['P']))
        I = np.array(np.matrix(PIDJSON['I']))
        D = np.array(np.matrix(PIDJSON['D']))
        PID = []
        for col in range(P.shape[1]):
            PID.append(PIDController.PID(P[:, col].reshape(-1, 1), I[:, col].reshape(-1, 1), D[:, col].reshape(-1, 1)))
        return PID



    def control(self, simVars):
        '''Apply control to the system (quadrotor)
        
        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 

        :returns: Commanded rotor speeds - array-like
        '''        
        # Extract relevant variables
        self.simVars = simVars
        self.step = simVars['currentTimeStep_index']
        self.dt = simVars['dt']
        self.addSensorNoise(simVars)
        self.state = simVars['state_noisy'][self.step].copy()
        self.reference = simVars['reference'][self.step]
        self.stateDerivative = simVars['stateDerivative'][self.step].copy()
        self.omega = simVars['inputs'][self.step].copy()
        self.droneParams = simVars['model'].droneParams
        self.moment = simVars['moments'][self.step].copy()
        self.force = simVars['forces'][self.step].copy()

        # Get rotor speed command
        self.accelerationLoop()
        self.rateLoop()

        self.commandedOmega = self.commandedOmega_INDI

        return self.commandedOmega


    def rateLoop(self):
        '''Run rate loop of the controller. Converts attitude commands (and thrust from acceleration loop) into rotor speeds.
        
        :returns: None, modifies object attributes in place. 
        '''        
        # State vector:
        # [roll, pitch, yaw, u, v, w, p, q, r, x, y, z]

        # Rate, p q r
        self.rate = self.state[:, 6:9]
        self.rate_des = self.reference[:, 6:9]
        self.rate_error = self.rate - self.rate_des

        # Rate dot
        self.rate_dot = self.stateDerivative[:, 6:9]
        self.rate_dot_des = np.zeros(self.rate_dot.shape)

        # Define error vectors
        self.error = self.rate_error
        self.x_dot_des = self.rate_dot_des

        # x_dot
        self.x_dot_meas = self.rate_dot

        # Define virtual input as x_dot = v = -1*k*error + x_dot_des
        self.virtualInput = -self.gains_ATT * self.error + self.x_dot_des

        # Get commmanded input, in the form [Mx, My, Mz]
        self.commandedInput_ATT = np.matmul(np.linalg.inv(self._controlEffectiveness_ATT()), (self.virtualInput - self.x_dot_meas).reshape(-1)) + self.commandedInput_ATT_prev
        
        # Concatenate [Mx, My, Mz, T] to convert into rotor speeds
        self.commandedInput = np.hstack((self.commandedInput_ATT.reshape(3), self.commandedT))

        # Convert commandedInput into commanded rotor speeds
        commandedOmega2 = np.matmul(self.FM2OmegaMapping(), self.commandedInput.T)
        # Enforce positive omega
        commandedOmega2 = commandedOmega2.copy()
        commandedOmega2[commandedOmega2 < 0] = 0
        # Convert from omega^2 to omega 
        self.commandedOmega_INDI = np.sqrt(commandedOmega2)

        self.commandedInput_ALT_prev = self.commandedInput_ALT.copy()
        self.commandedInput_ATT_prev = self.commandedInput_ATT.copy()
        self.commandedInput_prev = self.commandedInput.copy()


    def accelerationLoop(self):
        '''Run acceleration loop of the controller. Converts position into attitude and thrust commands. 
        
        :returns: None, modifies object attributes in place. 
        '''        
        # State vector:
        # [roll, pitch, yaw, u, v, w, p, q, r, x, y, z]

        # Position to Velocity in E-frame
        posRef_E = self.reference[:, 9:12].copy()
        velRef_E = self._PIDLoop(posRef_E, self.state[:, 9:12], self.posPID, self.dt)
        vel_E = QuatRot(Eul2Quat(self.state[:, :3]), self.state[:, 3:6], rot = 'B2E')
        self.reference[:, 3:6] = QuatRot(Eul2Quat(self.state[:, :3]), velRef_E, rot='E2B')

        # Velocity error in E-frame
        self.vel_E_error = vel_E - velRef_E

        # Velocity to Acc in E-Frame
        self.accRef_E = self._PIDLoop(velRef_E, vel_E, self.velAccPID, self.dt)
        
        # Virtual input
        self.virtualInput_ALT = -1*self.gains_ALT*self.vel_E_error + self.accRef_E

        # V_dot measured
        self.vel_dot_E_meas = QuatRot(Eul2Quat(self.state[:, :3]), self.stateDerivative[:, 3:6], rot = 'B2E')

        # Virtual input
        self.virtualInput_ALT[np.isnan(posRef_E)] = self.vel_dot_E_meas[np.isnan(posRef_E)]

        # Commanded input
        # Do not control np.nan positions
        self.virtualInput_ALT[np.isnan(posRef_E)] = self.vel_dot_E_meas[np.isnan(posRef_E)]
        self.commandedInput_ALT_prev[:2] = self.state[:, :2] # Need to do this since control effectiveness is linearized about current point, not previous command. T is already aligned properly in _controlEffectiveness_ALT()
        self.commandedInput_ALT = self.droneParams['m']*np.matmul(np.linalg.inv(self._controlEffectiveness_ALT()), (self.virtualInput_ALT - self.vel_dot_E_meas).T).reshape(-1) + self.commandedInput_ALT_prev
        self.commandedT = -1*self.commandedInput_ALT[-1]

        # Modify attitude reference
        attRef = self.reference[:, :3]
        attRef[:, :2] -= self.commandedInput_ALT[:2]

        # Attitude -> Rate
        rate_ref_E = self._PIDLoop(attRef, self.state[:, :3], self.attPID, self.dt)
        rate_ref_B = PhiThetaDot_to_PQR_Euler(self.state[:, :3], rate_ref_E)
        self.reference[:, 6:9] = rate_ref_B


    def _controlEffectiveness_ALT(self):
        '''Build thrust-component (in earth frame) control effectiveness matrix [Tx_E, Ty_E, Tz_E]
        
        :returns: 3x3 control effectiveness matrix 
        '''        
        att = self.state[:, :3]
        cPhi = np.cos(att[:, 0])
        sPhi = np.sin(att[:, 0])
        cTheta = np.cos(att[:, 1])
        sTheta = np.sin(att[:, 1])
        cPsi = np.cos(att[:, 2])
        sPsi = np.sin(att[:, 2])

        self.FM2OmegaMapping()
        T_mapping = self.A[3, :]
        T = np.matmul(T_mapping, np.square(self.omega.reshape(-1)))

        G = np.array([
            [(cPhi*sPsi - sPhi*cPsi*sTheta)*T, (cPhi*cPsi*cTheta)*T, sPhi*sPsi + cPhi*cPsi*sTheta],
            [(-1*sPhi*sPsi*sTheta - cPsi*cPhi)*T, (cPhi*sPsi*cTheta)*T, cPhi*sPsi*sTheta - cPsi*sPhi],
            [(-cTheta*sPhi)*T, -sTheta*cPhi*T, cPhi*cTheta]
        ])
        return G.reshape(3, 3)


    def _controlEffectiveness_ATT(self):
        '''Build rate control effectiveness matrix [Mx (Moment about x-axis), My (Moment about y-axis), Mz (Moment about z-axis)]
        
        :returns: 3x3 control effectiveness matrix 
        '''        
        controlEffectiveness = np.linalg.inv(self.droneParams['Iv'])
        return controlEffectiveness

    
    def FM2OmegaMapping(self):
        '''Maps the desired forces and moments to rotor speeds, based on estimates of the motor effectiveness (HDBeetle)
        
        :returns: 4x4 Matrix that maps forces and moments to rotor speeds. 
        '''
        # Mapping from F, M -> w1, w2, w3, w4
        if self.simVars['model']._simplifiedModel is None:
            self.kappaFz = 8.720395416240164e-07
            self.kappaMx = 2.99e-08 
            self.kappaMy = 3.69e-08
            self.kappaMz = 4.29e-09
        else:
            # Take w_2 variants as controller uses INDI Moments -> w^2 (rotor speed squared)
            # Need to convert units as kappaFx_w_2 expects w_2 in rad/s whereas controllers uses eRPM
            self.kappaFz = self.simVars['model']._simplifiedModel['kappaFz_w_2']*self.eRPM2Rads_2
            self.kappaMx = self.simVars['model']._simplifiedModel['kappaMx_w_2']*self.eRPM2Rads_2
            self.kappaMy = self.simVars['model']._simplifiedModel['kappaMy_w_2']*self.eRPM2Rads_2
            self.kappaMz = self.simVars['model']._simplifiedModel['kappaMz_w_2']*self.eRPM2Rads_2


        A_rate = np.array([
            [self.kappaMx, self.kappaMx, self.kappaMx, self.kappaMx],
            [self.kappaMy, self.kappaMy, self.kappaMy, self.kappaMy],
            [self.kappaMz, self.kappaMz, self.kappaMz, self.kappaMz]
        ])*self.signMask.T
        A_w = np.array([[self.kappaFz, self.kappaFz, self.kappaFz, self.kappaFz]])
        self.A = np.vstack((A_rate, A_w))
        self.Ainv = np.linalg.inv(self.A)
        return self.Ainv

    
    def _parseCombinedPID(self, combinedReference, state, PIDList, dt):
        '''Apply PID control block

        :param combinedReference: Reference value(s) to track at current time instant, as array
        :param state: Current state at time instant, as array (same shape as combinedReference)
        :param PIDList: List of PIDController objects per controlled state (instance of extractPIDsFromJSON() output)
        :param dt: Time step, in seconds

        :returns: PID output command, as array (same shape as state)
        '''        
        command = np.zeros((1, PIDList[0].outDim))
        for i, PID in enumerate(PIDList):
            if not np.isnan(combinedReference[:, i]):
                command += PID.control(combinedReference[:, i] - state[:, i], dt)
        return command


    def _PIDLoop(self, ref, actual, PID, dt):
        '''Apply PID control block, wrapper for _parseCombinedPID()

        :param ref: Reference value(s) to track at current time instant, as array
        :param actual: Current state at time instant, as array (same shape as ref)
        :param PID: List of PIDController objects per controlled state (instance of extractPIDsFromJSON() output)
        :param dt: Time step, in seconds

        :returns: PID output command, as array (same shape as state)
        '''
        dControlInput = self._parseCombinedPID(ref, actual, PID, dt)
        return dControlInput


    def addSensorNoise(self, simVars):
        '''Add noise to system states. Noise characteristics are determined by self.noiseBlock (default is Gaussian)
        
        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 
        '''        
        self.noiseBlock.addStateNoise(simVars)