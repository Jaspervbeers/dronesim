'''
INDI controller scheme for quadrotor control (normal vector control version.)

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 23-05-2023
'''

# Global imports
import numpy as np
import os
import json

# Local imports
from controllers import PID as PIDController
from funcs.angleFuncs import Eul2Quat, QuatRot, PhiThetaDot_to_PQR_Euler, Quat2Eul

# INDI Controller class
class controller:

    def __init__(self, signMask, noiseBlock, PIDGainPath, PIDGainFile = None):
        '''Initialize the (INDI_nDes) controller object

        :param signMask: Mapping of rotor speed layout to control rolling, pitching, and yawing inputs - 3x4 np.array
        :param noiseBlock: Noise block object (see noiseDisturbance > droneSensorNoise.py)
        :param PIDGainPath: Directory of .json file containing the PID Gain values - string
        :param PIDGainFile: (Optional) name of .json file with PID Gain values - string or None. Default is None; default file name is used.

        :returns: None
        '''
        self.name = 'drone_INDI_nDes'
        self.signMask = signMask
        self.noiseBlock = noiseBlock
        self.eRPM2Rads = 2*np.pi/60
        self.eRPM2Rads_2 = self.eRPM2Rads**2


        if PIDGainFile is None:
            PIDGainFile = 'droneINDIGains_nDes.json'
         
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

        # gains inner defines the INDI rate and vertical velocity error gains
        self.gains_inner = np.array(np.matrix(gainDict['VirtualControl_inner']['P'])).reshape(-1)

        # Initialize one-step-before values
        self.commandedInput_prev = np.zeros(4)
        self.deltaAtt_prev = np.zeros(3)

        return None


    def extractPIDsFromJSON(self, PIDJSON):
        '''Utility function to extract gains from loaded .json file

        :param PIDJSON: PID mapping - Dictionary
        
        :returns: PID object (see controllers > PID.py)
        '''
        # Extract gains
        P = np.array(np.matrix(PIDJSON['P']))
        I = np.array(np.matrix(PIDJSON['I']))
        D = np.array(np.matrix(PIDJSON['D']))
        PID = []
        # Construct PID objects
        for col in range(P.shape[1]):
            PID.append(PIDController.PID(P[:, col].reshape(-1, 1), I[:, col].reshape(-1, 1), D[:, col].reshape(-1, 1)))
        return PID


    def control(self, simVars):
        '''Apply control to the system (quadrotor)
        
        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 

        :returns: Commanded rotor speeds - array-like
        '''
        # Extract relevant variables for the current time step
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

        # Call acceleration loop (pos -> acc -> att)
        self.accelerationLoop()
        # Call rate loop (att & Thrust -> rotor speeds)
        self.rateLoop()

        self.commandedOmega = self.commandedOmega_INDI
        return self.commandedOmega


    def rateLoop(self):
        '''Run rate loop of the controller. Converts attitude and thrust commands into rotor speeds. 
        
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

        # Vertical velocity in E-Frame, w 
        #   NOTE: accelerationLoop() sets vel_E command
        self.w = self.vel_E[:, 2]
        self.w_des = self.vel_ref_E[:, 2]
        self.w_error = self.w - self.w_des

        # w dot (E-frame)
        self.vel_dot_B = self.stateDerivative[:, 3:6]
        self.vel_dot_E = QuatRot(Eul2Quat(self.state[:, :3]), self.vel_dot_B, rot = 'B2E')
        self.w_dot = self.vel_dot_E[:, 2]
        self.w_dot_des = np.zeros(self.w_dot.shape)

        # Define error vectors
        self.error = np.hstack((self.rate_error.reshape(3), self.w_error))
        self.x_dot_des = np.hstack((self.rate_dot_des.reshape(3), self.w_dot_des))

        # Define virtual input as x_dot = v = -1*k*error + x_dot_des
        self.virtualInput = -self.gains_inner * self.error + self.x_dot_des

        # x_dot
        self.x_dot_meas = np.hstack((self.rate_dot.reshape(3), self.w_dot))

        # Get commmanded input, in the form [Mx, My, Mz, T]
        self.commandedInput = np.matmul(np.linalg.inv(self._controlEffectiveness()), (self.virtualInput - self.x_dot_meas)) + self.commandedInput_prev

        # Convert commandedInput into commanded rotor speeds
        commandedOmega2 = np.matmul(self.FM2OmegaMapping(), self.commandedInput.T)
        # Enforce positive omega
        commandedOmega2 = commandedOmega2.copy()
        commandedOmega2[commandedOmega2 < 0] = 0
        # Convert from omega^2 to omega 
        self.commandedOmega_INDI = np.sqrt(commandedOmega2)
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
        self.vel_E = QuatRot(Eul2Quat(self.state[:, :3]), self.state[:, 3:6], rot = 'B2E')
        self.reference[:, 3:6] = QuatRot(Eul2Quat(self.state[:, :3]), velRef_E, rot='E2B')
        self.vel_ref_E = velRef_E

        velRef_E[np.isnan(posRef_E)] = self.vel_E[np.isnan(posRef_E)]

        # Velocity error in E-frame
        vel_E_error = self.vel_E - velRef_E

        # Velocity to Acc in E-Frame
        self.accRef_E = self._PIDLoop(velRef_E, self.vel_E, self.velAccPID, self.dt)
        
        # Virtual input
        gains_ALT = np.array([1, 1, 1])
        virtualInput_ALT = -1*gains_ALT*vel_E_error + self.accRef_E

        # Get thrust components in E-frame
        # NOTE: Actual thrust setting handled by inner loop, outerloop 'thrust' simply used to determine 
        #       suitable attitude. 
        commandedInput_ALT = self.droneParams['m']*(virtualInput_ALT - self._getVelDot_E(self.state[:, :3], self.state[:, 6:9], self.state[:, 3:6]))

        # Find orientation between desired and current thrust direction in E-frame
        nT_B = np.array([0, 0, -1]).reshape(1, -1)
        nT_E = QuatRot(Eul2Quat(self.state[:, :3]), nT_B, rot = 'B2E')

        nT_E_des = commandedInput_ALT/np.sqrt(np.sum(np.square(commandedInput_ALT)))
        nT_E_des[np.isnan(posRef_E)] = nT_E[np.isnan(posRef_E)]

        # Find angle between desired and current thrust direction in E-frame
        xi = self._getAngleBetweenTwoVectors(nT_E_des.reshape(-1), nT_E.reshape(-1))
        # Convert to quaternions
        xyz = np.cross(nT_E_des, nT_E)

        qw = np.array([np.cos(xi/2)])
        qx = np.sin(xi/2)*xyz[:, 0]
        qy = np.sin(xi/2)*xyz[:, 1]
        qz = np.sin(xi/2)*xyz[:, 2]

        quat = np.array((qw, qx, qy, qz))

        # Convert quaternion into Euler angles
        self.deltaAtt = Quat2Eul(quat)
        self.reference[:, :3] -= (self.deltaAtt + self.deltaAtt_prev)

        # Attitude -> Rate
        rate_ref_E = self._PIDLoop(self.reference[:, :3], self.state[:, :3], self.attPID, self.dt)
        # self.reference[:, 6:9] = rate_ref_E
        # Using PhiThetaDot_to_PQR_Euler can lead to issue for pitch = 90 deg
        rate_ref_B = PhiThetaDot_to_PQR_Euler(self.state[:, :3], rate_ref_E)
        self.reference[:, 6:9] = rate_ref_B

        self.deltaAtt_prev = self.deltaAtt.copy()


    def _controlEffectiveness(self):
        '''Build rate control effectiveness matrix [Mx (Moment about x-axis), My (Moment about y-axis), Mz (Moment about z-axis), T (Thrust)]
        
        :returns: 4x4 control effectiveness matrix 
        '''
        controlEffectiveness = np.zeros((4, 4))
        controlEffectiveness[:3, :3] = np.linalg.inv(self.droneParams['Iv'])
        controlEffectiveness[3, 3] = -1/self.droneParams['m']
        return controlEffectiveness


    def _getVelDot_E(self, att, rates, vel):
        '''Get (desired) acceleration in the earth frame. NOTE: Current implementation assumes we only want to counteract gravity (i.e. hover)
        
        :param att: Attitude, as an Nx3 array
        :param rates: Body rotational rates, as an Nx3 array
        :param vel: Body velocities, as an Nx3 array

        :returns: Acceleration in the earth frame
        '''
        return np.array([0, 0, self.droneParams['g']]).reshape(1, -1)


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


    def _getAngleBetweenTwoVectors(self, a, b):
        '''Function to determine the angle between two vectors, a and b
        
        :param a: First vector, as array with shape (3,)
        :param b: Second vector, as array with shape (3,)

        :returns: Angle, in radians, between a and b
        '''
        normAB = np.linalg.norm(a)*b
        normBA = np.linalg.norm(b)*a
        num = np.linalg.norm(normAB - normBA)
        den = np.linalg.norm(normAB + normBA)
        return 2*np.arctan2(num, den)
            