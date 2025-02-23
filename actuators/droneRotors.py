'''
Actuator dynamics for drone motors

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 03-05-2023
'''
# Imports
from numpy import zeros as np_zeros


# Base actuator dynamics class
class rotorDynamicsBase:

    def __init__(self, timeConstant, droneParameters, w0 = np_zeros((4, 1)), **kwargs):
        '''Initializes the rotoDynamicsBase object
        
        :param timeConstant: Actuator dynamics time constant, in seconds - Float
        :param droneParameters: Parameters of the quadrotor - Dictionary
        :param w0: Initial rotor speeds of each rotor - np.array-like of floats. Default is array of zeros.
        :param **kwargs: Additional keyword arguments to be passed, unused (Remains for legacy reasons)

        :return: None
        '''
        # Determine if rotor-specific time constants passed, or if all use the same value
        try:
            _ = len(timeConstant)
            self.tau = timeConstant
        except TypeError:
            self.tau = [timeConstant,]*len(w0)
        # Extract the saturation limits
        self.wUpper = droneParameters['max RPM']
        self.wLower = droneParameters['idle RPM']
        # Assign rotor-specific limits
        self.currentLims = []
        for i in range(w0.shape[0]):
            self.currentLims.append({'lower':float(self.wLower), 'upper':float(self.wUpper)})
        # Initialize one-step-before rotor speed values
        self.OutputCommand = w0.reshape(-1)
        self.OutputCommand_Rotor = w0.reshape(-1)[0]
        self.previousCommand = w0.reshape(-1)
    
    def saturate(self, i):
        '''Saturate the rotor, at index i, to rotor limits. Calls _saturate()
        
        :param i: Rotor index - integer
        
        :return: (Saturated) rotor speed value of rotor i
        '''
        return self._saturate(i, self.OutputCommand_Rotor)

    def _saturate(self, i, w):
        '''Saturate the rotor, at index i, to rotor limits.
        
        :param i: Rotor index - integer
        :param w: Rotor value - float
        
        :return: (Saturated) rotor speed value of rotor i
        '''
        # Check if rotor speed exceeds saturation limits
        if w < self.currentLims[i]['lower']:
            w = self.currentLims[i]['lower']            
        elif w > self.currentLims[i]['upper']:
            w = self.currentLims[i]['upper']
        return w
    
    def setInitial(self, w0):
        '''Set initial (or one step before) rotor speed values

        :param w0: Initial rotor speeds of each rotor - np.array-like of floats
        
        :return: None
        '''
        self.previousCommand = w0.reshape(-1)
        self.OutputCommand = w0.reshape(-1)
        self.OutputCommand_Rotor = w0.reshape(-1)[0]


# Simple actuator dynamics class; only rate limits & saturation
class rotorDynamicsV1(rotorDynamicsBase):

    def __init__(self, timeConstant, upper, lower, droneParameters, w0 = np_zeros((4, 1)), **kwargs):
        '''Initializes the rotoDynamicsV1 object
        
        :param timeConstant: Actuator dynamics time constant, in seconds - Float. Passed on to rotorDynamicsBase
        :param upper: Maximum allowable positive rate of change of rotor speed - Float.
        :param lower: Maximum allowable negative rate of change of rotor speed - Float.
        :param droneParameters: Parameters of the quadrotor - Dictionary. Passed on to rotorDynamicsBase
        :param w0: Initial rotor speeds of each rotor - np.array-like of floats. Passed on to rotorDynamicsBase. Default is array of zeros.
        :param **kwargs: Additional keyword arguments to be passed, unused (Remains for legacy reasons)

        :return: None
        '''        
        super().__init__(timeConstant, droneParameters, w0 = w0)
        self.name = 'rotorDynamics_V1'
        self.R = upper
        self.L = lower
        
    def getRisingRate(self, rotorNumber, dt):
        '''Limits increase of rotor speeds over a time step
        
        :param rotorNumber: Rotor index - integer
        :param dt: Time step, in seconds - float

        :returns: None, modifies rotor speed in place.
        '''
        self.OutputCommand_Rotor = dt * self.R + self.previousCommand[rotorNumber]

    def getFallingRate(self, rotorNumber, dt):
        '''Limits decrease of rotor speeds over a time step
        
        :param rotorNumber: Rotor index - integer
        :param dt: Time step, in seconds - float

        :returns: None, modifies rotor speed in place.
        '''        
        self.OutputCommand_Rotor = dt * self.L + self.previousCommand[rotorNumber]

    def getTargetRate(self, targetCommand, rotorNumber, dt):
        '''Calculates required rate change needed to reach commanded rotor speed from current rotor speed, over one time step
        
        :param targetCommand: Target rotor speed value - float
        :param rotorNumber: Rotor index - integer
        :param dt: Time step, in seconds - float

        :returns: None, sets target rotor speed attribute. 
        '''
        self.targetRate_Rotor = (targetCommand - self.previousCommand[rotorNumber])/dt

    def actuate(self, simVars):
        '''Apply actuator dynamics to commanded input in simulation

        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 

        :returns: True input vector, array-like. 
        '''
        # Get current step values
        step = simVars['currentTimeStep_index']
        # NOTE: step + 1 for targetCommand is it is the latest info from controller (i.e. next step)
        targetCommand = simVars['inputs_CMD'][step+1]
        dt = simVars['dt']
        # Prepare output command vector
        self.OutputCommand = np_zeros(targetCommand.shape)
        # Assign rotor speed values to each rotor, based on saturation, target rate, and achievable rate
        for i, rotorCommand in enumerate(targetCommand.reshape(-1)):
            self.getTargetRate(rotorCommand, i, dt)
            if self.targetRate_Rotor > self.R:
                self.getRisingRate(i, dt)
            elif self.targetRate_Rotor < self.L:
                self.getFallingRate(i, dt)
            else:
                self.OutputCommand_Rotor = rotorCommand
            self.OutputCommand[0, i] = self.saturate(i)
        self.previousCommand = self.OutputCommand.reshape(-1)
        return self.OutputCommand


class rotorDynamicsV2(rotorDynamicsBase):
    def __init__(self, timeConstant, droneParameters, w0 = np_zeros((4, 1)), **kwargs):
        '''Initializes the rotoDynamicsV2 object
        
        :param timeConstant: Actuator dynamics time constant, in seconds - Float. Passed on to rotorDynamicsBase
        :param droneParameters: Parameters of the quadrotor - Dictionary. Passed on to rotorDynamicsBase
        :param w0: Initial rotor speeds of each rotor - np.array-like of floats. Passed on to rotorDynamicsBase. Default is array of zeros.

        :return: None
        '''                
        super().__init__(timeConstant, droneParameters, w0 = w0)
        self.name = 'rotorDynamics_V2'

    def _help(self):
        '''
        Actuator dynamics based on the following actuator transfer function:
        H(s) = 1 / (tau*s + 1)
        Thus,
            -> Y(s)/U(s) = 1/(tau*s + 1)
            -> Y(s) * (tau*s + 1) = U(s)
        In time domain, 
            -> y_dot * tau + y = u
            -> y_dot = (1/tau) * (-y + u)
        In discrete form:
            y_dot[i] = (y[i+1] - y[i])/dt, where dt is the time step
        Thus, 
            y[i+1] - y[i] = dt * (1/tau) * (-y[i] + u[i])
            -> y[i+1] = y[i] + dt*(1/tau)*(-y[i] + u[i])
            -> y = True actuator response, eRPM
            -> u = Commanded actuator response, eRPM
        '''
        msg = 'Actuator dynamics based on the following actuator transfer function: \
                H(s) = 1 / (tau*s + 1)\
                Thus,\
                    -> Y(s)/U(s) = 1/(tau*s + 1)\
                    -> Y(s) * (tau*s + 1) = U(s)\
                In time domain, \
                    -> y_dot * tau + y = u\
                    -> y_dot = (1/tau) * (-y + u)\
                In discrete form:\
                    y_dot[i] = (y[i+1] - y[i])/dt, where dt is the time step\
                Thus, \
                    y[i+1] - y[i] = dt * (1/tau) * (-y[i] + u[i])\
                    -> y[i+1] = y[i] + dt*(1/tau)*(-y[i] + u[i])\
                    -> y = True actuator response, eRPM\
                    -> u = Commanded actuator response, eRPM\
                '
        print(msg)

    def actuate(self, simVars):
        '''Apply actuator dynamics to commanded input in simulation

        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 

        :returns: True input vector, array-like. 
        '''
        # Get current time step information
        step = simVars['currentTimeStep_index']
        # NOTE: step + 1 for targetCommand is it is the latest info from controller (i.e. next step)
        targetCommand = simVars['inputs_CMD'][step+1]
        dt = simVars['dt']
        # Prepare output command vector
        self.OutputCommand = np_zeros(targetCommand.shape)
        # Apply actuator dynamics and saturation
        for i, rotorCommand in enumerate(targetCommand.reshape(-1)):
            self.OutputCommand[0, i] = self._saturate(i, self.previousCommand[i] + dt*((1/self.tau[i])*(self._saturate(i, rotorCommand) - self.previousCommand[i])))
        self.previousCommand = self.OutputCommand.reshape(-1)
        return self.OutputCommand
    


class rotorDynamicsSaturation(rotorDynamicsBase):
    def __init__(self, timeConstant, droneParameters, w0 = np_zeros((4, 1)), **kwargs):
        '''Initializes the rotoDynamicsV2 object
        
        :param timeConstant: Actuator dynamics time constant, in seconds - Float. Passed on to rotorDynamicsBase
        :param droneParameters: Parameters of the quadrotor - Dictionary. Passed on to rotorDynamicsBase
        :param w0: Initial rotor speeds of each rotor - np.array-like of floats. Passed on to rotorDynamicsBase. Default is array of zeros.

        :return: None
        '''                
        super().__init__(timeConstant, droneParameters, w0 = w0)
        self.name = 'rotorDynamicsSaturation'

    def actuate(self, simVars):
        '''Apply actuator dynamics to commanded input in simulation

        :param simVars: Datastream of sim.py object - Dictionary of simulation variables, properties, and blocks. 

        :returns: True input vector, array-like. 
        '''
        # Get current time step information
        step = simVars['currentTimeStep_index']
        # NOTE: step + 1 for targetCommand is it is the latest info from controller (i.e. next step)
        targetCommand = simVars['inputs_CMD'][step+1]
        dt = simVars['dt']
        # Prepare output command vector
        self.OutputCommand = np_zeros(targetCommand.shape)
        # Apply actuator dynamics and saturation
        for i, rotorCommand in enumerate(targetCommand.reshape(-1)):
            self.OutputCommand[0, i] = self._saturate(i, rotorCommand)
        self.previousCommand = self.OutputCommand.reshape(-1)
        return self.OutputCommand


ACTUATOR_NAMES = {
    'base':rotorDynamicsBase,
    'rotorDynamicsV1':rotorDynamicsV1,
    'rotorSaturationOnly':rotorDynamicsSaturation,
    'rotorDynamicsV2':rotorDynamicsV2
}