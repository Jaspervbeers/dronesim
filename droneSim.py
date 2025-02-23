from models import droneModel
from controllers import controllerNames
from actuators import droneRotors
from funcs import droneEOM, integrators
from noiseDisturbance import droneSensorNoise
from animation import animate, drone
import numpy as np
import sim
import os
import json

# Check python version, since standalone models are mostly identified with python 3.8
import sys
if sys.version_info.major != 3 or sys.version_info.minor != 8:
    print(f'[ WARNING ] Current Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} may not be compatible with models (Python 3.8.12)')

'''
Load simulation configuration
'''
with open('droneSimConfig.json', 'r') as f:
    simConfig = json.load(f)

''' 
Model initialization

Load and initialize desired model using droneModel class. For default model, leave path = None. 
'''
# Load an identified model
if simConfig['model']['use identified model']:
    # Path to models, contained within droneSim directory
    modelPath = simConfig['model']['model path']
    # Choose model for simulation 
    modelID = simConfig['model']['model ID']

    # Due to offsets with the c.g. between and within flights, there are non-zero forces and moments which appear in the models. 
    # NOTE: That these non-zero forces and moments are also observable in the flight data itself. 
    # These non-zero forces and moments lead to unequal rotor speeds and non-zero attitude during hover.
    # The removal of this offsets can be toggled through NoOffset
    NoOffset = not simConfig['model']['correct for moment asymmetries']    # If False then force and moment offsets due to misaligned c.g. (with centroid) will be compensated for. 

    # Initialize the model
    model = droneModel.model(path = modelPath, modelID = modelID, NoOffset = NoOffset)

# Use default simple model
else:
    model = droneModel.simpleModel()

# Check if the models have been identified with gravity removed or not. If gravity has been removed, we should not remove it again in the EOM. 
if not model.hasGravity:
    model.droneParams['g'] = 0


'''
Noise block configuration

Here we configure our (sensor) noise statistics, which will be fed to the controller
'''
# Configure if noise should be added or not
addNoise = simConfig['noise']['add']
# Noise seed
noiseSeed = simConfig['noise']['seed']
# Define noise levels for each state
noiseMapping = {
    'roll':0.01,        # rad
    'pitch':0.01,       # rad
    'yaw':0.01,         # rad
    'u':0.01,           # m/s
    'v':0.01,           # m/s
    'w':0.01,           # m/s
    'p':np.sqrt(8.73e-05),       # rad/s (from MPU6000) 
    'q':np.sqrt(8.73e-05),       # rad/s (from MPU6000)
    'r':np.sqrt(8.73e-05),       # rad/s (from MPU6000)
    'x':0,
    'y':0,
    'z':0
}
noiseBlock = droneSensorNoise.droneNoiseBlock(globalSeed=noiseSeed)
if not addNoise:
    # Set noise scale to zero for no noise
    noiseBlock.mapStateNoiseScales(np.arange(0, 12, 1), np.zeros(12))
else:
    for k, v in noiseMapping.items():
        noiseBlock._setStateNoiseScaling(k, v)


'''
Controller initialization

Load controller (incl. subcontrollers, planners, etc.) by calling relevant class.

Here we use the provided PID controller for simplicity. Other controller architectures may also be designed.
'''
# First check if there are model-specific controller gains in the Models folder, otherwise use default in controllers folder. 
PIDGainPath = os.path.join(modelPath, modelID)
if not os.path.exists(os.path.join(PIDGainPath)):
    PIDGainPath = os.path.join(os.getcwd(), 'controllers')

signMask = model.getSignMask()
# In case there is an axis mismatch, specify it here.
# By default, we use a right-handed system where x points forwards (thumb finger), y to the left (index finger), and z down (middle finger)                    
signSwitch = np.array([1, 1, 1]) # Needed to match axis definition to drone configuration
RatePIDSignMask = signMask * signSwitch

# Initialize controller with control moment mapping and gains
controller = controllerNames.DRONE_CONTROLLER_NAMES[simConfig['controller']['controller']].controller(signMask, noiseBlock, PIDGainPath)



'''
Actuator dynamics 

Define rotor actuator dynamics and initialize the actuator (droneRotors) class
'''
# Parameterize Rotor dynamics
if 'taus' in model.droneParams.keys():
    timeConstant = [v for v in model.droneParams['taus'].values()]
    print('[ INFO ] Found model-specific actuator time constant. Using these over default.')
else:
    timeConstant = [1/35,]*4
rateUpper = 5e4
rateLower = -1*5e4
# Initialize actuator object
actuator = droneRotors.ACTUATOR_NAMES[simConfig['actuator']['actuator']](timeConstant = timeConstant, rateUpper = rateUpper, rateLower = rateLower, droneParameters=model.droneParams, w0 = np.zeros((4, 1)))

'''
Equations of Motion

Initalize the equations of motion for the drone
'''
EOM = droneEOM.EOM


'''
SIMULATION

Define simulation, reference signals, and run simulation
'''
# Define simulation parameters
# Simulation duration
tMax = simConfig['simulation']['duration']
# Time step
dt = simConfig['simulation']['time step']
# dt = 0.01 # More realtime for visualization
# Time array
time = np.arange(0, tMax, dt)

# Define state vector: [roll, pitch, yaw, u, v, w, p, q, r]
# - roll    = Roll angle of the quadrotor, in radians
# - pitch   = Pitch angle of the quadrotor, in radians
# - yaw     = Yaw angle of the quadrotor, in radians
# - u       = body velocity along x, in meters per second
# - v       = body velocity along y, in meters per second
# - w       = body velocity along z, in meters per second
# - p       = Roll rate about the x axis, in radians per second
# - q       = Pitch rate about the y axis, in radians per second
# - r       = yaw rate about the z axis, in radians per second
# - x       = position in x, in meters
# - y       = position in y, in meters
# - z       = position in z, in meters
state = np.zeros((len(time), 1, 12))
# Define inputs: [w1, w2, w3, w4]
# - wi      = Rotor speed of the ith rotor, in eRPM
rotorSpeeds = np.zeros((len(time), 1, 4)) 

# Define initial conditions
state[0, 0, [0, 1, 2]] = np.array(simConfig['initial state conditions']['attitude'])
state[0, 0, [9, 10, 11]] = np.array(simConfig['initial state conditions']['position'])

# Initialize reference signal.
# NOTE: To leave certain states uncontrolled, set associated reference to np.nan
reference = state.copy() 

doRollPitchYawTrack = True *0     # NOTE: Incompatible with upset controller
doMaxVel = True *0                # NOTE: Incompatible with upset controller
doVelocityTrack = True *0
doPositionTrack = simConfig['references']['do position track']
doCircleTrack = simConfig['references']['do circle track']

if doRollPitchYawTrack:
    # Check controller
    if controller.name == 'droneUpsetController_PID':
        print('[ WARNING ] Upset controller PID not compatible with this reference command! Will not track.')
    # Roll tracking 
    reference[int(2/dt):int(4/dt), 0, 0] = 1 # Radian
    # NOTE: Need to remove velocity control, along y, while giving pure attitude commands
    reference[int(2/dt):int(4/dt), 0, 4] = np.nan # m/s
    reference[int(2/dt):int(4/dt), 0, 10] = np.nan # m

    # Pitch tracking 
    reference[int(5/dt):int(7/dt), 0, 1] = 0.8 # Radian
    # NOTE: Need to remove velocity control, along x, while giving pure attitude commands
    reference[int(5/dt):int(7/dt), 0, 3] = np.nan # m/s
    reference[int(5/dt):int(7/dt), 0, 9] = np.nan # m

    # Yaw tracking 
    reference[int(8/dt):, 0, 2] = 0.5 # Radian
elif doMaxVel:
    # Check controller
    if controller.name == 'droneUpsetController_PID':
        print('[ WARNING ] Upset controller PID not compatible with this reference command! Will not track.')
    # Pitch tracking 
    reference[int(2/dt):, 0, 1] = -1.5 # Radian
    # Velocity along x
    reference[int(2/dt):, 0, 3] = np.nan # m/s
    # Position along x
    reference[int(2/dt):, 0, 9] = np.nan # m
elif doVelocityTrack:
    # Velocity along x
    reference[int(2/dt):int(4/dt), 0, 3] = 4
    # Velocity along y
    reference[int(3/dt):int(6/dt), 0, 4] = -3
    # Velocity along z (Recall, z is +ve downwards)
    reference[int(7/dt):, 0, 5] = -5
elif doPositionTrack:
    # Position in x, m
    reference[int(6/dt):, 0, 9] = 4
    # Position in y, m
    reference[int(6/dt):, 0, 10] = -2
    # Position in z, m (+ve downwards!)
    reference[int(2/dt):, 0, 11] = -1
elif doCircleTrack:
    # Position in x, m
    reference[int(3/dt):, 0, 9] = 4*np.sin(time[int(3/dt):])
    # Position in y, m
    reference[int(3/dt):, 0, 10] = 4*np.cos(time[int(3/dt):])
    # Position in z, m (+ve downwards!)
    reference[int(3/dt):, 0, 11] = -2
else:
    # Hovering flight, all references @ 0
    pass


# Define initial rotor speeds, set to idle
rotorSpeeds[0, :, :] = float(model.droneParams['idle RPM'])
# Set initial actuator rotor speed to initial rotor speed
actuator.setInitial(rotorSpeeds[0, :, :])

# Define animator for visualization
showAnimation = simConfig['animation']['show']
saveAnimation = simConfig['animation']['save']
# Define drone visualization object (actor)
droneViz = drone.body(model, origin=state[0, 0, 9:12], rpy=state[0, 0, :3])
# Initialize animation
animator = animate.animation()
# Add drone to animation with same name as loaded model
animator.addActor(droneViz, model.modelID)

# Pass on all simulation objects to runSim.sim() to define the simulator.
simulator = sim.sim(model, EOM, controller, actuator, animator=animator, showAnimation=showAnimation)

# Use custom integrator
simulator.assignIntegrator(integrators.droneIntegrator_Euler) # Faster
# simulator.assignIntegrator(integrators.droneIntegrator_rk4) # Two options; one uses model and another does not. See integrators.py

# Set initial state of simulator
simulator.setInitial(time, state, rotorSpeeds, reference)


'''
Rotor failures
'''
# Modify actuation to simulate reduction in motor efficiency
rotorIndices = [0, 1, 2, 3]
rotorLimits = [f*model.droneParams['max RPM'] for f in simConfig['faults']['rotor health after failure']]
timeFailure = simConfig['faults']['time of failure']

def capRotorsAt(trueInput, affectedRotors, limits):
    cappedInput = trueInput.copy()
    for i, rotorIdx in enumerate(affectedRotors):
        cappedInput[:, rotorIdx] = np.nanmin([trueInput[:, rotorIdx][0], limits[i]])
    return cappedInput

def modifiedActuator(simVars):
    trueInput = simulator.actuator.actuate(simVars)
    if simVars['currentTimeStep_index'] >= int(timeFailure/simVars['dt']):
        modifiedInput = capRotorsAt(trueInput, rotorIndices, rotorLimits)
    else:
        modifiedInput = trueInput.copy()
    return modifiedInput

if simConfig['faults']['inject failure']:
    simulator.doActuation = modifiedActuator


'''
Model linearization
'''
# # Add linear forces and moments to simVars
# simulator.addSimVar('forces_linear', simulator.forces.copy())
# simulator.addSimVar('moments_linear', simulator.moments.copy())
# # Linearize model around hover
# x0 = state[0, :, :9]*0
# curr_u = 1100
# simulator.addSimVar('curr_u', curr_u)
# u0 = rotorSpeeds[0, :, :]*0 + curr_u
# model.linearize(x0, u0)

# def modifiedGetForcesMoments(simVars):
#     step = simVars['currentTimeStep_index']
#     # if np.abs(simVars['curr_u'] - np.nanmean(simVars['inputs'][step+1, :, :])) > 100:
#     #     model.linearize(simVars['state'][step, :, :9], simVars['inputs'][step + 1, :, :])
#     #     simVars['curr_u'] = np.nanmean(simVars['inputs'][step+1, :, :])
#     simVars['forces_linear'][step + 1, :], simVars['moments_linear'][step + 1, :] = model.getForcesMoments_linear(simVars)
#     return model.getForcesMoments(simVars)

# simulator.doForcesMoments = modifiedGetForcesMoments



'''
MAIN SIMULATION LOOP
'''
# Print key variables:
print('[ INFO ] SIMULATION PARAMETERS')
print('{:^8}'.format('') + f'Controller: \t\t\t{controller.name}')
print('{:^8}'.format('') + f'Add noise: \t\t\t{addNoise}')
print('{:^8}'.format('') + f'Drone model: \t\t\t{model.modelID}')
print('{:^8}'.format('') + f'c.g. offset correction: \t{not NoOffset}')
print('{:^8}'.format('') + f'Actuators: \t\t\t{actuator.name}')
simulator.run()

# Extract results
state = simulator.state
stateDerivative = simulator.stateDerivative
forces = simulator.forces
moments = simulator.moments
cmdRotorSpeeds = simulator.inputs_CMD
rotorSpeeds = simulator.inputs


saveDir = 'simResults'
savePath = os.path.join(os.getcwd(), saveDir)
if not os.path.exists(saveDir):
    os.makedirs(savePath)

# Save animation
if saveAnimation:
    # Save animation
    if saveAnimation:
        print('[ INFO ] Saving animation...')
        simulator.simVars['aniSaver'](os.path.join(savePath, 'animated_realTime.mp4'), dpi = 350, fps = int(int(1/dt)/simulator.simVars['AnimationUpdateFactor']))
        # simulator.simVars['aniSaver'](os.path.join(savePath, 'animated_slowed.mp4'), dpi = 350, fps = 30)


from funcs import plotting
print('[ INFO ] Creating and saving plots...')

plotting.plotResults(simulator, savePath)
plotting.show()