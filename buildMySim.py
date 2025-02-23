'''
Script to build templates for the components necessary to use the simulation model. Users can then modify the components appropriately. 
'''
import os

def filenameExists(filepath, name):
    names = os.listdir(filepath)
    if name in names:
        return True
    else:
        return False


def askQuery(query, filepath):
    name = str(input(query)).split('.')[0] + '.py'
    if filenameExists(filepath, name):
        name = askQuery('Sorry, that name already exists. Please specify another: ', filepath)
    return name


simName = str(input('Please indicate a name for the SIMULATION (e.g. mySim): '))
useDefaults = str(input('Would you like to use default names for the simulation components? (y/n) '))

if useDefaults.lower() == 'y':
    controllerName = simName + '_controller.py'
    EOMName = simName + '_EOM.py'
    modelName = simName + '_model.py'
    actuatorName = simName + '_actuator.py'
else:
    controllerName = askQuery('Please indicate a name for the CONTROLLER file: ', os.path.join(os.getcwd(), 'controllers'))
    EOMame = askQuery('Please indicate a name for the EQUATIONS OF MOTION file: ', os.path.join(os.getcwd(), 'funcs'))
    modelName = askQuery('Please indicate a name for the MODEL file: ', os.path.join(os.getcwd(), 'models'))
    actuatorName = askQuery('Please indicate a name for the ACTUATOR file: ', os.path.join(os.getcwd(), 'actuators'))


currentDir = os.getcwd()

# Create actuator module
actuatorDir = os.path.join(currentDir, 'actuators')
with open(os.path.join(actuatorDir, actuatorName), 'w') as f:
    f.write('# This is your actuator class. Feel free to change the name to something more suitable.')
    f.write('\n# Below are the mandatory class methods.')
    f.write('\n# In the __init__ function, define/assign actuator parameters (e.g. rate constants)')
    f.write('\n# The actuate method is used by the simulation to run the actuator loop')
    f.write('\nclass {}:'.format(actuatorName[:-3]))
    f.write('\n\tdef __init__(self):')
    f.write(f'\n\t\tself.name = "{actuatorName[:-3]}"')
    f.write('\n\t\treturn None')
    f.write('\n\n\t# This function must take "simVars" as the only input. From simVars, the necessary variables (e.g. input command) can be extracted')
    f.write('\n\t# This function must return the true input (i.e. the commanded input subject to actuator dynamics)')
    f.write('\n\tdef actuate(self, simVars):')
    f.write('\n\t\t# Extract the current time step index and commanded input at the NEXT step (note: next step is used here since the commanded input was just updated in the simulation loop)')
    f.write('\n\t\tstep = simVars["currentTimeStep_index"]')
    f.write('\n\t\tcommandedInput = simVars["inputs_CMD"][step + 1]')
    f.write('\n\t\t# Place actuator dynamics here. Create and call relevant functions if necessary')
    f.write('\n\t\ttrueInput = commandedInput # This line is equivalent to no actuator dynamics')
    f.write('\n\t\treturn trueInput')
print(f'[ INFO ] Created actuator in: {actuatorDir}/{actuatorName}\n')

# Create controller module
controllerDir = os.path.join(currentDir, 'controllers')
with open(os.path.join(controllerDir, controllerName), 'w') as f:
    f.write('# This is your controller class. Feel free to change the name to something more suitable.')
    f.write('\n# Below are the mandatory class methods.')
    f.write('\n# In the __init__ function, define/assign controller parameters (e.g. gains)')
    f.write('\n# The control method is used by the simulation to run the total controller loop. As such, all inner-loop controller should be placed in this method.')
    f.write('\nclass {}:'.format(controllerName[:-3]))
    f.write('\n\tdef __init__(self, noiseBlock = None):')
    f.write(f'\n\t\tself.name = "{controllerName[:-3]}"')
    f.write('\n\t\tif noiseBlock is None:')
    f.write('\n\t\t\traise ValueError("Need to specify noiseBlock")')
    f.write('\n\t\tself.noiseBlock = noiseBlock')
    f.write('\n\t\treturn None')
    f.write('\n\n\t# This function must take "simVars" as the only input. From simVars, the necessary variables (e.g. states) can be extracted')
    f.write('\n\t# This function must return the control input (i.e. the command to system actuators)')
    f.write('\n\tdef control(self, simVars):')
    f.write('\n\t\t# Extract current step and state at this step')
    f.write('\n\t\tstep = simVars["currentTimeStep_index"]')
    f.write('\n\t\t# Use noiseBlock to simulate sensor noise (only perceived by controller, hence state_noisy)')
    f.write('\n\t\tself.noiseBlock.addStateNoise(simVars) # Updates simVars["state_noisy"] with state[step] + noise')
    f.write('\n\t\tstate = simVars["state_noisy"][step]')
    f.write('\n\t\t# Do control here. Create and call relevant functions as necessary')
    f.write('\n\t\tcontrolInput = None')
    f.write('\n\t\treturn controlInput')
print(f'[ INFO ] Created controller in: {controllerDir}/{controllerName}\n')

# Create EOM module
EOMDir = os.path.join(currentDir, 'funcs')
with open(os.path.join(EOMDir, EOMName), 'w') as f:
    f.write('# This is your definition for the equations of motion. Feel free to change the name to something more suitable.')
    f.write('\n# This function must take "simVars" as the only input. From simVars, the necessary variables (e.g. forces and moments) can be extracted.')
    f.write('\n# This function must return the state derivative.')
    f.write('\ndef EOM(simVars):')
    f.write('\n\t# Extract current step and state at this step')
    f.write('\n\tstep = simVars["currentTimeStep_index"]')
    f.write('\n\tstate = simVars["state"][step]')
    f.write('\n\t# Since the forces and moments have just been updated in the simulation loop, we need to extract the forces and moments of the next step')
    f.write('\n\tforces = simVars["forces"][step + 1]')
    f.write('\n\tmoments = simVars["moments"][step + 1]')
    f.write('\n\tx_dot = state # Calculate state derivative here')
    f.write('\n\treturn x_dot')
print(f'[ INFO ] Created equations of motion (EOM) in: {EOMDir}/{EOMName}\n')

# Create model module
modelDir = os.path.join(currentDir, 'models')
with open(os.path.join(modelDir, modelName), 'w') as f:
    f.write('# This is your model class. Feel free to change the name to something more suitable.')
    f.write('\n# Below are the mandatory class methods.')
    f.write('\n# In the __init__ function, define/assign/load/import model parameters')
    f.write('\n# The getForcesMoments method is used by the simulation to run apply the model')
    f.write('\nclass {}:'.format(modelName[:-3]))
    f.write('\n\tdef __init__(self):')
    f.write(f'\n\t\tself.name = "{modelName[:-3]}"')    
    f.write('\n\t\treturn None')
    f.write('\n\n\t# This function must take "simVars" as the only input. From simVars, necessary variables (e.g. trueInput) can be extracted')
    f.write('\n\t# This function must return the forces and moments')
    f.write('\n\tdef getForcesMoments(self, simVars):')
    f.write('\n\t\t# Extract the current step and state')
    f.write('\n\t\tstep = simVars["currentTimeStep_index"]')
    f.write('\n\t\tstate = simVars["state"][step]')
    f.write('\n\t\t# Since the control action has just been updated, take the next step')
    f.write('\n\t\ttrueInput = simVars["inputs"][step + 1]')
    f.write('\n\t\t# Calculate forces and moments here. Create and call relevant functions if necessary')
    f.write('\n\t\tF, M = 0, 0 # Just an example')
    f.write('\n\t\treturn F, M')
print(f'[ INFO ] Created model in: {modelDir}/{modelName}\n')

# Create simulation module
print(f'[ INFO ] Linking files in simulator ({simName})')
with open(os.path.join(currentDir, simName + '.py'), 'w') as  f:
    f.write('from actuators import {}'.format(actuatorName[:-3]))
    f.write('\nfrom controllers import {}'.format(controllerName[:-3]))
    f.write('\nfrom funcs import {}'.format(EOMName[:-3]))
    f.write('\nfrom models import {}'.format(modelName[:-3]))
    f.write('\nfrom noiseDisturbance import sensorNoise')
    f.write('\nimport sim')
    f.write('\nimport numpy as np # This can be removed if it is not used. Only used for giving initial conditions example.')
    f.write('\nimport matplotlib.pyplot as plt # This can be removed if it is not used. Only used for plotting sim results.')
    f.write('\n\n# Load and initialize model. Add (keyword) arguments as necessary')
    f.write('\nmodel = {}.{}()'.format(modelName[:-3], modelName[:-3]))
    f.write('\n\n# Initial condition and simulation parameters')
    f.write('\ndt = 0.1 # Time step of 0.1 seconds. Can also be variable but need to compute current time step within sim loop (more computations per loop)')
    f.write('\ntime = np.arange(0, 10, dt) # Create time array with time step dt')
    f.write('\nstate = np.zeros(time.shape).reshape(-1, 1, 1) # Define state vector, assume 1 state. Change as necessary, but middle index should be 1')
    f.write('\naction = np.zeros(time.shape).reshape(-1, 1, 1) # Define action vector, assume 1 input. Change as necessary, but middle index should be 1')
    f.write('\nreference = np.zeros(time.shape).reshape(-1, 1, 1) # Define reference vector')    
    f.write('\n\n# Initialize noise block')
    f.write('\naddNoise = False # Boolean to toggle noise in system')
    f.write('\nstateMapper = {"x1":0} # Map states to their indices in state vector')
    f.write('\n# Noise seed')
    f.write('\nnoiseSeed = 123456')
    f.write('\n# Define noise levels for each state')
    f.write('\nnoiseMapping = {"x1":0.1}')
    f.write('\nnoiseBlock = sensorNoise.noiseBlock(stateMapper)')
    f.write('\nif not addNoise:')
    f.write('\n\t# To turn noise off, set mapping of added noise to 0 for each state')
    f.write('\n\tnoiseBlock.mapStateNoiseScales(np.array(list(stateMapper.values())), np.array(list(stateMapper.values()))*0)')
    f.write('\nelse:')
    f.write('\n\tfor k, v in noiseMapping.items():')
    f.write('\n\t\tnoiseBlock._setStateNoiseScaling(k, v)')
    f.write('\n\n# Load and initialize controller. Add (keyword) arguments as necessary')
    f.write('\ncontroller = {}.{}(noiseBlock = noiseBlock)'.format(controllerName[:-3], controllerName[:-3]))
    f.write('\n\n# Load and initialize actuators. Add (keyword) arguments as necessary')
    f.write('\nactuator = {}.{}()'.format(actuatorName[:-3], actuatorName[:-3]))
    f.write('\n\n# Define equations of motion')
    f.write('\nEOM = {}.EOM'.format(EOMName[:-3]))
    f.write('\n\n\n# Construct simulator')
    f.write('\n# NOTE: If you would like to use quaternions, then specify the [roll, pitch, yaw] euler angle indices in the state vector under the keyword argument "angleIndices=[roll_index, pitch_index, yaw_index]"')
    f.write('\nsimulator = sim.sim(model, EOM, controller, actuator, angleIndices = [])')
    f.write('\nsimulator.setInitial(time, state, action, reference) # Set initial conditions')
    f.write('\n\n# Simulation loop')
    f.write('\nprint("Simulation configuration")')
    f.write('\nprint(f"{}")'.format(r'Model: \t\t\t{model.name}'))
    f.write('\nprint(f"{}")'.format(r'Controller: \t\t{controller.name}'))
    f.write('\nprint(f"{}")'.format(r'Add noise: \t\t{addNoise}'))
    f.write('\nprint(f"{}")'.format(r'Actuator: \t\t{actuator.name}'))
    f.write('\nsimulator.run()')
    f.write('\n\n# Extract results')
    f.write('\nstate = simulator.state')
    f.write('\nstateDerivative = simulator.stateDerivative')
    f.write('\ntrueActions = simulator.inputs')
    f.write('\ncmdActions = simulator.inputs_CMD')
    f.write('\nforces = simulator.forces')
    f.write('\nmoments = simulator.moments')
    f.write('\n\n# Plot some results')
    f.write('\nfig = plt.figure()')
    f.write('\nax = fig.add_subplot()')
    f.write('\nax.plot(simulator.time, state[:, 0, :])')
    f.write('\nax.set_xlabel("Time, s")')
    f.write('\nax.set_ylabel("State(s)")')
    f.write('\nax.set_title("Nothing happens because there are no dynamics yet!")')
    f.write('\nplt.show()')
print(f'[ INFO ] Created simulator in: {currentDir}/{simName}\n')