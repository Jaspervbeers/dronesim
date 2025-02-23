from actuators import exampleSim_actuator
from controllers import exampleSim_controller
from funcs import exampleSim_EOM
from models import exampleSim_model
from noiseDisturbance import sensorNoise
import sim
import numpy as np # This can be removed if it is not used. Only used for giving initial conditions example.

# Load and initialize model. Add (keyword) arguments as necessary
k = 0.8
m = 1
c = 1.5
model = exampleSim_model.exampleSim_model(k=k, c=c, m=m)

# Configure if noise should be added or not
addNoise = True
# Noise seed
noiseSeed = 123456
# Define noise levels for each state
noiseMapping = {
    'x':0.005,        
    'x_dot':0.001
}
noiseBlock = sensorNoise.noiseBlock(stateMapping = {'x':0, 'x_dot':1},globalSeed=noiseSeed)
if not addNoise:
    # Set noise scale to zero for no noise
    noiseBlock.mapStateNoiseScales(np.arange(0, 2, 1), np.zeros(2))
else:
    for k, v in noiseMapping.items():
        noiseBlock._setStateNoiseScaling(k, v)

# Load and initialize controller. Add (keyword) arguments as necessary
P = np.array(3).reshape(1, 1)
I = np.array(1).reshape(1, 1)
D = np.array(2).reshape(1, 1)
controller = exampleSim_controller.exampleSim_controller(P, I, D, noiseBlock = noiseBlock)

# Load and initialize actuators. Add (keyword) arguments as necessary
actuator = exampleSim_actuator.exampleSim_actuator()

# Define equations of motion
EOM = exampleSim_EOM.EOM

# Construct simulator
simulator = sim.sim(model, EOM, controller, actuator)

# Initial condition and simulation parameters
dt = 0.01 # Time step of 0.01 seconds. Can also be variable but need to compute current time step within sim loop (more computations per loop)
time = np.arange(0, 10, dt) # Create time array with time step dt
state = np.zeros((len(time), 2)).reshape(-1, 1, 2) 
action = np.zeros(time.shape).reshape(-1, 1, 1) # Define action vector, assume 1 input. Change as necessary
reference = np.zeros(time.shape).reshape(-1, 1, 1) # Define reference vector
# reference[int(5/dt):] = -0.3

# Initial conditions
state[0, 0, 0] = 0.5
state[0, 0, 1] = 0.01

# # Book-keeping
# stateDerivative = state.copy() # Create state derivative vector to store x_dot
# forces = np.zeros(time.shape) # Create forces vector to hold computed forces
# moments = np.zeros(time.shape) # Create moments vector to hold computed moments
# cmdAction = action.copy() # Create commanded action for controller output (i.e. before actuator dynamics)

# # Simulation loop
# for i in tqdm(range(len(time)-1)):
# 	state[i+1], stateDerivative[i+1], forces[i+1], moments[i+1], cmdAction[i+1], action[i+1] = simulator.run(state[i], reference[i], dt, modelParams = model.modelParams) # Add any additional keyword arguments as necessary

# Initialize simulator
simulator = sim.sim(model, EOM, controller, actuator, angleIndices=[])
simulator.setInitial(time, state, action, reference)
print('[ INFO ] SIMULATION PARAMETERS')
print('{:^8}'.format('') + f'Controller: \t\t\t{controller.name}')
# print('{:^8}'.format('') + f'Add noise: \t\t\t{addNoise}')
print('{:^8}'.format('') + f'Drone model: \t\t\t{model.name}')
print('{:^8}'.format('') + f'Actuators: \t\t\t{actuator.name}')

# Run simulation
simulator.run()

# Plot results
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time, state[:, 0, 1], color = 'coral', alpha = 0.8, label = 'velocity')
ax.plot(time, state[:, 0, 0], color = 'firebrick', label='position')
ax.plot(time, reference[:, 0, :], color = 'k', linestyle = '--', label = 'reference')
# ax.plot(time, forces, color = 'gold', label = 'input force')
ax.set_xlabel(r'$\mathbf{Time} \quad [s]$', fontsize = 16)
ax.set_ylabel(r'$\mathbf{Magnitude} \quad [m]/[ms^{-1}]$', fontsize = 16)
ax.legend(fontsize=14)
ax.grid(which='major', alpha = 0.3, axis='both')
ax.grid(which='minor', alpha = 0.2, axis='both', linestyle = '--')
plt.show()