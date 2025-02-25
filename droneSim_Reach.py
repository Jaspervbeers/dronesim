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



# Get reference points to reach along the surface of a sphere
points = []
r = 4 # Sphere radius   
d = 20 # Controls number of points to reach (i.e. resolution of sphere)
for a in np.arange(0, 360, d):
    # for b in np.arange(0, 360, d):
    for b in np.arange(0, 180 + d, d):
        _x = r * np.sin(b/180*np.pi) * np.cos(a/180*np.pi)
        _y = r * np.sin(b/180*np.pi) * np.sin(a/180*np.pi)
        _z = r * np.cos(b/180*np.pi)
        points.append([_x, _y, _z])

points = np.array(points)

'''
MAIN SIMULATION LOOP
'''
from tqdm import tqdm

reached_tau = []
reached_T = []
reached_max = []
reached_min = []
for p in tqdm(points):
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
        'roll':0.01,                    # rad
        'pitch':0.01,                   # rad
        'yaw':0.01,                     # rad
        'u':0.01,                       # m/s
        'v':0.01,                       # m/s
        'w':0.01,                       # m/s
        'p':np.sqrt(8.73e-05),          # rad/s (from MPU6000) 
        'q':np.sqrt(8.73e-05),          # rad/s (from MPU6000)
        'r':np.sqrt(8.73e-05),          # rad/s (from MPU6000)
        'x':0,                          # m
        'y':0,                          # m
        'z':0                           # m (note: -ve is upwards)    
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
    tMax = 1
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


    # Define initial rotor speeds, set to idle
    rotorSpeeds[0, :, :] = float(model.droneParams['idle RPM'])
    # Set initial actuator rotor speed to initial rotor speed
    actuator.setInitial(rotorSpeeds[0, :, :])

    # # Define animator for visualization
    # showAnimation = True #*0
    # saveAnimation = False
    # # Define drone visualization object (actor)
    # droneViz = drone.body(model, origin=state[0, 0, 9:12], rpy=state[0, 0, :3])
    # # Initialize animation
    # animator = animate.animation()
    # # Add drone to animation with same name as loaded model
    # animator.addActor(droneViz, model.modelID)

    reference = state.copy()*0
    reference[:, :, 9] = p[0]
    reference[:, :, 10] = p[1]
    reference[:, :, 11] = p[2] 
    # Pass on all simulation objects to runSim.sim() to define the simulator.
    simulator = sim.sim(model, EOM, controller, actuator, showAnimation=False)
    # simulator = sim.sim(model, EOM, controller, actuator, animator=animator, showAnimation=showAnimation)

    # Use custom integrator
    simulator.assignIntegrator(integrators.droneIntegrator_Euler) # Faster
    # simulator.assignIntegrator(integrators.droneIntegrator_rk4) # Two options; one uses model and another does not. See integrators.py

    # Set initial state of simulator
    simulator.setInitial(time, state, rotorSpeeds, reference)

    simulator.run()
    xyz = simulator.state[-1, 0, 9:]
    reached_T.append(xyz * 1)
    xyz = simulator.state[int(len(time)/2)-1, 0, 9:]
    reached_tau.append(xyz * 1)
    xyz = np.nanmin(simulator.state[:, 0, 9:], axis = 0)
    reached_min.append(xyz * 1)
    xyz = np.nanmax(simulator.state[:, 0, 9:], axis = 0)
    reached_max.append(xyz * 1)



reached_T = np.array(reached_T)
reached_tau = np.array(reached_tau)
reached_max = np.array(reached_max)
reached_min = np.array(reached_min)


'''
Make plots
'''
import matplotlib.pyplot as plt
from funcs import plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull

myOrange = '#e67d0a'
myBlue = '#008bb4'
myGreen = 'mediumseagreen'
myYellow = '#ffbe3c'
myRed = 'firebrick'
myGrey = 'gainsboro'
myVelvet = 'mediumvioletred'
myOrangeRed = '#E5340B'
myPurple = 'mediumorchid'


# Function to add projections on each plane
def add_projection2d(ax, triangles, zorder = 1, plane="xy", color="tab:blue", alpha=0.3):
    projected = []
    #
    for tri in triangles:
        if plane == "xy":
            projected.append([(tri[0][0], tri[0][1]),
                                (tri[1][0], tri[1][1]),
                                (tri[2][0], tri[2][1])])
        elif plane == "xz":
            projected.append([(tri[0][0], tri[0][2]),
                                (tri[1][0], tri[1][2]),
                                (tri[2][0], tri[2][2])])
        elif plane == "yz":
            projected.append([(tri[0][1], tri[0][2]),
                                (tri[1][1], tri[1][2]),
                                (tri[2][1], tri[2][2])])
    #
    poly = PolyCollection(projected, facecolor=color, alpha=alpha, edgecolor='none', zorder = zorder)
    ax.add_collection(poly)



def makeFRSPlot(FRS, fig = None, colorFRS = 'tab:blue', zorder = 1, r = 1, lbl1 = None, limd = 1, alpha = 0.5):
    hull_FRS = ConvexHull(FRS)
    parentFig = False
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 0], sharey = ax2)
    else:
        ax = fig.axes[0]
        ax1 = fig.axes[1]
        ax2 = fig.axes[2]
        ax3 = fig.axes[3]
        parentFig = True
    ax.set_xlim([-r+limd, r-limd])
    ax.set_ylim([-r+limd, r-limd])
    ax.set_zlim([-r+limd, r-limd])
    #
    ax.add_collection3d(Poly3DCollection(FRS[hull_FRS.simplices], facecolors=colorFRS, alpha=alpha, zorder = zorder))
    add_projection2d(ax1, FRS[hull_FRS.simplices], plane = 'xy', color = colorFRS, zorder = zorder, alpha = alpha)
    add_projection2d(ax2, FRS[hull_FRS.simplices], plane = 'xz', color = colorFRS, zorder = zorder, alpha = alpha)
    add_projection2d(ax3, FRS[hull_FRS.simplices], plane = 'yz', color = colorFRS, zorder = zorder, alpha = alpha)
    #
    if not parentFig:
        ax.set_box_aspect([1,1,1])
        ax.set_zlim([-r+limd, r-limd])
        for _ax in fig.axes:
            _ax.set_xlim([-r+limd, r-limd])
            _ax.set_ylim([-r+limd, r-limd])
        # ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax.invert_zaxis()
        #
        handles = []
        if lbl1 is not None:
            plotting.addLegendPatch(handles, color = colorFRS, label = lbl1)
        if len(handles):
            ax.legend(handles = handles, fontsize = 16)
        ax.set_xlabel(r'$\mathbf{x}$, m', fontsize = 16)
        ax.set_ylabel(r'$\mathbf{y}$, m', fontsize = 16)
        ax.set_zlabel(r'$\mathbf{z}$, m', fontsize = 16)
        #
        ax1.set_xlabel(r'$\mathbf{x}$, m', fontsize = 12)
        ax1.set_ylabel(r'$\mathbf{y}$, m', fontsize = 12)
        ax2.set_xlabel(r'$\mathbf{x}$, m', fontsize = 12)
        ax2.set_ylabel(r'$\mathbf{z}$, m', fontsize = 12)
        ax3.set_ylabel(r'$\mathbf{z}$, m', fontsize = 12)
        ax3.set_xlabel(r'$\mathbf{y}$, m', fontsize = 12)
        for _ax in [ax1, ax2, ax3]:
            plotting.prettifyAxis(_ax)
        ax1.annotate(
            r'$\mathbf{X-Y}$ plane', 
            xy=(ax1.get_xlim()[0], ax1.get_ylim()[1]), 
            xytext = (10, -12), 
            textcoords = 'offset points', 
            ha = 'left', va = 'top', 
            fontsize = 12,
            bbox = dict(boxstyle = 'round,pad=0.3', edgecolor = 'black', facecolor = 'whitesmoke'), zorder = 11
        )
        ax2.annotate(
            r'$\mathbf{X-Z}$ plane', 
            xy=(ax2.get_xlim()[0], ax2.get_ylim()[1]), 
            xytext = (10, -12), 
            textcoords = 'offset points', 
            ha = 'left', va = 'top', 
            fontsize = 12,
            bbox = dict(boxstyle = 'round,pad=0.3', edgecolor = 'black', facecolor = 'whitesmoke'), zorder = 11
        )
        ax3.annotate(
            r'$\mathbf{Y-Z}$ plane', 
            xy=(ax3.get_xlim()[0], ax3.get_ylim()[1]), 
            xytext = (10, -12), 
            textcoords = 'offset points', 
            ha = 'left', va = 'top', 
            fontsize = 12,
            bbox = dict(boxstyle = 'round,pad=0.3', edgecolor = 'black', facecolor = 'whitesmoke'), zorder = 11
        )
    return fig


fig = makeFRSPlot(reached_T, r = r)
fig.suptitle(f'Forward rechable set (@t = {time[-1]} seconds)')
fig.axes[0].azim = -125
fig.axes[0].elev = 21

plt.subplots_adjust(
    top=0.89,
    bottom=0.065,
    left=0.07,
    right=0.98,
    hspace=0.2,
    wspace=0.2
)

plt.show()