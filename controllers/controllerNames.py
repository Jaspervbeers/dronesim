from controllers import droneINDI
from controllers import droneINDI_nDes
from controllers import droneNDI
from controllers import dronePIDController
#from controllers import droneVPV

DRONE_CONTROLLER_NAMES = {
    'drone_INDI':droneINDI,
    'drone_INDI_nDes':droneINDI_nDes,
    'drone_NDI':droneNDI,
    'drone_PID':dronePIDController,
    #'drone_VPV':droneVPV
}