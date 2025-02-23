'''
Drone actor class for animations; defines quadrotor body and aesthetics, and updates position + orientation 

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 22-05-2023
'''

from funcs.angleFuncs import QuatRot, Eul2Quat
import numpy as np

from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d

class body:

    def __init__(self, model, origin = [0, 0, 0], rpy = [0, 0, 0]):
        # Get drone specific parameters
        self.getParamsFrom(model)
        
        # Initialize position and orientation
        self.x, self.y, self.z = origin[0], origin[1], origin[2] # m
        self.origin = np.array([self.x, self.y, self.z]).reshape(-1, 3)
        self._init_rpy = rpy
        self.quat = Eul2Quat(np.array(rpy).reshape(-1, 3))
        self.x_vec = np.array([1, 0, 0]).reshape(-1, 3)*self.b
        self.y_vec = np.array([0, 1, 0]).reshape(-1, 3)*self.b
        self.z_vec = np.array([0, 0, 1]).reshape(-1, 3)*self.b

        # Initialize rotorArms
        self.rotorArms = []
        self._initRotorArms()

        # Define default drone body kwargs
        self.bodyPlotKwargs = {
            'linewidth':2,
            'color':'k'
        }
        self.rotorPlotKwargs = {
            'color':'k'
        }
        self.rotor_default_alpha = 0.5
        self.positionHistoryKwargs = {
            # 'color':'#e67d0a'
            'color':'#00A6D6'
        }
        self.historyColor = self.positionHistoryKwargs['color']
        self.historyLinewidth = 1.2
        return None


    def _initRotorArms(self):
        self.l = np.cos(np.pi/4)*self.b
        perms = np.array([
            [self.l, self.l, 0],
            [self.l, -self.l, 0],
            [-self.l, self.l, 0],
            [-self.l, -self.l, 0]
            ])
        self.mapping = [self.rotorNumberMap['front right'], self.rotorNumberMap['front left'], self.rotorNumberMap['aft right'], self.rotorNumberMap['aft left']]
        for p, m in zip(perms, self.mapping):
            ra = arm(m, self.origin, p, rpy = self._init_rpy)
            ra.addRotor(self.origin, -1*self.b/3)
            self.rotorArms.append(ra)
        return None


    def _initDraw(self, ax, omega = None):
        self.rotorArms_artists = {}
        for ra in self.rotorArms:
            artists = {}
            # Rotor Frame
            lineOriginArm, = ax.plot(
                            [self.origin[:, 0][0], ra.origin[:, 0][0]], 
                            [self.origin[:, 1][0], ra.origin[:, 1][0]], 
                            [self.origin[:, 2][0], ra.origin[:, 2][0]],
                            **self.bodyPlotKwargs)
            artists.update({'lineOriginArm':lineOriginArm})
            lineArmRotor, = ax.plot(
                            [ra.origin[:, 0][0], ra.rotor.origin[:, 0][0]], 
                            [ra.origin[:, 1][0], ra.rotor.origin[:, 1][0]], 
                            [ra.origin[:, 2][0], ra.rotor.origin[:, 2][0]],
                            **self.bodyPlotKwargs)
            artists.update({'lineArmRotor':lineArmRotor})

            # Rotor plane
            Patch = self.makeCircle((0, 0), self.R, alpha = self.mapRPMtoAlpha(omega[:, ra.id - 1]), **self.rotorPlotKwargs)
            ax.add_patch(Patch)
            path = Patch.get_path() #Get the path and the associated transform
            trans = Patch.get_patch_transform()
            path = trans.transform_path(path) #Apply the transform
            Patch.__class__ = art3d.PathPatch3D #Change the class
            Patch._code3d = path.codes #Copy the codes
            Patch._facecolor3d = Patch.get_facecolor #Get the face color    
            verts = path.vertices #Get the vertices in 2D
            verts3D = np.zeros((len(verts), 3))
            verts3D[:, :2] = verts
            artists.update({'_rp_verts':verts3D.copy()})
            # Rotate
            vertsNew = QuatRot(np.vstack((self.quat,)*len(verts)), verts3D, rot = 'B2E')
            Patch._segment3d = vertsNew.reshape(-1, 3) + ra.rotor.origin.reshape(-1)
            artists.update({'rotorPlane':Patch})

            self.rotorArms_artists.update({ra.id:artists})

        # Update axis
        x_vec = QuatRot(self.quat, self.x_vec, rot = 'B2E') + self.origin
        y_vec = QuatRot(self.quat, self.y_vec, rot = 'B2E') + self.origin
        z_vec = QuatRot(self.quat, self.z_vec, rot = 'B2E') + self.origin

        xLine, = ax.plot(
            [self.origin[:, 0][0], x_vec[:, 0][0]], 
            [self.origin[:, 1][0], x_vec[:, 1][0]], 
            [self.origin[:, 2][0], x_vec[:, 2][0]],
            color = 'r', alpha = 0.5, linewidth = 1)

        yLine, = ax.plot(
            [self.origin[:, 0][0], y_vec[:, 0][0]], 
            [self.origin[:, 1][0], y_vec[:, 1][0]], 
            [self.origin[:, 2][0], y_vec[:, 2][0]],
            color = 'b', alpha = 0.5, linewidth = 1)

        zLine, = ax.plot(
            [self.origin[:, 0][0], z_vec[:, 0][0]], 
            [self.origin[:, 1][0], z_vec[:, 1][0]], 
            [self.origin[:, 2][0], z_vec[:, 2][0]],
            color = 'g', alpha = 0.5, linewidth = 1)

        self.axisLines = {'x':xLine, 'y':yLine, 'z':zLine}


    def draw(self, ax, omega = None):
        '''
        Update drone pose by manipulating matplotlib artist objects directly.
        '''
        # Update drone frame and rotors
        for ra, artists in zip(self.rotorArms, self.rotorArms_artists.values()):
            artists['lineOriginArm'].set_xdata(np.array([self.origin[:, 0][0], ra.origin[:, 0][0]]))
            artists['lineOriginArm'].set_ydata(np.array([self.origin[:, 1][0], ra.origin[:, 1][0]]))
            artists['lineOriginArm'].set_3d_properties(np.array([self.origin[:, 2][0], ra.origin[:, 2][0]]))
            artists['lineOriginArm'].set_color(self.bodyPlotKwargs['color'])

            artists['lineArmRotor'].set_xdata(np.array([ra.origin[:, 0][0], ra.rotor.origin[:, 0][0]]))
            artists['lineArmRotor'].set_ydata(np.array([ra.origin[:, 1][0], ra.rotor.origin[:, 1][0]]))
            artists['lineArmRotor'].set_3d_properties(np.array([ra.origin[:, 2][0], ra.rotor.origin[:, 2][0]]))
            artists['lineArmRotor'].set_color(self.bodyPlotKwargs['color'])

            verts3D = artists['_rp_verts'].copy()
            vertsNew = QuatRot(np.vstack((self.quat,)*len(verts3D)), verts3D, rot = 'B2E')
            artists['rotorPlane']._segment3d = vertsNew.reshape(-1, 3) + ra.rotor.origin.reshape(-1)
            artists['rotorPlane'].set_alpha(self.mapRPMtoAlpha(omega[:, ra.id - 1]))
            artists['rotorPlane'].set_color(self.rotorPlotKwargs['color'])

        # Update body reference frame
        x_vec = QuatRot(self.quat, self.x_vec, rot = 'B2E') + self.origin
        y_vec = QuatRot(self.quat, self.y_vec, rot = 'B2E') + self.origin
        z_vec = QuatRot(self.quat, self.z_vec, rot = 'B2E') + self.origin

        self.axisLines['x'].set_xdata(np.array([self.origin[:, 0][0], x_vec[:, 0][0]]))
        self.axisLines['x'].set_ydata(np.array([self.origin[:, 1][0], x_vec[:, 1][0]]))
        self.axisLines['x'].set_3d_properties(np.array([self.origin[:, 2][0], x_vec[:, 2][0]]))

        self.axisLines['y'].set_xdata(np.array([self.origin[:, 0][0], y_vec[:, 0][0]]))
        self.axisLines['y'].set_ydata(np.array([self.origin[:, 1][0], y_vec[:, 1][0]]))
        self.axisLines['y'].set_3d_properties(np.array([self.origin[:, 2][0], y_vec[:, 2][0]]))

        self.axisLines['z'].set_xdata(np.array([self.origin[:, 0][0], z_vec[:, 0][0]]))
        self.axisLines['z'].set_ydata(np.array([self.origin[:, 1][0], z_vec[:, 1][0]]))
        self.axisLines['z'].set_3d_properties(np.array([self.origin[:, 2][0], z_vec[:, 2][0]]))


    def drawBrute(self, ax, omega = None):
        '''
        Update drone pose by clearing old artists and drawing new artists
        '''
        for ra in self.rotorArms:
            # Update drone frame
            ax.plot(
                [self.origin[:, 0][0], ra.origin[:, 0][0]], 
                [self.origin[:, 1][0], ra.origin[:, 1][0]], 
                [self.origin[:, 2][0], ra.origin[:, 2][0]],
                **self.bodyPlotKwargs)
            ax.plot(
                [ra.origin[:, 0][0], ra.rotor.origin[:, 0][0]], 
                [ra.origin[:, 1][0], ra.rotor.origin[:, 1][0]], 
                [ra.origin[:, 2][0], ra.rotor.origin[:, 2][0]],
                **self.bodyPlotKwargs)
            
            # Update rotors
            Patch = self.makeCircle((0, 0), self.R, alpha = self.mapRPMtoAlpha(omega[:, ra.id - 1]), **self.rotorPlotKwargs)
            ax.add_patch(Patch)
            path = Patch.get_path() #Get the path and the associated transform
            trans = Patch.get_patch_transform()
            path = trans.transform_path(path) #Apply the transform
            Patch.__class__ = art3d.PathPatch3D #Change the class
            Patch._code3d = path.codes #Copy the codes
            Patch._facecolor3d = Patch.get_facecolor #Get the face color    
            verts = path.vertices #Get the vertices in 2D
            verts3D = np.zeros((len(verts), 3))
            verts3D[:, :2] = verts
            # Rotate
            vertsNew = QuatRot(np.vstack((self.quat,)*len(verts)), verts3D, rot = 'B2E')
            Patch._segment3d = vertsNew.reshape(-1, 3) + ra.rotor.origin.reshape(-1)

        # Update axis
        x_vec = QuatRot(self.quat, self.x_vec, rot = 'B2E') + self.origin
        y_vec = QuatRot(self.quat, self.y_vec, rot = 'B2E') + self.origin
        z_vec = QuatRot(self.quat, self.z_vec, rot = 'B2E') + self.origin

        ax.plot(
            [self.origin[:, 0][0], x_vec[:, 0][0]], 
            [self.origin[:, 1][0], x_vec[:, 1][0]], 
            [self.origin[:, 2][0], x_vec[:, 2][0]],
            color = 'r', alpha = 0.5, linewidth = 2)

        ax.plot(
            [self.origin[:, 0][0], y_vec[:, 0][0]], 
            [self.origin[:, 1][0], y_vec[:, 1][0]], 
            [self.origin[:, 2][0], y_vec[:, 2][0]],
            color = 'b', alpha = 0.5, linewidth = 2)

        ax.plot(
            [self.origin[:, 0][0], z_vec[:, 0][0]], 
            [self.origin[:, 1][0], z_vec[:, 1][0]], 
            [self.origin[:, 2][0], z_vec[:, 2][0]],
            color = 'g', alpha = 0.5, linewidth = 2)


    def getParamsFrom(self, model):
        # Probe model for parameters
        self.droneParams = model.droneParams
        self.R = self.droneParams['R']
        self.b = self.droneParams['b']
        self.minRPM = self.droneParams['idle RPM']
        self.maxRPM = self.droneParams['max RPM']
        self.rotorNumberMap = self.droneParams['rotor configuration']


    def mapRPMtoAlpha(self, rpm):
        # Map rotor speed rpm to alpha values of the rotor discs in the visualization
        return float(0.1 + (0.9-0.1)/(self.maxRPM - self.minRPM)*(rpm - self.minRPM))


    def update(self, xyz, rotation):
        # Translate
        self.origin = xyz
        # Save rotation
        self.quat = rotation
        for ra in self.rotorArms:
            ra.update(xyz, rotation)


    def makeCircle(self, originxy, radius, **kwargs):
        return Circle(originxy, radius, **kwargs)

    def _setRotorColor(self, color):
        self.rotorPlotKwargs.update({'color':color})

    def _setFrameColor(self, color):
        self.bodyPlotKwargs.update({'color':color})

    def _setColor(self, color):
        self._setFrameColor(color)
        self._setRotorColor(color)

    def _setHistoryColor(self, color):
        self.positionHistoryKwargs.update({'color':color})
        self.historyColor = self.positionHistoryKwargs['color']

    def _setRotorAlpha(self, alpha):
        self.rotor_default_alpha = alpha

    def _setFrameThickness(self, thickness):
        self.bodyPlotKwargs.update({'linewidth':thickness})



class arm:

    def __init__(self, id, origin, endPoint, rpy = [0, 0, 0]):
        self.id = int(id)
        self._init_rpy = rpy
        self.quat = Eul2Quat(np.array(rpy).reshape(-1, 3))
        # self.origin = QuatRot(self.quat, (origin + endPoint).reshape(-1, 3), rot = 'B2E')
        self.origin = QuatRot(self.quat, (endPoint).reshape(-1, 3), rot = 'B2E') + origin
        self.localOrigin = endPoint.reshape(-1, 3)
        return None

    def addRotor(self, origin, verticalOffset):
        self.rotor = rotor(origin, self.localOrigin, verticalOffset, rpy = self._init_rpy)

    def update(self, xyz, rotation):
        # Update arm pose
        self.quat = rotation
        self.origin = QuatRot(self.quat, self.localOrigin, rot = 'B2E') + xyz
        self.rotor.update(xyz, rotation)


class rotor:

    def __init__(self, origin, endPoint, verticalOffset, rpy = [0, 0, 0]):
        self.quat = Eul2Quat(np.array(rpy).reshape(-1, 3))
        self.offset = verticalOffset
        _endPoint = endPoint.copy()
        _endPoint[:, 2] = endPoint[:, 2] + verticalOffset
        # self.origin = QuatRot(self.quat, (endPoint).reshape(-1, 3), rot = 'B2E')
        self.origin = QuatRot(self.quat, (_endPoint).reshape(-1, 3), rot = 'B2E') + origin
        self.localOrigin = _endPoint.reshape(-1, 3)
        return None

    def update(self, xyz, rotation):
        # Update rotor disc pose
        self.quat = rotation
        self.origin = QuatRot(rotation, self.localOrigin, rot = 'B2E') + xyz