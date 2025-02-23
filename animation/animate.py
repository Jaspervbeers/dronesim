'''
Animation class for droneSim: Handles real-time animation, as well as post-simulation animations. 

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 22-05-2023
'''


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D as mlines
import numpy as np

class animation:

    def __init__(self, posHorizon = 150):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        self.actors = {}
        self.positions = {}
        self.rotations = {}
        self.posHorizon = posHorizon
        self.AnimationUpdateFactor = None
        self.viewIsInitialized = False
        self.property_ax3d_azim = -60
        # self.property_ax3d_azim = -10
        # self.property_ax3d_elev = 30
        self.property_ax3d_elev = 20
        # self.property_ax3d_elev = 10
        self.property_ax3d_roll = 0
        # self.property_ax3d_azim = 90
        # self.property_ax3d_elev = 90
        # self.property_ax3d_roll = 0
        # TODO: Add option to give time series of axis properties 
        #   Can do this externally via setting axis properties and setting viewIsInitialized = False
        return None


    def updateBrute(self, frames, objectsPose):
        '''
        Update actor poses by redrawing the relevant artists.
        This method is typically slower than update(), but does not require a separate initialization. 
        '''
        # NOTE simVars should come in as list of simVars
        if frames % list(objectsPose.values())[0]['AnimationUpdateFactor'] == 0:
            self.ax.clear()
            for actr in self.actors.values():
                sv = objectsPose[actr.name]
                step = sv['currentTimeStep_index']
                xyz = sv['state'][step][:, 9:12]
                q = sv['quat'][step]
                omega = sv['inputs'][step]
                actr.drawBrute(xyz, q, self.ax, omega = omega)
                h = self.positionHistory(frames)
                self.ax.plot(sv['state'][(frames - h):frames, 0, 9], sv['state'][(frames - h):frames, 0, 10], sv['state'][(frames - h):frames, 0, 11], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)

            self.ax.set_xlim(*np.sort([xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]))
            self.ax.set_ylim(*np.sort([xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]))
            self.ax.set_zlim(*np.sort([xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]))
            self.ax.set_xlabel(r'$\mathbf{x}$ [m]')
            self.ax.set_ylabel(r'$\mathbf{y}$ [m]')
            self.ax.set_zlabel(r'$\mathbf{z}$ [m]')
            self.ax.invert_zaxis()
            # self.ax.invert_yaxis()
            # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
            # Below is a work around
            ylim = self.ax.get_ylim()
            self.ax.set_yticks( self.ax.get_yticks() )
            self.ax.set_ylim(ylim[::-1])

            plt.pause(sv['dt']/sv['AnimationUpdateFactor'])
            plt.show(block=False)


    def _initActorsDraw(self, objectsPose):
        for actr in self.actors.values():
            sv = objectsPose[actr.name]
            step = sv['currentTimeStep_index']
            xyz = sv['state'][step][:, 9:12]
            q = sv['quat'][step]
            omega = sv['inputs'][step]
            actr._initDraw(xyz, q, self.ax, omega = omega)
            self.positionHistory_lines, = self.ax.plot(sv['state'][0, 0, 9], sv['state'][0, 0, 10], sv['state'][0, 0, 11], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)

        self.ax.set_xlim(*np.sort([xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]))
        self.ax.set_ylim(*np.sort([xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]))
        self.ax.set_zlim(*np.sort([xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]))
        self.ax.set_xlabel(r'$\mathbf{x}$ [m]')
        self.ax.set_ylabel(r'$\mathbf{y}$ [m]')
        self.ax.set_zlabel(r'$\mathbf{z}$ [m]')
        self.ax.invert_zaxis()
        # self.ax.invert_yaxis()
        # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
        # Below is a work around
        ylim = self.ax.get_ylim()
        self.ax.set_yticks( self.ax.get_yticks() )
        self.ax.set_ylim(ylim[::-1])

        # plt.pause(sv['dt']/sv['AnimationUpdateFactor'])
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.0001)
        plt.show(block=False)
        plt.close(1)


    def update(self, frames, objectsPose):
        # NOTE simVars should come in as list of simVars
        '''
        Update the actor poses by updating existing artist objects. 
        '''
        if frames % list(objectsPose.values())[0]['AnimationUpdateFactor'] == 0:
            for actr in self.actors.values():
                sv = objectsPose[actr.name]
                step = sv['currentTimeStep_index']
                xyz = sv['state'][step][:, 9:12]
                q = sv['quat'][step]
                omega = sv['inputs'][step]
                actr.draw(xyz, q, self.ax, omega = omega)
                h = self.positionHistory(frames)
                self.positionHistory_lines.set_xdata(np.array(sv['state'][(frames - h):frames, 0, 9]))
                self.positionHistory_lines.set_ydata(np.array(sv['state'][(frames - h):frames, 0, 10]))
                self.positionHistory_lines.set_3d_properties(np.array(sv['state'][(frames - h):frames, 0, 11]))

            self.ax.set_xlim(*np.sort([xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]))
            self.ax.set_ylim(*np.sort([xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]))
            self.ax.set_zlim(*np.sort([xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]))
            self.ax.invert_zaxis()
            # self.ax.invert_yaxis()
            # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
            # Below is a work around
            ylim = self.ax.get_ylim()
            self.ax.yaxis.set_major_locator(ticker.AutoLocator())
            self.ax.set_ylim(ylim[::-1])
            if not self.viewIsInitialized:
                self.ax.azim = self.property_ax3d_azim
                self.ax.elev = self.property_ax3d_elev
                self.ax.roll = self.property_ax3d_roll
                self.viewIsInitialized = True
            handles = []
            # handles.append(mlines([], [], linestyle = 'None', label = 't = {:.4f} [s]'.format(frames*sv['dt'])))
            handles.append(mlines([], [], linestyle = 'None', label = 't = {:.4f} [s]'.format(frames*sv['dt'])))
            self.ax.legend(handles = handles, loc = 'upper right')

            # plt.pause(0.0001)
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.0001)
            # plt.show(block=False)


    def addActor(self, actr, name):
        a = actor(actr, name)
        self.actors.update({name:a})
        self.positions.update({name:a.xyz})
        self.rotations.update({name:a.q})


    def positionHistory(self, i):
        return np.min([self.posHorizon, i])

    def _positionHistory(self, i):
        return self.positionHistory(i)

    def updatePosterior(self, frames, objectsPose):
        '''
        Update actor poses after the simulation (all data available), for saving an animation.
        '''
        # self.ax.cla()
        idx = int(frames * list(objectsPose.values())[0]['AnimationUpdateFactor'])
        self.ax.clear()
        for actr in self.actors.values():
            sv = objectsPose[actr.name]
            xyz = sv['state'][idx][:, 9:12]
            q = sv['quat'][idx]
            omega = sv['inputs'][idx]
            if 'ActorColorHistory' in sv.keys():
                actr.actor._setColor(sv['ActorColorHistory'][idx])
            actr.drawBrute(xyz, q, self.ax, omega = omega)
            h = self.positionHistory(idx)
            self.ax.plot(sv['state'][(idx - h):idx, 0, 9], sv['state'][(idx - h):idx, 0, 10], sv['state'][(idx - h):idx, 0, 11], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)

        self.ax.set_xlim(*np.sort([xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]))
        self.ax.set_ylim(*np.sort([xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]))
        self.ax.set_zlim(*np.sort([xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]))
        self.ax.set_xlabel(r'$\mathbf{x}$ [m]')
        self.ax.set_ylabel(r'$\mathbf{y}$ [m]')
        self.ax.set_zlabel(r'$\mathbf{z}$ [m]')
        self.ax.invert_zaxis()
        # self.ax.invert_yaxis()
        # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
        # Below is a work around
        ylim = self.ax.get_ylim()
        self.ax.set_yticks( self.ax.get_yticks() )
        self.ax.set_ylim(ylim[::-1])
        if not self.viewIsInitialized:
            self.ax.azim = self.property_ax3d_azim
            self.ax.elev = self.property_ax3d_elev
            self.ax.roll = self.property_ax3d_roll
            self.viewIsInitialized = True
        handles = []
        handles.append(mlines([], [], linestyle = 'None', label = 't = {:.4f} [s]'.format(sv["time"][idx])))
        self.ax.legend(handles = handles, loc = 'upper right')


    def posteriorAnimation(self, objectsPose):
        '''
        Make animation of the simulation
        '''
        plt.close('all')
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')
        N = len(list(objectsPose.values())[0]['time'])
        dt = list(objectsPose.values())[0]['dt']
        fctr = list(objectsPose.values())[0]['AnimationUpdateFactor']
        self.ani = FuncAnimation(self.fig, self.updatePosterior, fargs=(objectsPose,), 
                    frames = int(N/fctr), 
                    interval = dt*1e03*fctr, 
                    save_count=int(N/fctr), repeat = False)
        return self.ani


    def quickSaveAnimation(self, filename, fps = 250, dpi = 300):
        '''
        Save simulation animation 
        '''
        self.ani.save(filename, writer = 'ffmpeg', fps = fps, dpi = dpi)


    def snapShot(self, objectsPose, times, figsize = (10, 10), uniformAxes = False):
        plt.close('all')
        self.fig = plt.figure(figsize = figsize)
        self.ax = self.fig.add_subplot(111, projection = '3d')
        fctrs = np.linspace(0.1, 1, len(times))
        droneParams = objectsPose[list(objectsPose.keys())[0]]['model'].droneParams
        hardOmegas = np.linspace(droneParams['idle RPM'], droneParams['max RPM'], len(times))
        xlims = []
        ylims = []
        zlims = []
        # lims = []
        for t, f, o in zip(times, fctrs, hardOmegas):
            for actr in self.actors.values():
                sv = objectsPose[actr.name]
                time = sv['time']
                idx = np.where(time >= t)[0][0]
                xyz = sv['state'][idx][:, 9:12]
                q = sv['quat'][idx]
                omega = sv['inputs'][idx]
                # omega = np.ones((1, 4))*o
                # actr.actor.bodyPlotKwargs.update({'alpha':f})
                # # Due to factor, rotor speeds can go below min
                # for i, o in enumerate(omega[0, :].copy()):
                #     import code
                #     code.interact(local=locals())
                #     omega[:, i] = np.min((np.max((o, sv['model'].droneParams['idle RPM'])), sv['model'].droneParams['max RPM']))
                if 'ActorColorHistory' in sv.keys():
                    actr.actor._setColor(sv['ActorColorHistory'][idx])
                actr.drawBrute(xyz, q, self.ax, omega = omega)
                h = self.positionHistory(idx)
                self.ax.plot(sv['state'][(idx - h):idx, 0, 9], sv['state'][(idx - h):idx, 0, 10], sv['state'][(idx - h):idx, 0, 11], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)

            xlims.append(np.sort([xyz[:, 0] - actr.actor.R*8, xyz[:, 0] + actr.actor.R*8]))
            ylims.append(np.sort([xyz[:, 1] - actr.actor.R*8, xyz[:, 1] + actr.actor.R*8]))
            zlims.append(np.sort([xyz[:, 2] - actr.actor.R*8, xyz[:, 2] + actr.actor.R*8]))


        if uniformAxes:
            lims = xlims + ylims + zlims
            self.ax.set_xlim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])
            self.ax.set_ylim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])
            self.ax.set_zlim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])  
        else:
            self.ax.set_xlim([np.sort(np.array(xlims).reshape(-1))[0], np.sort(np.array(xlims).reshape(-1))[-1]])
            self.ax.set_ylim([np.sort(np.array(ylims).reshape(-1))[0], np.sort(np.array(ylims).reshape(-1))[-1]])
            self.ax.set_zlim([np.sort(np.array(zlims).reshape(-1))[0], np.sort(np.array(zlims).reshape(-1))[-1]])
           
        self.ax.set_xlabel(r'$\mathbf{x}$ [m]', fontsize = 16)
        self.ax.set_ylabel(r'$\mathbf{y}$ [m]', fontsize = 16)
        self.ax.set_zlabel(r'$\mathbf{z}$ [m]', fontsize = 16)
        self._handleAxisFlips()
        self.fig.subplots_adjust(left = 0, right = 0.94, bottom = 0, top = 1)
        # plt.tight_layout()
        return self.fig

    # def _handleAxisFlips(self):
    #     self.ax.invert_zaxis()
    #     # NOTE: Because of how the matplotlib 3d plotting works, ax.invert_yaxis() does not work as expected
    #     # Below is a work around
    #     ylim = self.ax.get_ylim()
    #     self.ax.yaxis.set_major_locator(ticker.AutoLocator())
    #     self.ax.set_ylim(ylim[::-1])

    def _handleAxisFlips(self):
        self._handleFlipZ()
        self._handleFlipY()

    def _handleFlipZ(self):
        self.ax.invert_zaxis()

    def _handleFlipY(self):
        ylim = self.ax.get_ylim()
        self.ax.yaxis.set_major_locator(ticker.AutoLocator())
        self.ax.set_ylim(ylim[::-1])

    def closeAll(self):
        for i in plt.get_fignums():
            plt.close(i)
            self.fig = None
            self.figLive = None

    def _update(self, frames, objectsPose, *args):
        '''
        Update actor poses after the simulation (all data available), for saving an animation.
        '''
        # self.ax.cla()
        idx = int(frames * self.AnimationUpdateFactor)
        self.ax.clear()
        for actr in self.actors.values():
            sv = objectsPose[actr.name]
            xyz = sv['position'][idx]
            q = sv['rotation_q'][idx]
            if 'inputs' in sv:
                omega = sv['inputs'][idx]
            else:
                omega = None
            actr.drawBrute(xyz, q, self.ax, omega = omega)   
            h = self._positionHistory(idx)
            self.ax.plot(sv['position'][(idx - h):idx, 0, 0], sv['position'][(idx - h):idx, 0, 1], sv['position'][(idx - h):idx, 0, 2], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)
            self.xLims += [xyz[:, 0] - actr.actor.R*10, xyz[:, 0] + actr.actor.R*10]
            self.yLims += [xyz[:, 1] - actr.actor.R*10, xyz[:, 1] + actr.actor.R*10]
            self.zLims += [xyz[:, 2] - actr.actor.R*10, xyz[:, 2] + actr.actor.R*10]

        self.ax.set_xlim(np.nanmin(self.xLims[int((idx - h)/self.AnimationUpdateFactor):]), np.nanmax(self.xLims[int((idx - h)/self.AnimationUpdateFactor):]))
        self.ax.set_ylim(np.nanmin(self.yLims[int((idx - h)/self.AnimationUpdateFactor):]), np.nanmax(self.yLims[int((idx - h)/self.AnimationUpdateFactor):]))
        self.ax.set_zlim(np.nanmin(self.zLims[int((idx - h)/self.AnimationUpdateFactor):]), np.nanmax(self.zLims[int((idx - h)/self.AnimationUpdateFactor):]))
        self.ax.set_xlabel(r'$\mathbf{x}$, m', fontsize = 14)
        self.ax.set_ylabel(r'$\mathbf{y}$, m', fontsize = 14)
        self.ax.set_zlabel(r'$\mathbf{z}$, m', fontsize = 14)
        self._handleAxisFlips()
        if self._animate_params['hide axis']:
            # self.ax.grid(False)
            self.ax.set_axis_off()
        if not self.viewIsInitialized:
            self.ax.azim = self.property_ax3d_azim
            self.ax.elev = self.property_ax3d_elev
            self.ax.roll = self.property_ax3d_roll
            self.viewIsInitialized = True
        handles = []
        # handles.append(mlines([], [], linestyle = 'None', label = 't = {:.4f} [s]'.format(idx*self.dt)))
        handles.append(mlines([], [], linestyle = 'None', label = r'$dt$ $=$ ' + '{:.4f} '.format(self.dt) + r'[$s$]'))
        handles.append(mlines([], [], linestyle = 'None', label = r'$\mathbf{t}$ $\mathbf{=}$ ' + '{:.4f}'.format(sv['time'][idx]) + r'[$\mathbf{s}$]'))
        self.ax.legend(handles = handles, loc = 'upper right')


    def animate(self, time, objectsPose, forceClose = True, figure = None, wrapper = None, wrapperKwargs = {}, axisLims = {'x':None, 'y':None, 'z':None}, hideAxis = False):
        '''
        Make animation of the simulation
        '''
        self._animate_params = {}
        self._animate_params.update({'hide axis':hideAxis})
        if forceClose:
            self.closeAll()
        if self.AnimationUpdateFactor is None:
            self.setFPS(time, 30)
        if figure is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection = '3d')
        else:
            self.fig = figure
            self.ax = self.fig.gca()
        self.wrapper = wrapper
        if wrapper is not None:
            self.updateFunc = self.wrappedUpdate
            wrapperKwargs.update({'AnimationUpdateFactor':self.AnimationUpdateFactor})
        else:
            self.updateFunc = self._update
        self.axisLims = axisLims
        updateFunc = self.fixAxisWrapper
        fargs = (objectsPose, wrapperKwargs)
        self.xLims, self.yLims, self.zLims = [], [], []
        self.ani = FuncAnimation(self.fig, updateFunc, fargs=fargs, 
                    frames = int(self.N/self.AnimationUpdateFactor),
                    save_count = int(self.N/self.AnimationUpdateFactor),
                    repeat = False)
        return self.ani


    def wrappedUpdate(self, frames, objectPoses, wrapperKwargs):
        self.wrapper(frames, objectPoses, wrapperKwargs)
        self._update(frames, objectPoses)


    def fixAxisWrapper(self, frames, objectPoses, *args):
        self.updateFunc(frames, objectPoses, *args)
        mapping = {'x':self.setXLim, 'y':self.setYLim, 'z':self.setZLim}
        for ax, lim in self.axisLims.items():
            if lim is not None:
                mapping[ax](lim)

    def setXLim(self, lim):
        self.ax.set_xlim(lim)

    def setYLim(self, lim):
        self.ax.set_ylim(lim)
        self._handleFlipY()

    def setZLim(self, lim):
        self.ax.set_zlim(lim)
        self._handleFlipZ()


    def saveAnimation(self, filename, fps = None, dpi = 300):
        '''
        Save simulation animation 
        '''
        if self.ani is None:
            raise ValueError('Cannot save animation if it has not been run! Please use viz.animate(time, objectPoses)')
        if fps is None:
            fps = self.fps
        self.ani.save(filename, writer = 'ffmpeg', fps = fps, dpi = dpi)
        self.ani.event_source.stop()
        del self.ani
        self.ani = None
        self.closeAll()
    

    def asImage(self, objectsPose, times, parentFig = None, figsize = (10, 10), uniformAxes = False, encodeRotorSpeeds = True, axisLims = {'x':None, 'y':None, 'z':None}):
        if parentFig is None:
            self.closeAll()
            self.fig2 = plt.figure(figsize = figsize)
            self.ax = self.fig2.add_subplot(111, projection = '3d')
        else:
            self.fig2 = parentFig
            self.ax = self.fig2.axes[-1]
        fctrs = np.linspace(0.1, 1, len(times))
        if len(fctrs) == 1:
            fctrs = [1]
        try:
            # Get droneParams from model, if passed in ObjectPoses
            droneParams = objectsPose[list(objectsPose.keys())[0]]['model'].droneParams
        except KeyError:
            # Get droneParams from actor
            try:
                droneParams = {}
                droneParams.update({'idle RPM':list(self.actors.values())[0].actor.minRPM})
                droneParams.update({'max RPM':list(self.actors.values())[0].actor.maxRPM})
            except AttributeError:
                droneParams = None
        # NOTE: If droneParams cannot be infered, then we need to pass some dummy omegas through
        if droneParams is not None:
            hardOmegas = np.linspace(droneParams['idle RPM'], droneParams['max RPM'], len(times))
        else:
            hardOmegas = np.linspace(0.1, 0.9, len(times))
        xlims = []
        ylims = []
        zlims = []
        for t, f, o in zip(times, fctrs, hardOmegas):
            for actr in self.actors.values():
                sv = objectsPose[actr.name]
                time = sv['time']
                idx = np.where(time >= t)[0][0]
                xyz = sv['position'][idx]
                q = sv['rotation_q'][idx]
                if encodeRotorSpeeds:
                    omega = sv['inputs'][idx]
                else:
                    omega = np.ones((1, 4))*o
                actr.actor.bodyPlotKwargs.update({'alpha':f})
                if 'ActorColorHistory' in sv.keys():
                    actr.actor._setColor(sv['ActorColorHistory'][idx])
                actr.drawBrute(xyz, q, self.ax, omega = omega)
                h = self._positionHistory(idx)
                self.ax.plot(sv['position'][(idx - h):idx, 0, 0], sv['position'][(idx - h):idx, 0, 1], sv['position'][(idx - h):idx, 0, 2], color = actr.actor.historyColor, linewidth = actr.actor.historyLinewidth)

            xlims.append(np.sort([xyz[:, 0] - actr.actor.R*8, xyz[:, 0] + actr.actor.R*8]))
            ylims.append(np.sort([xyz[:, 1] - actr.actor.R*8, xyz[:, 1] + actr.actor.R*8]))
            zlims.append(np.sort([xyz[:, 2] - actr.actor.R*8, xyz[:, 2] + actr.actor.R*8]))

        if uniformAxes:
            lims = xlims + ylims + zlims
            self.ax.set_xlim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])
            self.ax.set_ylim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])
            self.ax.set_zlim([np.sort(np.array(lims).reshape(-1))[0], np.sort(np.array(lims).reshape(-1))[-1]])  
        else:
            self.ax.set_xlim([np.sort(np.array(xlims).reshape(-1))[0], np.sort(np.array(xlims).reshape(-1))[-1]])
            self.ax.set_ylim([np.sort(np.array(ylims).reshape(-1))[0], np.sort(np.array(ylims).reshape(-1))[-1]])
            self.ax.set_zlim([np.sort(np.array(zlims).reshape(-1))[0], np.sort(np.array(zlims).reshape(-1))[-1]])

        # Override automatic axis if lims are specified
        if axisLims['x'] is not None:
            self.ax.set_xlim(axisLims['x'])
        if axisLims['y'] is not None:
            self.ax.set_ylim(axisLims['y'])
        if axisLims['z'] is not None:
            self.ax.set_zlim(axisLims['z'])
      
        self.ax.set_xlabel(r'$\mathbf{x}$, m', fontsize = 16)
        self.ax.set_ylabel(r'$\mathbf{y}$, m', fontsize = 16)
        self.ax.set_zlabel(r'$\mathbf{z}$, m', fontsize = 16)
        self._handleAxisFlips()
        self.fig2.subplots_adjust(left = 0, right = 0.94, bottom = 0, top = 1)
        fig = self.fig2
        # fig.show()
        return fig

    def setFPS(self, time, fps):
        self.N = len(time)
        self.dt = time[1] - time[0]
        self.fps = fps
        self.AnimationUpdateFactor = (1/self.dt)/fps

    def check_alive(self):
        return plt.fignum_exists(self.fig.number)



'''
Class to wrap actor objects. 
Any classes this inherits from must have the .draw() and .update() methods
- These methods must take the arguments as shown below
- update() updates the pose of the actor
- draw() draws this updated pose on the visualization window
'''
class actor:

    def __init__(self, actr, name):
        self.name = name
        self.actor = actr
        self.xyz = self.actor.origin
        self.q = self.actor.quat

    def _initDraw(self, xyz, rotation, ax, **kwargs):
        self.update(xyz, rotation)
        self.actor._initDraw(ax, **kwargs)

    def draw(self, xyz, rotation, ax, **kwargs):
        self.update(xyz, rotation)
        self.actor.draw(ax, **kwargs)

    def drawBrute(self, xyz, rotation, ax, **kwargs):
        self.update(xyz, rotation)
        self.actor.drawBrute(ax, **kwargs)

    def update(self, xyz, rotation):
        self.actor.update(xyz, rotation)