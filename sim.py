import numpy as np
from tqdm import tqdm
from funcs import angleFuncs

class sim:

    def __init__(self, model, EOM, controller, actuator, showAnimation = True, animator = None, integrator = 'Euler', intergratorKwargs = {}, angleIndices = [0, 1, 2], run_mode = 'finite'):
        # Determine if simulation is finite (has pre-defined end time) or if infinite (can continue until user quits)
        if run_mode.lower() not in ['finite', 'infinite']:
            raise ValueError(f'Unknown run_mode: ({run_mode}). Expect "infinite" or "finite"')
        self.run_mode = 0 if run_mode == 'finite' else 1
        if self.run_mode == 1 and animator is None:
            raise ValueError('Cannot run simulation with no end time without an animator.')
        self.chunk_size = 1000
        self._chunk_vars_to_extend = []
        self._run_mode_1_counter = 0
        self._run_mode_1_default_dt = 0.004
        self._run_mode_1_handler = {True:self._extend_simVars, False:self._pass}
        self.run_mode_1_termination = None
        self.simVars = {}
        self.model = model
        self.EOM = EOM
        self.controller = controller
        self.actuator = actuator
        self.integrator_properties = {'kwargs':intergratorKwargs}
        knownIntegrators = {'Euler':self._integrator_Euler}
        if integrator not in knownIntegrators:
            raise ValueError('Unknown integrator scheme "{}"'.format(integrator))
        else:
            self.integrator = knownIntegrators[integrator]

        self.angleIndices = angleIndices
        
        # Wrapable functions, outside of the simulation, we assign 
        # different functions to these to do extra steps before/after these blocks
        # -> For example, see wrapForecaster below
        self.doControl = self.controller.control
        self.doActuation = self.actuator.actuate
        self.doForcesMoments = self.model.getForcesMoments
        self.doEOM = self.EOM

        self.animate = self._dummyAnimator
        self.showAnimation = showAnimation
        self.animator = animator
        if animator is not None:
            # self.animationUpdateRateFactor = 8 # ~30 FPS for saved animation
            self.animationUpdateRateFactor = 4 # ~60 FPS for saved animation
            # Set default termination condition for infinite simulations
            self.setTerminationCondition(self._animatorCloseTermination)
        return None

    def _integrator_Euler(self, simVars):
        step = simVars['currentTimeStep_index']
        x = simVars['state'][step]
        # NOTE: we take step + 1 for state derivative since it is updated from EOM (i.e. latest available)
        x_dot = simVars['stateDerivative'][step+1]
        dt = simVars['dt']
        return x + x_dot*dt

    def assignIntegrator(self, integrator):
        self.integrator = integrator

    def setInitial(self, time, state, inputs, reference, **kwargs):
        if self.run_mode == 0:
            self.time = time
            self.T = time[-1]
            self.dt = 0
            self.state = state
            self.inputs = inputs
            self.reference = reference
        elif self.run_mode == 1:
            # If run_mode is infinite, assume supplied information is only vector
            self.time = np.ones(self.chunk_size) * -1
            self.T = None
            self.dt = self._run_mode_1_default_dt
            self.state = np.zeros((len(self.time), 1, state.shape[-1]))
            self.inputs = np.zeros((len(self.time), 1, inputs.shape[-1]))
            self.reference = np.zeros((len(self.time), 1, reference.shape[-1]))
            self.time[0] = time[0]
            self.state[0, :, :] = state[0, :, :]
            self.inputs[0, :, :] = inputs[0, :, :]
            self.reference[0, :, :] = reference[0, :, :]
            # self._quit_event_capture = keyboard_utils._Getch_quit()
        if len(self.time) != len(self.state) or len(self.time) != len(self.inputs) or len(self.time) != len(self.reference):
            raise ValueError('Axis mismatch; either the state, inputs, or reference vector does not match the time array')
        self.currentTimeStep_index = 0
        self.stateDerivative = np.zeros(self.state.shape)
        if len(self.angleIndices) > 0:
            self.quat = np.zeros((len(self.time), 1, 4))
            self.quat[0, :, :] = angleFuncs.Eul2Quat(self.state[0, :, self.angleIndices])
            self.quatDot = np.zeros((len(self.time), 1, 4))
        else:
            self.quat = None
            self.quatDot = None
        self.forces = np.zeros((len(self.time), 1, 3))
        self.moments = np.zeros((len(self.time), 1, 3))
        self.inputs_CMD = self.inputs.copy()
        self._assignSimVars(**kwargs)
        if self.animator is not None and self.showAnimation:
            # self.ani = self.animator.InitAni([self.simVars])
            # self.animate = self.animator.animate
            self.addSimVar('AnimationUpdateFactor', self.animationUpdateRateFactor)
            self.addSimVar('animator', self.animator)
            self.animate = self.animator.update
            self.animator._initActorsDraw({self.model.modelID:self.simVars})

    def _assignSimVars(self, **kwargs):
        # Add essential
        self.addSimVar('time', self.time)
        self.addSimVar('T', self.T)
        self.addSimVar('dt', self.dt)
        self.addSimVar('state', self.state)
        self.addSimVar('state_noisy', self.state.copy()*0, addToChunkVarsToExtend=True)
        self.addSimVar('quat', self.quat)
        self.addSimVar('quatDot', self.quatDot)
        self.addSimVar('inputs', self.inputs)
        self.addSimVar('inputs_CMD', self.inputs_CMD)
        self.addSimVar('reference', self.reference)
        self.addSimVar('currentTimeStep_index', self.currentTimeStep_index)
        self.addSimVar('stateDerivative', self.stateDerivative)
        self.addSimVar('stateDerivative_noisy', self.stateDerivative.copy()*0, addToChunkVarsToExtend=True)
        self.addSimVar('forces', self.forces)
        self.addSimVar('moments', self.moments)
        self.addSimVar('model', self.model)
        self.addSimVar('controller', self.controller)
        self.addSimVar('EOM', self.EOM)
        self.addSimVar('actuator', self.actuator)
        self.addSimVar('intergrator', self.integrator)
        self.addSimVar('intergrator_properties', self.integrator_properties)
        # Add any additional arguments
        if len(kwargs):
            for k in kwargs.keys():
                self.addSimVar(k, kwargs[k])

    def addSimVar(self, name, variable, addToChunkVarsToExtend=False):
        # Check for conflicts
        if name in self.simVars.keys():
            raise ValueError(f'{name} already exists in simulation variables!')
        else:
            self.simVars.update({name:variable})
        if addToChunkVarsToExtend:
            self._chunk_vars_to_extend.append(name)

    def updateSimVar(self, name, variable):
        self.simVars.update({name:variable})

    def _dummyAnimator(self, *args):
        return None

    def _extend_chunk(self, target, val = 0):
        shape = target.shape
        N = shape[-1]
        extension = np.zeros((self.chunk_size, 1, N)) + val
        target_extended = np.concatenate((target, extension), axis = 0).reshape(-1, 1, N)
        return target_extended

    def _extend_simVars(self):
        self.time = np.concatenate((self.time, -1*np.ones(self.chunk_size)))
        self.state = self._extend_chunk(self.state)
        # Need to update as attributes are re-assigned
        self.quat = self._extend_chunk(self.quat)
        self.quatDot = self._extend_chunk(self.quatDot)
        self.inputs = self._extend_chunk(self.inputs)
        self.inputs_CMD = self._extend_chunk(self.inputs_CMD)
        self.reference = self._extend_chunk(self.reference, val = self.reference[-3])
        self.stateDerivative = self._extend_chunk(self.stateDerivative)
        self.forces = self._extend_chunk(self.forces)
        self.moments = self._extend_chunk(self.moments)
        self.updateSimVar('state', self.state)
        self.updateSimVar('quat', self.quat)
        self.updateSimVar('quatDot', self.quatDot)
        self.updateSimVar('inputs', self.inputs)
        self.updateSimVar('inputs_CMD', self.inputs_CMD)
        self.updateSimVar('reference', self.reference)
        self.updateSimVar('stateDerivative', self.stateDerivative)
        self.updateSimVar('forces', self.forces)
        self.updateSimVar('moments', self.moments)
        self._run_mode_1_counter = 0
        for v in self._chunk_vars_to_extend:
            self.updateSimVar(v, self._extend_chunk(self.simVars[v]))

    def _trim_simVars(self, N):
        self.time = self.time[:N]
        self.state = self.state[:N]
        self.quat = self.quat[:N]
        self.quatDot = self.quatDot[:N]
        self.inputs = self.inputs[:N]
        self.inputs_CMD = self.inputs_CMD[:N]
        self.reference = self.reference[:N]
        self.stateDerivative = self.stateDerivative[:N]
        self.forces = self.forces[:N]
        self.moments = self.moments[:N]
        # Need to update since attributes are reassigned
        self.updateSimVar('state', self.state)
        self.updateSimVar('quat', self.quat)
        self.updateSimVar('quatDot', self.quatDot)
        self.updateSimVar('inputs', self.inputs)
        self.updateSimVar('inputs_CMD', self.inputs_CMD)
        self.updateSimVar('reference', self.reference)
        self.updateSimVar('stateDerivative', self.stateDerivative)
        self.updateSimVar('forces', self.forces)
        self.updateSimVar('moments', self.moments)
        for v in self._chunk_vars_to_extend:
            self.simVars[v] = self.simVars[v][:N]

    def _pass(self):
        pass

    
    def setTerminationCondition(self, func, funcKwargs = {}):
        self.run_mode_1_termination = func
        self._run_mode_1_termination_kwargs = funcKwargs


    def _animatorCloseTermination(self, **kwargs):
        return self.animator.check_alive()


    def run(self):
        if self.run_mode == 0:
            for i in tqdm(range(len(self.time) - 1), leave = False):
                self.currentTimeStep_index = i
                # Need to update index in simVars since we reassign it here
                self.updateSimVar('currentTimeStep_index', self.currentTimeStep_index)
                # Update timestep, in case it is variable
                self.dt = self.time[self.currentTimeStep_index+1] - self.time[self.currentTimeStep_index]
                # Need to update dt in simVars since we reassign it here
                self.updateSimVar('dt', self.dt)
                self._run_step(i)
        elif self.run_mode == 1:
            if self.run_mode_1_termination is None:
                raise ValueError('No termination condition set for simulation (and no animator is passed).\n Please set a termination condition through sim.setTerminationCondition(function)')
            self._run_while()
            # As we extend data arrays by "chunks" when needed, we may terminate before the end of a chunk
            filled = np.where(self.time < 0)[0][0]
            self._trim_simVars(filled)
            self.T = self.time[-1]

        if self.animator is not None:
            ani = self.animator.posteriorAnimation({self.model.modelID:self.simVars})
            aniSaver = self.animator.saveAnimation
            self.addSimVar('ani', ani)
            self.addSimVar('aniSaver', aniSaver)
    
    def _run_while(self):
        i = 0
        while self.run_mode_1_termination(**self._run_mode_1_termination_kwargs):
            self._run_mode_1_handler[self._run_mode_1_counter == self.chunk_size-2]()
            self.currentTimeStep_index = i
            # Need to update index in simVars since we reassign it here
            self.updateSimVar('currentTimeStep_index', self.currentTimeStep_index)
            # Propagate time
            self.time[i + 1] = self.time[i] + self._run_mode_1_default_dt
            self.updateSimVar('dt', self._run_mode_1_default_dt)
            self._run_step(i)
            self._run_mode_1_counter += 1
            i += 1

    def _run_step(self, step):
        # Call controller
        self.inputs_CMD[self.currentTimeStep_index + 1] = self.doControl(self.simVars)
        # Convert commanded inputs into true inputs through actuator
        self.inputs[self.currentTimeStep_index + 1] = self.doActuation(self.simVars)
        # Use models to get forces and moments from current state and inputs
        self.forces[self.currentTimeStep_index + 1], self.moments[self.currentTimeStep_index + 1] = self.doForcesMoments(self.simVars)
        # Use equations of motion to obtain state derivative & move quadrotor
        # self.stateDerivative[self.currentTimeStep_index + 1] = self.EOM(self.simVars)
        self.stateDerivative[self.currentTimeStep_index + 1] = self.doEOM(self.simVars)
        # Update state using integrator
        self.state[self.currentTimeStep_index + 1] = self.integrator(self.simVars)
        # Animate, if animator is passed
        if self.animator is not None:
            self.animate(step, {self.model.modelID:self.simVars})
        # self.animate()