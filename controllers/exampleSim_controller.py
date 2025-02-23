from controllers import PID

class exampleSim_controller:
	def __init__(self, P, I, D, noiseBlock = None):
		self.PID = PID.PID(P, I, D)
		self.name = 'example_MSD_controller'
		if noiseBlock is None:
			raise ValueError('Must have noiseBlock!')
		self.noiseBlock = noiseBlock
		return None

	def control(self, simVars):
		step = simVars['currentTimeStep_index']
		dt = simVars['dt']
		reference = simVars['reference'][step]
		self.noiseBlock.addStateNoise(simVars)
		state = simVars['state_noisy'][step]
		# Control only position
		error = reference - state[:, 0]
		controlInput = self.PID.control(error, dt)
		return controlInput