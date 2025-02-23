import numpy as np

# This is your model class. Feel free to change the name to something more suitable.
# Below are the mandatory class methods.
# In the __init__ function, define/assign/load/import model parameters
# The getForcesMoments method is used by the simulation to run apply the model
class exampleSim_model:
	def __init__(self, k = 0, c = 0, m = 0):
		modelParams = {'k':k, 'c':c, 'm':m}
		self.modelParams = modelParams
		self.name = 'massSpringDamper_model'
		return None

	def getForcesMoments(self, simVars):
		# Extract vars
		step = simVars['currentTimeStep_index']
		trueAction = simVars['inputs'][step + 1] # True action just updated, so current = step + 1
		state = simVars['state'][step] # State not yet updated, so current = step
		# Calculate forces and moments here. Create and call relevant functions if necessary
		mass = self.modelParams['m']
		k = self.modelParams['k']
		c = self.modelParams['c']
		u = trueAction/mass
		wn = np.sqrt(k/mass)
		damping = c/(2*mass*wn)
		# Compute force
		_F = np.array(mass*(u - 2*damping*wn*state[:, 1] - wn**2 * state[:, 0])).reshape(-1)
		F = np.hstack((_F, 0, 0))
		# No moments
		M = np.zeros((1, 3))
		return F, M