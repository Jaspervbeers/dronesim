import numpy as np

# This function must return the state derivative.
def EOM(simVars):
	# Extract necessary parameters
	modelParams = simVars['model'].modelParams
	step = simVars['currentTimeStep_index']
	state = simVars['state'][step] # State not yet updated in simulation loop, so current = step
	F = simVars['forces'][step + 1][0] # Force just updated in simulation loop, so current = step + 1
	# Calculate state derivative here
	x_dot = np.matmul(np.matrix([[0, 1], [-modelParams['k']/modelParams['m'], -modelParams['c']/modelParams['m']]]), state.T).reshape(-1, 1) + np.matrix([[0], [F[0]/modelParams['m']]])
	return np.array(x_dot).reshape(-1)