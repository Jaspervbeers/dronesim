'''
Actuator dynamics for example (mass-spring-damper) system

Created by: Jasper van Beers (j.j.vanbeers@tudelft.nl; jasper@vanbeers.dev)
Last modified: 22-05-2023
'''

class exampleSim_actuator:
	def __init__(self):
		self.name = 'example_MSD_actuator'
		return None
	
	def setInitial(self, x):
		self.previousCommand = x

	def actuate(self, simVars):
		# Place actuator dynamics here. Create and call relevant functions if necessary
		step = simVars['currentTimeStep_index']
		commandedInput = simVars['inputs_CMD'][step + 1] # Command just updated, so next step
		trueInput = commandedInput # This line is equivalent to no actuator dynamics
		self.previousCommand = trueInput
		return trueInput