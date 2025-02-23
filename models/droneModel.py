'''
Class which imports and makes use of the identified drone model

Created by Jasper van Beers
Contact: j.j.vanbeers@tudelft.nl
Date: 09-02-2022
'''
import os
import dill as pickle
import numpy as np
import json
from scipy.misc import derivative

class model:
	def __init__(self, path = None, modelID = None, NoOffset = False):
		# If no path to model folder specified, take default model
		if path is None:
			cwd = os.getcwd()
			# self.modelID = 'MDL-MetalBeetle-Indoor-022022-001'
			self.modelID = 'MDL-HDBeetle-NN-II-NGP003'
			self.modelPath = os.path.join(cwd, 'models', self.modelID)
		else:
			self.modelPath = path
			if modelID is None:
				self.modelID = self.modelPath.split(os.sep)[-1]
			else:
				self.modelID = modelID
				self.modelPath = os.path.join(path, modelID)
		# Load models
		self.FxModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-Fx.pkl'))
		self.FyModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-Fy.pkl'))
		self.FzModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-Fz.pkl'))
		self.MxModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-Mx.pkl'))
		self.MyModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-My.pkl'))
		self.MzModel = self._loadModels(os.path.join(self.modelPath, self.modelID + '-Mz.pkl'))
		# Get drone params
		self.droneParams = self.FxModel.droneParams
		self._simplifiedModel = None
		if 'simpleModel' in self.droneParams.keys():
			self._simplifiedModel = self.droneParams['simpleModel']
		self.isNormalized = self.FxModel.isNormalized
		self.hasGravity = self.FxModel.hasGravity
		# Set offsets, if they exist
		self.NoOffset = NoOffset
		if not self.NoOffset:
			self.checkOffsets()
		self.setOffsets()
		# Check if there is a derivative system available
		self.DiffSys = None
		if os.path.exists(os.path.join(self.modelPath, 'PQR-DiffSys.pkl')):
			self.DiffSys = self._loadModels(os.path.join(self.modelPath, 'PQR-DiffSys.pkl'))
		return None


	def _loadModels(self, path):
		with open(path, 'rb') as f:
			loadedModel = pickle.load(f)
		return loadedModel



	def getSignMask(self):
		# Define dictionary which maps the rotor numbers to their position on the quadrotor (e.g. rotor 4 is located 'front left')
		invRotorConfig = {v:k for k, v in self.droneParams['rotor configuration'].items()}
		# Extract ditcionary which maps the yaw sign to CW (clockwise) and CCW (counterclockwise) rotation of the rotors
		r_sign = self.droneParams['r_sign']
		# Create a toggle to parse through rotor mapping using text cues (e.g. front, left, etc.)
		signMap = {True:1, False:-1}
		# Create a matrix mask describing the signs of each rotor on each of the control moments
		# -> A matrix which describes how the rotor configuration produces rotations of the quadrotor and the sign therein
		self.signMask = np.array([[signMap[invRotorConfig[1].endswith('left')], signMap[invRotorConfig[2].endswith('left')], signMap[invRotorConfig[3].endswith('left')], signMap[invRotorConfig[4].endswith('left')]],
							[signMap[invRotorConfig[1].startswith('front')], signMap[invRotorConfig[2].startswith('front')], signMap[invRotorConfig[3].startswith('front')], signMap[invRotorConfig[4].startswith('front')]],
							r_sign[self.droneParams['rotor 1 direction']]*np.array([1, -1, -1, 1])]).T
		return self.signMask.copy()



	def getForcesMoments(self, simVars):
		step = simVars['currentTimeStep_index']
		# We use the true state here (instead of noisy) since this is where the system really is. 
		state = simVars['state'][step][:, :9]
		# NOTE: trueAction is step+1 since we just updated actuator information in sim loop. (i.e. latest actuator info)
		trueAction = simVars['inputs'][step+1]

		# Prepare inputs
		modelInputs = self.FxModel.droneGetModelInput(state, trueAction)

		if self.isNormalized:
			# Extract normalizing factors
			F_den = np.array(modelInputs['F_den']).reshape(-1)
			M_den = np.array(modelInputs['M_den']).reshape(-1)
		else:
			F_den, M_den = 1, 1		

		# Make predictions
		Fx = self.FxModel.predict(modelInputs).__array__()[0][0]*F_den - self.Fx0
		Fy = self.FyModel.predict(modelInputs).__array__()[0][0]*F_den - self.Fy0
		Fz = self.FzModel.predict(modelInputs).__array__()[0][0]*F_den - self.Fz0
		Mx = self.MxModel.predict(modelInputs).__array__()[0][0]*M_den - self.Mx0
		My = self.MyModel.predict(modelInputs).__array__()[0][0]*M_den - self.My0
		Mz = self.MzModel.predict(modelInputs).__array__()[0][0]*M_den - self.Mz0

		# import code
		# code.interact(local=locals())

		# Build force and moment vectors
		F = (np.array([Fx, Fy, Fz])).reshape(1, -1)
		M = (np.array([Mx, My, Mz])).reshape(1, -1)
		
		return F, M

	def checkOffsets(self, numIterations = 3, meshSize = 1000):
		if not self.NoOffset:
			if not os.path.exists(os.path.join(self.modelPath, 'FMOffsets.json')):
				print('[ WARNING ] Could not find FM offsets in droneParams. Attempting to approximate.')
				# First, attempt to find hovering thrust
				omegaMin = self.droneParams['idle RPM']
				omegaMax = self.droneParams['max RPM']
				if 'g' not in self.droneParams.keys():
					g = 9.81
				else:
					g = self.droneParams['g']		
				for i in range(numIterations):
					omegaHover = self._searchHoverThrust(g, omegaMin, omegaMax, meshSize=meshSize)
					# Next iteration parameters
					dOmega = 0.1*omegaHover
					omegaMin = omegaHover - dOmega
					omegaMax = omegaHover + dOmega
				# Use estimate of hovering thrust to determine FM offsets
				state = np.zeros((1, 9))
				omega = np.ones((1, 9))*omegaHover
				modelInputs = self.FzModel.droneGetModelInput(state, omega)
				if self.isNormalized:
					# Extract normalizing factors
					F_den = np.array(modelInputs['F_den']).reshape(-1)
					M_den = np.array(modelInputs['M_den']).reshape(-1)
				else:
					F_den, M_den = 1, 1
				Fx0_est = self.FxModel.predict(modelInputs).__array__()[0][0]*F_den
				Fy0_est = self.FyModel.predict(modelInputs).__array__()[0][0]*F_den
				Fz0_est = self.FzModel.predict(modelInputs).__array__()[0][0]*F_den	+ self.droneParams['m']*g
				Mx0_est = self.MxModel.predict(modelInputs).__array__()[0][0]*M_den
				My0_est = self.MyModel.predict(modelInputs).__array__()[0][0]*M_den
				Mz0_est = self.MzModel.predict(modelInputs).__array__()[0][0]*M_den
				FMOffsets = {
					'Fx0':float(Fx0_est/F_den),
					'Fy0':float(Fy0_est/F_den),
					'Fz0':float(Fz0_est/F_den),
					'Mx0':float(Mx0_est/M_den), 
					'My0':float(My0_est/M_den),
					'Mz0':float(Mz0_est/M_den)}
				# Save values to file and set moments
				with open(os.path.join(self.modelPath, 'FMOffsets.json'), 'w') as f:
					json.dump(FMOffsets, f, indent = 4)
				self.setOffsets()
			else:
				print('[ INFO ] Found existing FM offsets, already using these.')
		else:
			self.setZeroOffset()


	def _searchHoverThrust(self, g, omegaMin, omegaMax, meshSize = 1000):
		state = np.zeros((meshSize, 9))
		omega = np.zeros((meshSize, 4)) + np.linspace(omegaMin, omegaMax, num = meshSize).reshape(-1, 1)
		modelInputs = self.FxModel.droneGetModelInput(state, omega)
		Fz = self.FzModel.predict(modelInputs)
		if self.isNormalized:
			# Extract normalizing factors
			F_den = np.array(modelInputs['F_den']).reshape(-1)
			Fz = np.array(Fz).reshape(-1)*F_den
		FzHover = self.droneParams['m']*g
		idxHover = np.argmin(np.abs(Fz + FzHover))
		# print(f'Omega: {omega[idxHover][0]} \t Fz_h: {FzHover} \t Fz: {Fz[idxHover]}')
		return omega[idxHover][0]


	def linearize(self, x, u, dxs = None, dus = None):
		if dxs is None:
			dxs = x.copy().reshape(-1)[:9]
			dxs[:3] = 0.1 # Attitude, radians
			dxs[3:6] = 0.1 # Velocity, radians
			dxs[6:] = np.sqrt(0.1) # Rate, radians
		if dus is None:
			dus = u.copy().reshape(-1)[:4]
			dus[:] = 100
		self.LinearFx = _linearModel(self.FxModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.Fx0, modelType = 'force')
		self.LinearFy = _linearModel(self.FyModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.Fy0, modelType = 'force')
		self.LinearFz = _linearModel(self.FzModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.Fz0, modelType = 'force')
		self.LinearMx = _linearModel(self.MxModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.Mx0, modelType = 'moment')
		self.LinearMy = _linearModel(self.MyModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.My0, modelType = 'moment')
		self.LinearMz = _linearModel(self.MzModel, x, u, dxs, dus, self.getSignMask().T, modelBias = self.Mz0, modelType = 'moment')

	
	def getForcesMoments_linear(self, simVars):
		step = simVars['currentTimeStep_index']
		# We use the true state here (instead of noisy) since this is where the system really is. 
		state = simVars['state'][step][:, :9]
		# NOTE: trueAction is step+1 since we just updated actuator information in sim loop. (i.e. latest actuator info)
		trueAction = simVars['inputs'][step+1]

		# Make predictions
		Fx = self.LinearFx.predict(state, trueAction)
		Fy = self.LinearFy.predict(state, trueAction)
		Fz = self.LinearFz.predict(state, trueAction)
		Mx = self.LinearMx.predict(state, trueAction)
		My = self.LinearMy.predict(state, trueAction)
		Mz = self.LinearMz.predict(state, trueAction)

		# Build force and moment vectors
		F = (np.array([Fx, Fy, Fz])).reshape(1, -1)
		M = (np.array([Mx, My, Mz])).reshape(1, -1)
		
		return F, M


	def _linear2StateSpace(self):
		'''
		Returns a state space representation for the linearized models around point a, b in the form
			dx_dot = A*dx + B*du 
		where d denotes delta operator (i.e. dx = (x - a), du = (u - b), and dx_dot = (x_dot - f(a, b)))
			-> Typically a, b is chosen such that f(a,b) = 0. But this is not always the case. 
		NOTE: We also assume that derivative of attitude (roll, pitch, yaw) is approximately equal to p, q, r. 
			-> Invovling the proper transformation added non-linearity. 
			-> Hence, we assume that p, q, r are not changing rapidly 
		Output:
			- A matrix
			- B matrix
			- C matrix
			- D matrix 
			- Deltas (vector of f(a,b) to add to dY = Cx + Du to obtain true Y)
		'''
		# x = [roll, pitch, yaw, u, v, w, p, q, r]
		# dx = [p, q, r, ax, ay, az, p_dot, q_dot, r_dot]
		# 	Linearized models give back forces and moments, so we need to apply appropriate scaling according to droneParams
		
		# Build A matrix
		# Attitudes -> roll_dot, pitch_dot, yaw_dot = p, q, r
		A_att = np.zeros((3, 9))
		A_att[0, 6] = 1 # roll_dot = p
		A_att[1, 7] = 1	# pitch_dot = q
		A_att[2, 8] = 1 # yaw_dot = r

		# Velocities -> ax, ay, az
		A_vel = np.zeros((3, 9))
		accFactor = 1/self.droneParams['m']

		# Rates -> 
		A_rates = np.zeros((3, 9))
		invIv = np.linalg.inv(self.droneParams['Iv'])

		for col in range(9):
			# Velocities
			A_vel[0, col] = accFactor*self.LinearFx.fx[col]
			A_vel[1, col] = accFactor*self.LinearFy.fx[col]
			A_vel[2, col] = accFactor*self.LinearFz.fx[col]

			# Rates
			A_rates[0, col] = np.matmul(invIv[0, :], np.array([self.LinearMx.fx[col], self.LinearMy.fx[col], self.LinearMz.fx[col]]))
			A_rates[1, col] = np.matmul(invIv[1, :], np.array([self.LinearMx.fx[col], self.LinearMy.fx[col], self.LinearMz.fx[col]]))
			A_rates[2, col] = np.matmul(invIv[2, :], np.array([self.LinearMx.fx[col], self.LinearMy.fx[col], self.LinearMz.fx[col]]))

		A = np.vstack((A_att, A_vel, A_rates))

		# Build B matrix
		B = np.zeros((9, 4))

		# Inputs do not influence roll_dot, pitch_dot, and yaw_dot, so remain at zero
		for col in range(4):
			# Accelerations
			B[3, col] = accFactor*self.LinearFx.fu[col]
			B[4, col] = accFactor*self.LinearFy.fu[col]
			B[5, col] = accFactor*self.LinearFz.fu[col]

			# Rates
			B[6, col] = np.matmul(invIv[0, :], np.array([self.LinearMx.fu[col], self.LinearMy.fu[col], self.LinearMz.fu[col]]))
			B[7, col] = np.matmul(invIv[1, :], np.array([self.LinearMx.fu[col], self.LinearMy.fu[col], self.LinearMz.fu[col]]))
			B[8, col] = np.matmul(invIv[2, :], np.array([self.LinearMx.fu[col], self.LinearMy.fu[col], self.LinearMz.fu[col]]))

		# Construct C and D to return true delta forces (i.e. ignoring f(a, b) and any additional biases)
		C = np.zeros((6, 9))
		C[0, 3] = self.droneParams['m']
		C[1, 4] = self.droneParams['m']
		C[2, 5] = self.droneParams['m']
		C[3:, 6:] = self.droneParams['Iv']

		D = np.zeros((6, 4))

		# Construct final bias/delta vector
		Deltas = np.zeros((6, 1))
		Deltas[0] = self.LinearFx.f0 - self.LinearFx.bias
		Deltas[1] = self.LinearFy.f0 - self.LinearFy.bias
		Deltas[2] = self.LinearFz.f0 - self.LinearFz.bias
		Deltas[3] = self.LinearMx.f0 - self.LinearMx.bias
		Deltas[4] = self.LinearMy.f0 - self.LinearMy.bias
		Deltas[5] = self.LinearMz.f0 - self.LinearMz.bias

		self.LinearModelStateSpace = (A, B, C, D, Deltas)

		return A, B, C, D, Deltas


	def setOffsets(self):
		if not self.NoOffset:
			try:
				with open(os.path.join(self.modelPath, 'FMOffsets.json'), 'r') as f:
					FMOffsets = json.load(f)
				self.Fx0 = np.float(FMOffsets['Fx0'])
				self.Fy0 = np.float(FMOffsets['Fy0'])
				self.Fz0 = np.float(FMOffsets['Fz0'])				
				self.Mx0 = np.float(FMOffsets['Mx0'])
				self.My0 = np.float(FMOffsets['My0'])
				self.Mz0 = np.float(FMOffsets['Mz0'])
			except FileNotFoundError:
				self.setZeroOffset()
		else:
			self.setZeroOffset()
	
	def setZeroOffset(self):
		self.Fx0 = 0
		self.Fy0 = 0
		self.Fz0 = 0
		self.Mx0 = 0
		self.My0 = 0
		self.Mz0 = 0

	def _bundle(self, name, model, offset):
		m = {'name':name}
		m.update({'regressors':model.polynomial})
		m.update({'coefficients':str(list(model.coefficients.__array__().reshape(-1)))})
		m.update({'Offset':offset})
		return m

	def toFile(self, filename = 'model.json'):
		try:
			_ = self.FxModel.polynomial
		except KeyError:
			raise NotImplementedError('toFile only supported for (standalone)Polynomial models')
		m = {}
		m.update({'Fx':self._bundle('Fx-poly', self.FxModel, self.Fx0)})
		m.update({'Fy':self._bundle('Fy-poly', self.FyModel, self.Fy0)})
		m.update({'Fz':self._bundle('Fz-poly', self.FzModel, self.Fz0)})
		m.update({'Mx':self._bundle('Mx-poly', self.MxModel, self.Mx0)})
		m.update({'My':self._bundle('My-poly', self.MyModel, self.My0)})
		m.update({'Mz':self._bundle('Mz-poly', self.MzModel, self.Mz0)})
		# Add droneParams
		with open(f'{filename}', 'w') as f:
			json.dump(m, f, indent = 4)
		return None


class _linearModel:

	def __init__(self, model, xa, ub, dxs, dus, signMask, order = 3, modelBias = 0, modelType = 'force'):
		if modelType.lower() not in ['force', 'moment']:
			raise ValueError(f'_linearModel: Expected modelType of "force" or "moment" but {modelType} passed')
		xin = model.droneGetModelInput(xa, ub)
		if modelType.lower() == 'force':
			self.modelNorm = float(np.array(xin['F_den']).reshape(-1))
		elif modelType.lower() == 'moment':
			self.modelNorm = float(np.array(xin['M_den']).reshape(-1))
		self.bias = modelBias
		self.signMask = signMask
		self.xa = xa
		self.ub = ub
		self.model = model
		self.f0 = model.predict(xin).__array__()[0][0]
		fx = []
		for i, (xi, dxi) in enumerate(zip(xa[0, :], dxs)):
			fx.append(derivative(self.wrapx, xi, n = 1, dx = dxi, order = order, args = (i,)))
		self.fx = np.array(fx) 
		fu = []
		for i, (ui, dui) in enumerate(zip(ub[0, :], dus)):
			fu.append(derivative(self.wrapu, ui, n = 1, dx = dui, order = order, args = (i,)))
		self.fu = np.array(fu)

	def __call__(self, x, u):
		return self.predict(x, u)

	def predict(self, x, u):
		dx = x - self.xa
		du = u - self.ub
		fx = np.nansum((dx.reshape(-1)*self.fx))
		fu = np.nansum((du.reshape(-1)*self.fu))
		return (self.f0 + fx + fu - self.bias)/self.modelNorm

	def wrapx(self, x, idx):
		xa = self.xa.copy()
		xa[:, idx] = x
		xin = self.model.droneGetModelInput(xa, self.ub)
		return self.model.predict(xin).__array__()[0][0]

	def wrapu(self, y, idx):
		ub = self.ub.copy()
		ub[:, idx] = y
		xin = self.model.droneGetModelInput(self.xa, ub)
		return self.model.predict(xin).__array__()[0][0]


class simpleModel:

	def __init__(self, kappa = 8.720395416240164e-07, tau = 4.29e-09, hasGravity = True):
		from funcs.angleFuncs import Eul2Quat, QuatRot
		self.modelID = 'SimpleModel'
		# Get drone params
		self.droneParams = {
							'R': 0.0381, 
							'b': 0.077, 
							'Iv': np.array([[0.000865, 0.      , 0.      ],
       										[0.      , 0.00107 , 0.      ],
											[0.      , 0.      , 0.00171 ]]), 
							'rotor configuration': {
								'front left': 4, 
								'front right': 2, 
								'aft right': 1, 
								'aft left': 3
							}, 
							'rotor 1 direction': 'CW', 
							'idle RPM': 200.0, 
							'max RPM': 2100.0, 
							'm': 0.433, 
							'g': 9.81, 
							'rho': 1.225, 
							'r_sign': {'CCW': -1, 'CW': 1}
							}
		self.getSignMask()
		self.kappa = kappa
		self.tau = tau
		self.MBc = np.array([
			[self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa],
			[self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa, self.droneParams['b']*self.kappa],
			[self.tau, self.tau, self.tau, self.tau]
		])*self.signMask.T
		self.hasGravity = hasGravity
		self.Eul2Quat = Eul2Quat
		self.QuatRot = QuatRot

		
	def getSignMask(self):
		# Define dictionary which maps the rotor numbers to their position on the quadrotor (e.g. rotor 4 is located 'front left')
		invRotorConfig = {v:k for k, v in self.droneParams['rotor configuration'].items()}
		# Extract ditcionary which maps the yaw sign to CW (clockwise) and CCW (counterclockwise) rotation of the rotors
		r_sign = self.droneParams['r_sign']
		# Create a toggle to parse through rotor mapping using text cues (e.g. front, left, etc.)
		signMap = {True:1, False:-1}
		# Create a matrix mask describing the signs of each rotor on each of the control moments
		# -> A matrix which describes how the rotor configuration produces rotations of the quadrotor and the sign therein
		self.signMask = np.array([[signMap[invRotorConfig[1].endswith('left')], signMap[invRotorConfig[2].endswith('left')], signMap[invRotorConfig[3].endswith('left')], signMap[invRotorConfig[4].endswith('left')]],
							[signMap[invRotorConfig[1].startswith('front')], signMap[invRotorConfig[2].startswith('front')], signMap[invRotorConfig[3].startswith('front')], signMap[invRotorConfig[4].startswith('front')]],
							r_sign[self.droneParams['rotor 1 direction']]*np.array([1, -1, -1, 1])]).T
		return self.signMask.copy()


	def getForcesMoments(self, simVars):
		step = simVars['currentTimeStep_index']
		# We use the true state here (instead of noisy) since this is where the system really is. 
		state = simVars['state'][step][:, :9].copy()
		# NOTE: trueAction is step+1 since we just updated actuator information in sim loop. (i.e. latest actuator info)
		trueAction = simVars['inputs'][step+1].copy()

		# Make predictions
		Fx = 0
		Fy = 0
		Fz = -1*self.kappa*np.sum(np.square(trueAction))
		# if self.hasGravity:
		# 	Fz += self.droneParams['g']*self.droneParams['m']

		M = np.matmul(self.MBc, np.square(trueAction).T)

		F = np.hstack((Fx, Fy, Fz)).reshape(1, -1)
		M = np.array(M).reshape(1, -1)

		return F, M

# class _linearModel:

# 	def __init__(self, model, xa, ub, dxs, dUs, signMask, order = 3, modelBias = 0, modelType = 'force'):
# 		if modelType.lower() not in ['force', 'moment']:
# 			raise ValueError(f'_linearModel: Expected modelType of "force" or "moment" but {modelType} passed')
# 		xin = model.droneGetModelInput(xa, ub)
# 		if modelType.lower() == 'force':
# 			self.modelNorm = float(np.array(xin['F_den']).reshape(-1))
# 		elif modelType.lower() == 'moment':
# 			self.modelNorm = float(np.array(xin['M_den']).reshape(-1))
# 		self.bias = modelBias
# 		self.model = model
# 		self.signMask = signMask
# 		self.xa = xa
# 		self.ub = ub
# 		self.Ub = self.mapU(ub)
# 		self.f0 = model.predict(xin).__array__()[0][0]
# 		fx = []
# 		for i, (xi, dxi) in enumerate(zip(xa[0, :], dxs)):
# 			fx.append(derivative(self.wrapx, xi, n = 1, dx = dxi, order = order, args = (i,)))
# 		self.fx = np.array(fx) 
# 		fu = []
# 		for i, (ui, dui) in enumerate(zip(self.Ub[0, :], dUs)):
# 			fu.append(derivative(self.wrapu, ui, n = 1, dx = dui, order = order, args = (i,)))
# 		self.fu = np.array(fu)

# 	def __call__(self, x, u):
# 		return self.predict(x, u)

# 	def predict(self, x, u):
# 		dx = x - self.xa
# 		du = self.mapU(u) - self.Ub
# 		fx = np.nansum((dx.reshape(-1)*self.fx))
# 		fu = np.nansum((du.reshape(-1)*self.fu))
# 		return (self.f0 + fx + fu - self.bias)/self.modelNorm

# 	def wrapx(self, x, idx):
# 		xa = self.xa.copy()
# 		xa[:, idx] = x
# 		xin = self.model.droneGetModelInput(xa, self.ub)
# 		return self.model.predict(xin).__array__()[0][0]

# 	def wrapu(self, y, idx):
# 		Ub = self.ub.copy()
# 		Ub[:, idx] = y
# 		ub = self.unmapU(Ub)
# 		xin = self.model.droneGetModelInput(self.xa, ub)
# 		return self.model.predict(xin).__array__()[0][0]

# 	def mapU(self, u):
# 		controlMoments = np.matmul(self.signMask, u.reshape(4, 1)).reshape(-1)
# 		omegaTotal = np.nansum(u)
# 		U = np.hstack((controlMoments, omegaTotal)).reshape(-1, 4)
# 		return U

# 	def unmapU(self, U):
# 		signMask = np.vstack((self.signMask, np.ones(4)))
# 		u = np.matmul(np.linalg.inv(signMask), U.reshape(4, 1)).reshape(1, 4)
# 		return u