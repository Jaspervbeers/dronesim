import numpy as np

class noiseBlock:

    def __init__(self, stateMapping, noiseType = 'normal', noiseScale_state = 0.1, noiseScale_input = 0, globalSeed = 0):
        self.noiseType = noiseType.lower()
        self.noiseTypeMapper = {
            'normal':self.addNoise_normal
            }        
        if self.noiseType not in self.noiseTypeMapper.keys():
            raise ValueError(f'Unknown noiseType: "{noiseType}"')
        self.noisify = self.noiseTypeMapper[self.noiseType]
        self.setSeed(globalSeed)
        self.noiseScaleVal_state = noiseScale_state
        self.noiseScaleVal_input = noiseScale_input
        self.stateIndexMapping = stateMapping
        self.noiseScaleVal_state = noiseScale_state
        self.noiseScaleVal_input = noiseScale_input
        self._initStateNoiseScaling()
        return None

    def setSeed(self, seed):
        # Use numpy RandomState to create a RNG object which has consistent outputs for a given root seed
        self.rng = np.random.RandomState(seed)

    def addNoise_normal(self, signal, scale = 0.1, **kwargs):
        # Return signal with added noise
        noiseVec = self.rng.normal(scale = scale, size = signal.shape)
        return signal + noiseVec

    def addBias(self, signal, bias):
        return signal + bias        

    def _setStateNoiseScaling(self, state, scale):
        try:
            stateIdx = int(state)
        except ValueError:
            stateIdx = self.stateIndexMapping[state]
        self.noiseScaling_states.update({stateIdx:scale})

    def _initStateNoiseScaling(self):
        self.noiseScaling_states = {}
        for state in self.stateIndexMapping.keys():
            self._setStateNoiseScaling(state, self.noiseScaleVal_state)

    def mapStateNoiseScales(self, states, scales):
        # Assign noise variances to corresponding states, shape states = shape scales
        if states.shape != scales.shape:
            raise ValueError(f'Shape of states ({states.shape}) misaligned with scales ({scales.shape})')
        for state, scale in zip(states, scales):
            self._setStateNoiseScaling(state, scale)

    def addStateNoise(self, simVars):
        step = simVars['currentTimeStep_index']
        states = simVars['state'][step].copy()
        statesDerivatives = simVars['stateDerivative'][step].copy()
        noisyStates = states.copy()*0
        noisyStateDerivatives = statesDerivatives.copy()*0
        for idx in self.stateIndexMapping.values():
            noisyStates[:, idx] = self.noisify(states[:, idx], self.noiseScaling_states[idx])
            noisyStateDerivatives[:, idx] = self.noisify(statesDerivatives[:, idx], self.noiseScaling_states[idx])
        simVars['state_noisy'][step] = noisyStates
        simVars['stateDerivative_noisy'][step] = noisyStateDerivatives