from noiseDisturbance import sensorNoise

class droneNoiseBlock(sensorNoise.noiseBlock):

    def __init__(self, noiseType = 'normal', noiseScale_state = 0.1, noiseScale_input = 0, globalSeed = 0):
        stateIndexMapping = {
            'roll':0,
            'pitch':1,
            'yaw':2,
            'u':3,
            'v':4,
            'w':5,
            'p':6,
            'q':7,
            'r':8,
            'x':9,
            'y':10,
            'z':11
            }
        super().__init__(stateIndexMapping, noiseType=noiseType, noiseScale_state=noiseScale_state, noiseScale_input=noiseScale_input, globalSeed=globalSeed)