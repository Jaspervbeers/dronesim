from numpy import zeros as np_zeros
from numpy import sum as np_sum
from numpy import array as np_array
from numpy import matmul as np_matmul


class PID:
    def __init__(self, P, I, D, refreshIntegrator = False):
        self.P = P
        self.I = I
        self.D = D
        self.refreshIntegrator = refreshIntegrator
        # self._integralErrorMemorySeconds = 1
        self._integralErrorMemorySeconds = 0.1
        # self._integralErrorMemory = 10
        self._integralUpperLim = 1
        self._integralLowerLim = -1
        self.Dsmoothing = 3
        self.outDim, self.inDim = P.shape
        self.error = np_zeros((1, P.shape[1]))
        self.I_error = np_zeros((1, I.shape[1]))
        self.d_error = np_zeros((1, D.shape[1]))
        self.PIDHistory = {'P':[], 'I':[], 'D':[]}
        self.ErrorHistory = {'P':[], 'I':[], 'D':[]}

    # def _integrateError(self, error):
    #     if len(self.ErrorHistory['I']) > self._integralErrorMemory and self.refreshIntegrator:
    #         self.I_error = np_sum(np_array(self.ErrorHistory['I']).reshape(-1, self.I.shape[1])[-self._integralErrorMemory:, :], axis = 0).reshape(1, self.I.shape[1]) + (error)*self.dt
    #         # import code
    #         # code.interact(local=locals())
    #     else:
    #         self.I_error = np_sum(np_array(self.ErrorHistory['I']).reshape(-1, self.I.shape[1]), axis = 0).reshape(1, self.I.shape[1]) + (error)*self.dt    
    
    def _integrateError(self, error):
        self.I_error += (error)*self.dt
        self.I_error[self.I_error < -1] = -1
        self.I_error[self.I_error > 1] = 1 
    

    def _differentiateError(self, error):
        self.d_error = (error - self.error)/self.dt


    def control(self, error, dt, **kwargs):
        self.dt = dt
        self._integralErrorMemory = int(self._integralErrorMemorySeconds/dt)
        error = np_array(error).reshape(1, self.inDim)
        self._differentiateError(error)
        self._integrateError(error)
        u = np_matmul(self.P, error.T) + np_matmul(self.I, self.I_error.T) + np_matmul(self.D, self.d_error.T)
        self.ErrorHistory['P'].append(error)
        # self.ErrorHistory['I'].append(self.I_error)
        self.ErrorHistory['I'].append(error*self.dt)
        self.ErrorHistory['D'].append(self.d_error)
        self.error = error
        return u.T
    
    def forceRefresh(self):
        '''
        Force all errors to zero. 
        '''
        self.d_error = 0
        self.I_error = 0
        self.PIDHistory = {'P':[], 'I':[], 'D':[]}
        self.ErrorHistory = {'P':[], 'I':[], 'D':[]}