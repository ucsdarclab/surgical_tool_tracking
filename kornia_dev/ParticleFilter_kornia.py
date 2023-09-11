import numpy as np
import warnings
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
        
class ParticleFilter:
    def __init__(self, num_states, initialDistributionFunc, motionModelFunc, obsModelFunc, 
                 num_particles=200, min_num_effective_particles=0.5):
        
        self._particles = np.zeros((num_particles, num_states))
        self._weights   = np.ones((num_particles))/float(num_particles)
        self._min_num_effective_particles = min_num_effective_particles
        
        self._initialDistributionFunc = initialDistributionFunc
        self._motionModelFunc = motionModelFunc
        self._obsModelFunc = obsModelFunc
    
    def normalizeWeights(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                self._weights = self._weights/np.sum(self._weights)
            except Warning as e:
                self._weights = np.ones(self._particles.shape[0])/float(self._particles.shape[0])

    @timeit
    def initializeFilter(self, **kwargs):
        # Initialize particles here, the kwargs is passed to initialDistributionFunc
        for p_idx, _ in enumerate(self._particles):
            self._particles[p_idx], _ = self._initialDistributionFunc(**kwargs)
    
    @timeit
    def predictionStep(self, **kwargs):
        # Predict particles here, the kwargs is passed to motionModelFunc
        for p_idx, particle in enumerate(self._particles):
            self._particles[p_idx], _ = self._motionModelFunc(particle, **kwargs)

    @timeit   
    def updateStep(self, args):
        for obsModelFunc_idx in range(len(self._obsModelFunc)):
            obsModelFunc = self._obsModelFunc[obsModelFunc_idx]
            obsModelParams = args[obsModelFunc_idx]
            #print('obsModelParams: {}'.format(obsModelParams))
            # Update particle weights here, the kwargs is passed to obsModelFunc
            for p_idx, particle in enumerate(self._particles):
                obs_prob = obsModelFunc(particle, **obsModelParams)
                self._weights[p_idx] = self._weights[p_idx]*obs_prob
            
        self.normalizeWeights()
        
        # Resample if particles are depleted
        if self._min_num_effective_particles > 1./np.sum(self._weights**2):
            self._particles = self._particles[self.resample(), :] 
            self._weights   = np.ones((num_particles))/float(num_particles)
        
    
    def resample(self):
        n = len(self._weights)
        indices = []
        C = [0.0] + [np.sum(self._weightsweights[: i + 1]) for i in range(n)]
        u0, j = np.random.random(), 0
        for u in [(u0 + i) / n for i in range(n)]:
            while u > C[j]:
                j += 1
            indices.append(j - 1)
        return indices
    
    def getMeanParticle(self):
        self.normalizeWeights()
        return np.dot(self._weights, self._particles)