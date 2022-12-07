import numpy as np
from utils import *
from scipy.stats import norm, uniform

#This is based off the work from: https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_johnhw_pfilter_blob_master_pfilter_pfilter.py&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=ZPYORsBKtIuaYhAHxrmu_A&m=_Jmx087vh8DER0PdtiQapaEcQ0lzRpXRBz8VnoQJsgo&s=74JF1gZ25dmV8adiAO5-_a6T1WyDyQWI7a4e3rmDAzA&e= 
#Main modifications are:
    #Observations no longer need to be a single dimension vector, can now be matrices (or larger datatypes)
    #Added probability to prior sampling
    #Update step now scales weights rather than previously setting by the current observation.

## Resampling based on the examples at: https://urldefense.proofpoint.com/v2/url?u=https-3A__github.com_rlabbe_Kalman-2Dand-2DBayesian-2DFilters-2Din-2DPython_blob_master_12-2DParticle-2DFilters.ipynb&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=ZPYORsBKtIuaYhAHxrmu_A&m=_Jmx087vh8DER0PdtiQapaEcQ0lzRpXRBz8VnoQJsgo&s=QqAU1yIr4654910_Q4Cz_ToQLdFFCdKTEPOipyKcPWw&e= 
## originally by Roger Labbe, under an MIT License
def systematic_resample(weights):
    n = len(weights)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    return create_indices(positions, weights)


def stratified_resample(weights):
    n = len(weights)
    positions = (np.random.uniform(0, 1, n) + np.arange(n)) / n
    return create_indices(positions, weights)



def residual_resample(weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    # take int(N*w) copies of each weight
    num_copies = (n * weights).astype(np.uint32)
    k = 0
    for i in range(n):
        for _ in range(num_copies[i]):  # make n copies
            indices[k] = i
            k += 1
    # use multinormial resample on the residual to fill up the rest.
    residual = weights - num_copies  # get fractional part
    residual /= np.sum(residual)
    cumsum = np.cumsum(residual)
    cumsum[-1] = 1
    indices[k:n] = np.searchsorted(cumsum, np.random.uniform(0, 1, n - k))
    return indices


def create_indices(positions, weights):
    n = len(weights)
    indices = np.zeros(n, np.uint32)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return indices


### end rlabbe's resampling functions


def multinomial_resample(weights):
    return np.random.choice(np.arange(len(weights)), p=weights, size=len(weights))


# resample function from https://urldefense.proofpoint.com/v2/url?u=http-3A__scipy-2Dcookbook.readthedocs.io_items_ParticleFilter.html&d=DwIGAg&c=-35OiAkTchMrZOngvJPOeA&r=ZPYORsBKtIuaYhAHxrmu_A&m=_Jmx087vh8DER0PdtiQapaEcQ0lzRpXRBz8VnoQJsgo&s=hox5tIpwCKX8_hMFh5OgtG0lHIAbQBtr-CHH8EfQUdY&e= 
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.0] + [np.sum(weights[: i + 1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0 + i) / n for i in range(n)]:
        while u > C[j]:
            j += 1
        indices.append(j - 1)
    return indices

def squared_error(x, y, sigma=1, **kwargs):
    """
        RBF kernel, supporting masked values in the observation
        Parameters:
        -----------
        x : array (N,D) array of values
        y : array (N,D) array of values
        Returns:
        -------
        distance : scalar
            Total similarity, using equation:
                d(x,y) = e^((-1 * (x - y) ** 2) / (2 * sigma ** 2))
            summed over all samples. Supports masked arrays.
    """
    dx = (x - y) ** 2
    d = np.ma.sum(dx, axis=1)
    return np.exp(-d / (2.0 * sigma ** 2))


def gaussian_noise(x, sigmas, **kwargs):
    """Apply diagonal covaraiance normally-distributed noise to the N,D array x.
    Parameters:
    -----------
        x : array
            (N,D) array of values
        sigmas : array
            D-element vector of std. dev. for each column of x
    """
    n = np.random.normal(np.zeros(len(sigmas)), sigmas, size=(x.shape[0], len(sigmas)))
    
    #TODO: Make this more efficient. Right now this is crazy crazy slow...
    prob = np.ones((x.shape[0]))
    #for i, sample_list in enumerate(n):
    #    for j, sample in enumerate(sample_list):
    #        prob[i] *= 1.0/(sigmas[j]*np.sqrt(2*np.pi))*np.exp(-0.5*(sample/sigmas[j])**2)
            
    return x + n, prob



def prior_fn(n,scale_init_sigma,sigma_t,sigma_r):
    x = np.zeros((n, 6))
    p = np.zeros((n))
    
    for i, particle in enumerate(x):
        # Sample from gaussian to initialize particle
        particle = np.array([norm.rvs(loc=0, scale=scale_init_sigma*sigma_t),
                             norm.rvs(loc=0, scale=scale_init_sigma*sigma_t),
                             norm.rvs(loc=0, scale=scale_init_sigma*sigma_t),
                             norm.rvs(loc=0, scale=scale_init_sigma*sigma_r),
                             norm.rvs(loc=0, scale=scale_init_sigma*sigma_r),
                             norm.rvs(loc=0, scale=scale_init_sigma*sigma_r),
                             ])
        # Get probability of sample
        p[i]  = norm.pdf(particle[0], 0, scale_init_sigma*sigma_t)
        p[i] *= norm.pdf(particle[1], 0, scale_init_sigma*sigma_t)
        p[i] *= norm.pdf(particle[2], 0, scale_init_sigma*sigma_t)
        p[i] *= norm.pdf(particle[3], 0, scale_init_sigma*sigma_r)
        p[i] *= norm.pdf(particle[4], 0, scale_init_sigma*sigma_r)
        p[i] *= norm.pdf(particle[5], 0, scale_init_sigma*sigma_r)
        
    return x, p


def identity(x, **kwargs):
    return x
                   

class ParticleFilter(object):
    """A particle filter object which maintains the internal state of a population of particles, and can
    be updated given observations.
    
    Attributes:
    -----------
    
    n_particles : int
        number of particles used (N)
    d : int
        dimension of the internal state
    resample_proportion : float
        fraction of particles resampled from prior at each step
    particles : array
        (N,D) array of particle states
    original_particles : array
        (N,D) array of particle states *before* any random resampling replenishment
        This should be used for any computation on the previous time step (e.g. computing
        expected values, etc.)
    mean_hypothesis : array 
        The current mean hypothesized observation
    mean_state : array
        The current mean hypothesized internal state D
    map_hypothesis: 
        The current most likely hypothesized observation
    map_state:
        The current most likely hypothesized state
    n_eff:
        Normalized effective sample size, in range 0.0 -> 1.0
    weight_entropy:
        Entropy of the weight distribution (in nats)
    hypotheses : array
        The (N,...) array of hypotheses for each particle
    weights : array
        N-element vector of normalized weights for each particle.
    """

    def __init__(
        self,
        prior_fn,
        observe_fn=None,
        resample_fn=None,
        n_particles=200,
        dynamics_fn=None,
        noise_fn=None,
        weight_fn=None,
        resample_proportion=None,
        column_names=None,
        internal_weight_fn=None,
        transform_fn=None,
        n_eff_threshold=1.0,
    ):
        """
        
        Parameters:
        -----------
        
        prior_fn : function(n) = > states, prob
                a function that generates N samples from the prior over internal states as
                an (N,D) particle array with corresponding N probabilities
        observe_fn : function(states) => observations
                    transformation function from the internal state to the sensor state. Takes an (N,D) array of states 
                    and returns the expected sensor output as an array (e.g. a (N,W,H) tensor if generating W,H dimension images).
        resample_fn: A resampling function weights (N,) => indices (N,), afterwards it is assumed all weights are 1/N
        n_particles : int 
                     number of particles in the filter
        dynamics_fn : function(states) => states
                      dynamics function, which takes an (N,D) state array and returns a new one with the dynamics applied.
        noise_fn : function(states) => states, prob
                    noise function, takes a state vector and returns a new one with noise added and the prob of that occuring
        weight_fn :  function(hypothesized, real) => weights
                    computes the distance from the real sensed variable and that returned by observe_fn. Takes
                    a an array of N hypothesised sensor outputs (e.g. array of dimension (N,W,H)) and the observed output (e.g. array of dimension (W,H)) and 
                    returns a strictly positive weight for the each hypothesis as an N-element vector. 
                    This should be a *similarity* measure, with higher values meaning more similar, for example from an RBF kernel.
        internal_weight_fn :  function(states, observed) => weights
                    Reweights the particles based on their *internal* state. This is function which takes
                    an (N,D) array of internal states and the observation and 
                    returns a strictly positive weight for the each state as an N-element vector. 
                    Typically used to force particles inside of bounds, etc.       
        transform_fn: function(states, weights) => transformed_states
                    Applied at the very end of the update step, if specified. Updates the attribute
                    `transformed_particles`. Useful when the particle state needs to be projected
                    into a different space.
        resample_proportion : float
                    proportion of samples to draw from the initial on each iteration.
        n_eff_threshold=1.0: float
                    effective sample size at which resampling will be performed (0.0->1.0). Values
                    <1.0 will allow samples to propagate without the resampling step until
                    the effective sample size (n_eff) drops below the specified threshold.
        column_names : list of strings
                    names of each the columns of the state vector
        
        """
        self.resample_fn = resample_fn or resample
        self.column_names = column_names
        self.prior_fn = prior_fn
        self.n_particles = n_particles
        self.init_filter()
        self.n_eff_threshold = n_eff_threshold
        self.d = self.particles.shape[1]
        self.observe_fn = observe_fn or identity
        self.dynamics_fn = dynamics_fn or identity
        self.noise_fn = noise_fn or identity
        self.weight_fn = weight_fn or squared_error
        self.transform_fn = transform_fn
        self.transformed_particles = None
        self.resample_proportion = resample_proportion or 0.0
        self.internal_weight_fn = internal_weight_fn
        self.original_particles = np.array(self.particles)
        
        #FLORIAN ADDED THIS
        self.weights = (1.0/self.n_particles)*np.ones((self.n_particles,))

        #self.mean_state = np.zeros(6)

    def init_filter(self, mask=None):
        """Initialise the filter by drawing samples from the prior.
        
        Parameters:
        -----------
        mask : array, optional
            boolean mask specifying the elements of the particle array to draw from the prior. None (default)
            implies all particles will be resampled (i.e. a complete reset)
        """
        #new_sample, prob = self.prior_fn(self.n_particles)
        new_sample, _ = self.prior_fn(self.n_particles)
        
        # resample from the prior
        if mask is None:
            self.particles = new_sample
            #self.weights   = prob
        else:
            self.particles[mask, :] = new_sample[mask, :]
            #self.weights[mask]   = prob[mask]
        
        #self.weight_normalisation = np.sum(self.weights)        
        #print(self.weight_normalisation)
        #self.weights = self.weights / self.weight_normalisation
        #print("-------------------------------")
        #print(self.weights)
    
    def update(self, observed=None, **kwargs):
        """Update the state of the particle filter given an observation.
        
        Parameters:
        ----------
        
        observed: array
            The observed output, in the same format as observe_fn() will produce. This is typically the
            input from the sensor observing the process (e.g. a camera image in optical tracking).
            If None, then the observation step is skipped, and the filter will run one step in prediction-only mode.
        kwargs: any keyword arguments specified will be passed on to:
            observe_fn(y, **kwargs)
            weight_fn(x, **kwargs)
            dynamics_fn(x, **kwargs)
            noise_fn(x, **kwargs)
            internal_weight_function(x, y, **kwargs)
            transform_fn(x, **kwargs)
        """

        #FLORIAN ADDED THIS
        #self.weight_normalisation = np.sum(self.weights)
        #self.weights = self.weights / self.weight_normalisation

        # resample according to weights of the particles (prior)
        #indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
        #self.particles = self.particles[indices]
        # apply dynamics and noise
        self.particles, _ = self.noise_fn( self.dynamics_fn(self.particles, **kwargs), **kwargs)
        

        # hypothesise observations
        self.hypotheses = self.observe_fn(self.particles, **kwargs)
        #print("---------------")
        #print(self.hypotheses)
        #print(self.weights)

        if observed is not None:
            # compute similarity to observations
            # force to be positive

            weights = np.clip(
                                self.weights * np.array(self.weight_fn(self.hypotheses,observed, **kwargs)),
                                0, np.inf,
                             )
        else:
            # we have no observation, so all particles weighted the same
            weights = self.weights * np.ones((self.n_particles,))

        #FLORIAN ADDED THIS
        #weights *= self.weights
        
        # apply weighting based on the internal state
        # most filters don't use this, but can be a useful way of combining
        # forward and inverse models
        if self.internal_weight_fn is not None:
            internal_weights = self.internal_weight_fn(
                self.particles, observed, **kwargs
            )
            internal_weights = np.clip(internal_weights, 0, np.inf)
            internal_weights = internal_weights / np.sum(internal_weights)
            weights *= internal_weights
        
        # normalise weights to resampling probabilities
        self.weight_normalisation = np.sum(weights)
        print(self.weight_normalisation)
        
        #FLORIAN ADDED THIS
        if self.weight_normalisation == 0:
            return
        
        self.weights = weights / self.weight_normalisation

        # Compute effective sample size and entropy of weighting vector.
        # These are useful statistics for adaptive particle filtering.
        self.n_eff = (1.0 / np.sum(self.weights ** 2)) / self.n_particles
        self.weight_entropy = np.sum(self.weights * np.log(self.weights))

        # preserve current sample set before any replenishment
        self.original_particles = np.array(self.particles)

        # store mean (expected) hypothesis
        self.mean_hypothesis = np.sum(self.hypotheses.T * self.weights, axis=-1).T
        self.mean_state = np.sum(self.particles.T * self.weights, axis=-1).T
        self.cov_state = np.cov(self.particles, rowvar=False, aweights=self.weights)

        # store MAP estimate
        argmax_weight = np.argmax(self.weights)
        self.map_state = self.particles[argmax_weight]
        self.map_hypothesis = self.hypotheses[argmax_weight]

        # apply any post-processing
        if self.transform_fn:
            self.transformed_particles = self.transform_fn(
                self.original_particles, self.weights, **kwargs
            )
        else:
            self.transformed_particles = self.original_particles
            

        # resampling (systematic resampling) step
        if self.n_eff < self.n_eff_threshold:
            indices = self.resample_fn(self.weights)
            self.particles = self.particles[indices, :]
            self.weights = (1.0/self.n_particles)*np.ones((self.n_particles,))

        # randomly resample some particles from the prior
        if self.resample_proportion > 0:
            random_mask = (
                np.random.random(size=(self.n_particles,)) < self.resample_proportion
            )
            self.resampled_particles = random_mask
            self.init_filter(mask=random_mask)
            
