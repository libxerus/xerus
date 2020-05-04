import numpy as np

def easy(xs):
	marginal = lambda x: np.sin(np.pi*x)
	xs = np.asarray(xs)
	assert xs.ndim == 2
	ret = np.prod(marginal(xs), axis=0).reshape(-1, 1)
	assert ret.shape == (xs.shape[1], 1)
	return ret

def hard(xs):
	marginal = lambda x: abs(np.sin(np.pi*x))
	xs = np.asarray(xs)
	assert xs.ndim == 2
	ret = np.prod(marginal(xs), axis=0).reshape(-1, 1)
	assert ret.shape == (xs.shape[1], 1)
	return ret
