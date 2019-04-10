import sys; sys.path.insert(0, '.')
import numpy as np
import xerus as xe
from scipy.optimize import bisect

from basis import HermitePolynomials
from samplers import Uniform, CMDensity, CMSampler, CMWeights, CartesianProductSampler, test_CMSamples, test_CMWeights, approx_quantiles, constant, gaussian
from measures import BasisMeasure, MeasurementList, IdentityMeasure

# the function to approximate
# from functions import easy as fnc
from functions import hard as fnc


n_samples = 10000
n_test_samples = 10000


def rejection_sampler(density, domain):
	from samplers import interpolate, scan_AffineSampler, RejectionSampler
	nodes = interpolate(density, domain, eps=1e-1)
	sampler_1d = scan_AffineSampler(nodes, density)
	return RejectionSampler(sampler_1d.domain, density, sampler_1d)


def quantiles_for_variance(variance, tol):
	basis = HermitePolynomials(6, mean=0, variance=variance)
	density = gaussian(0, variance)  # the base density for the 1-dimensional sampler
	cm_density = CMDensity(basis, density, check=False)  # checks is_orhtonormal(basis, density)
	q = 5*variance
	q = approx_quantiles(cm_density, tol, (-q,q))
	return q


tol = 1e-3
variance = bisect(lambda v: quantiles_for_variance(v, tol/10)-1-tol, 1e-4, 1, xtol=tol)
q = quantiles_for_variance(variance, tol/10)
# assert q <= 1 and abs(q-1) < tol, q

dom = (-1,1)
basis = HermitePolynomials(6, mean=0, variance=variance)

density_1d = gaussian(0, variance)  # the base density for the 1-dimensional sampler
density = lambda xs: np.prod(density_1d(xs), axis=1)

cm_sampler_1d = CMSampler(basis, density_1d, dom)
test_sampler_1d = rejection_sampler(density_1d, dom)

errors = np.empty((2,8))
for e, sampler_1d in enumerate([cm_sampler_1d, test_sampler_1d]):
	for d in range(1,errors.shape[1]+1):
		print("Sampler:", ["CM", "Ref"][e])
		print("Dimension:", d)
		sampler = CartesianProductSampler([sampler_1d]*d)
		test_sampler = CartesianProductSampler([test_sampler_1d]*d)
		measures = [BasisMeasure(basis)]*d

		nodes = sampler.sample(n_samples)
		print("Sample significance:", test_CMSamples(sampler, nodes))
		wgts = CMWeights(sampler, density, nodes)
		print("Weight error:", test_CMWeights(sampler, density, nodes, wgts))

		ml = MeasurementList(measures)

		tensor = xe.Tensor.from_ndarray
		meas = ml(nodes.T)  # input shape: order, n_samples
		meas = np.moveaxis(meas, 0, 1)                         # redundant with new xe interf.
		meas = [[tensor(cmp_m) for cmp_m in m] for m in meas]  # redundant with new xe interf.
		vals = fnc(nodes.T)  # input shape: order, n_samples --> n_samples, 1 (1 == x_dim)
		vals = [tensor(val) for val in vals]                   # redundant with new xe interf.

		reco = xe.uq_ra_adf(meas, vals, wgts, (1,) + ml.dimensions, targeteps=1e-8, maxitr=30)

		test_nodes = test_sampler.sample(n_test_samples)
		id = np.eye(1)
		assert id.shape == (1,1)
		test_pts = [[id]*n_test_samples] + list(test_nodes.T)  #TODO: überarbeite die Rückgabeshape der sampler nachdem das xe interface umgestellt wurde!

		ml = MeasurementList([IdentityMeasure((1, 1)), *ml.measures])
		test_vals = ml.evaluate(reco, test_pts)
		test_vals = [val.to_ndarray() for val in test_vals]  # redundant with new xe interf.
		ref_vals = fnc(test_nodes.T)

		error = np.linalg.norm(test_vals - ref_vals, axis=1)**2
		error = np.sqrt(np.mean(error))
		print("Error: {:.2e}".format(error))
		errors[e,d-1] = error
np.save("errors."+__file__[5:-3]+".npy", errors)
