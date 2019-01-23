import unittest
import xerus as xe
import numpy as np
from numpy.polynomial.legendre import legval

class TestReconstruction(unittest.TestCase):
    def test_small_reconstruction_weighted(self):
        # the function to approximate
        def fnc(x, y):
            return np.sin(2*np.pi*x)*(y[0] + 0.1*y[1]**2) + np.cos(2*np.pi*x)*y[1]

        x_dim = 100
        y_dim = 2
        n_samples = 10000
        n_test_samples = 100
        deg = 2
        basis = xe.PolynomBasis.Legendre

        x = np.linspace(0, 1, x_dim)
        def discretized_fnc(y):
            return fnc(x, y)

        nodes = 2*np.random.rand(n_samples, y_dim)-1
        values = [xe.Tensor.from_ndarray(discretized_fnc(y)) for y in nodes]
        vector = lambda x: xe.Tensor.from_ndarray(legval(x, np.eye(deg+1)))
        measurements = [[vector(ni) for ni in node] for node in nodes]

        dimension = [x_dim] + [deg+1]*y_dim
        reco = xe.uq_ra_adf(measurements, values, dimension, targeteps=1e-8, maxitr=70)

        #TODO: implement a xerus function: tt_evaluate(tt, pos, pos2meas) where pos2meas is a function pos2meas(int mode, int idx, pos) that calculates the idx-th basis function in the given mode
        #TODO: implement a xerus function: measurements(pos_vector, pos2meas) 

        test_nodes = 2*np.random.rand(n_test_samples, y_dim)-1
        error = 0
        for y in test_nodes:
            res = xe.uq_tt_evaluate(reco, y, basis).to_ndarray()
            ref = discretized_fnc(y)
            error += np.linalg.norm(res - ref) / np.linalg.norm(ref)
        error /= n_test_samples

        self.assertLessEqual(error, 1e-3)

    def test_small_reconstruction(self):
        # the function to approximate
        def fnc(x, y):
            return np.sin(2*np.pi*x)*(y[0] + 0.1*y[1]**2) + np.cos(2*np.pi*x)*y[1]

        x_dim = 100
        y_dim = 2
        n_samples = 10000
        n_test_samples = 100

        x = np.linspace(0, 1, x_dim)
        def discretized_fnc(y):
            return fnc(x, y)

        nodes = 2*np.random.rand(n_samples, y_dim)-1

        measurements = xe.UQMeasurementSet()
        for y in nodes:
            u = discretized_fnc(y)
            measurements.add(y, xe.Tensor.from_ndarray(u))

        basis = xe.PolynomBasis.Legendre
        dimension = [x_dim] + [3]*y_dim
        reco = xe.uq_ra_adf(measurements, basis, dimension, targeteps=1e-8, maxitr=70)

        test_nodes = 2*np.random.rand(n_test_samples, y_dim)-1
        error = 0
        for y in test_nodes:
            res = xe.uq_tt_evaluate(reco, y, basis).to_ndarray()
            ref = discretized_fnc(y)
            error += np.linalg.norm(res - ref) / np.linalg.norm(ref)
        error /= n_test_samples

        self.assertLessEqual(error, 1e-3)

    # def test_large_reconstruction(self):
    #     data = np.load('samples.npz')
    #     measurements = xe.UQMeasurementSet()
    #     for y,u in zip(data['nodes'], data['values']):
    #         measurements.add(y, xe.Tensor.from_ndarray(u))
    #     basis = xe.PolynomBasis.Legendre
    #     dimension = [data['values'].shape[1]] + [8]*data['nodes'].shape[1]
    #     reco = xe.uq_ra_adf(measurements, basis, dimension, targeteps=1e-8, maxitr=1000)
    #     #TODO: just assert that the residuum is below 1e-2
    #     ref = xe.load("reconstruction.xrs")
    #     self.assertLessEqual(xe.frob_norm(reco - ref), 1e-8)

if __name__ == '__main__':
    unittest.main()