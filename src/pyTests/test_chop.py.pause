import unittest
import numpy as np
import xerus as xe


def generate_random_tttensors(num_tests, max_order=4, max_dimension=30, max_rank=30, random=None):
	orders = random.randint(low=1, high=max_order+1, size=num_tests)
	dimensions = [random.randint(low=1, high=max_dimension, size=order) for order in orders]
	ranks = [random.randint(low=1, high=max_rank, size=(order-1)) for order in orders]
	for dim, rk in zip(dimensions, ranks):
		yield xe.TTTensor.random(dim.tolist(), rk.tolist())


def single_test(A):
	mirrored = lambda pos: A.degree()-1-pos
	name_d = "-".join(map(str, A.dimensions))
	name_r = "-".join(map(str, A.ranks()))
	name = "test_chop_{}_{}".format(name_d, name_r)

	def test_chop(self):
		L,l,e,r,R = xe.indices(5)
		for corePosition in range(A.degree()):
			Al,Ar = A.chop(corePosition)
			Ac = A.get_component(corePosition)
			res = xe.TensorNetwork()
			res(L^corePosition,e,R^mirrored(corePosition)) << Al(L&1,l) * Ac(l,e,r) * Ar(r,R&1)
			# norm(A - res)**2 == norm(A)**2 - 2*inner(A,res) + norm(res)**2
			nA = xe.frob_norm(A)
			nres = xe.frob_norm(res)
			inner = xe.Tensor()
			inner() << A(e&0) * res(e&0)
			self.assertLessEqual(nA**2 - 2*inner[0] + nres**2, 5e-5)

	return name, test_chop


def build_TestChop(seed, num_tests):
	random = np.random.RandomState(seed)
	odir = dict(single_test(t) for t in generate_random_tttensors(num_tests, random=random))
	return type("TestChop", (unittest.TestCase,), odir)
TestChop = build_TestChop(0, 100)
