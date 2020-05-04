// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2019 Benjamin Huber and Sebastian Wolf.
//
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org
// or contact us at contact@libXerus.org.

/**
 * @file
 * @brief Definition of the python bindings.
 */

#include "misc.h"


PYBIND11_MODULE(xerus, m) {
	m.doc() = "\
The `xerus` library is a general purpose library for numerical calculations with higher order tensors, Tensor-Train Decompositions / Matrix Product States and general Tensor Networks.\n\
The focus of development was the simple usability and adaptibility to any setting that requires higher order tensors or decompositions thereof.\n\
\n\
The key features include:\n\
* Modern code and concepts incorporating many features of the `C++11` standard.\n\
* Full python bindings with very similar syntax for easy transitions from and to c++.\n\
* Calculation with tensors of arbitrary orders using an intuitive Einstein-like notation `A(i,j) = B(i,k,l) * C(k,j,l);`.\n\
* Full implementation of the Tensor-Train decompositions (MPS) with all neccessary capabilities (including Algorithms like ALS, ADF and CG).\n\
* Lazy evaluation of (multiple) tensor contractions featuring heuristics to automatically find efficient contraction orders.\n\
* Direct integration of the `blas` and `lapack`, as high performance linear algebra backends.\n\
* Fast sparse tensor calculation by usage of the `suiteSparse` sparse matrix capabilities.\n\
* Capabilites to handle arbitrary Tensor Networks.\n\
";

	// xerus version
	{
		std::ostringstream tmp;
		tmp << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_REVISION;
		m.attr("__version__") = tmp.str();
	}
	m.attr("VERSION_MAJOR") = VERSION_MAJOR;
	m.attr("VERSION_MINOR") = VERSION_MINOR;
	m.attr("VERSION_REVISION") = VERSION_REVISION;
	m.attr("VERSION_COMMIT") = VERSION_COMMIT;

	expose_indexedTensors(m);
	expose_factorizations(m);

	expose_tensor(m);
	expose_tensorNetwork(m);
	expose_ttnetwork(m);
	expose_htnetwork(m);

	expose_leastSquaresAlgorithms(m);
	expose_recoveryAlgorithms(m);

	expose_misc(m);
}
