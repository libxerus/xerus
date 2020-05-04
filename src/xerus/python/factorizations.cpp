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
 * @brief Definition of the python bindings of tensor factorizations.
 */


#include "misc.h"
using namespace internal;

void expose_factorizations(module& m) {
	class_<TensorFactorisation>(m,"TensorFactorisation")
		.def("__rlshift__", +[](TensorFactorisation &_rhs, std::vector<IndexedTensor<Tensor>*> &_lhs){
			_rhs(_lhs);
		})
	;
	class_<SVD, TensorFactorisation>(m,"SVD_temporary")
		.def_readonly("input", &SVD::input)
		.def_readonly("maxRank", &SVD::maxRank)
		.def_readonly("epsilon", &SVD::epsilon)
	;
	m.def("SVD", +[](IndexedTensor<Tensor> &_rhs, size_t _maxRank, double _eps)->TensorFactorisation*{
		return new SVD(std::move(_rhs), _maxRank, _eps);
	}, keep_alive<0,1>(), return_value_policy::take_ownership, // result is treated as a new object
															   // but the argument will not be destroyed before the result is destroyed
		arg("source"), arg("maxRank")=std::numeric_limits<size_t>::max(), arg("eps")=EPSILON
	);

	class_<QR, TensorFactorisation>(m,"QR_temporary").def_readonly("input", &QR::input);
	m.def("QR", +[](IndexedTensor<Tensor>& _rhs)->TensorFactorisation*{
		return new QR(std::move(_rhs));
	}, keep_alive<0,1>(), return_value_policy::take_ownership // result is treated as a new object
	);                                        // but the argument will not be destroyed before the result is destroyed

	class_<RQ, TensorFactorisation>(m,"RQ_temporary").def_readonly("input", &RQ::input);
	m.def("RQ", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new RQ(std::move(_rhs));
	}, keep_alive<0,1>(), return_value_policy::take_ownership // result is treated as a new object
	);	 	 	 	 	 	 	 	 	 	  // but the argument will not be destroyed before the result is destroyed

	class_<QC, TensorFactorisation>(m,"QC_temporary").def_readonly("input", &QC::input);
	m.def("QC", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new QC(std::move(_rhs));
	}, keep_alive<0,1>(), return_value_policy::take_ownership // result is treated as a new object
	);																				// but the argument will not be destroyed before the result is destroyed

	class_<CQ, TensorFactorisation>(m,"CQ_temporary").def_readonly("input", &CQ::input);
	m.def("CQ", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new CQ(std::move(_rhs));
	}, keep_alive<0,1>(), return_value_policy::take_ownership // result is treated as a new object
	);	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 // but the argument will not be destroyed before the result is destroyed
}
