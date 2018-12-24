// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf.
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


#define NO_IMPORT_ARRAY
#include "misc.h"

using namespace internal;

void expose_factorizations() {
	class_<TensorFactorisation, boost::noncopyable>("TensorFactorisation", boost::python::no_init)
		.def("__rlshift__", +[](TensorFactorisation &_rhs, object &_lhs){
			std::vector<IndexedTensor<Tensor>*> tmp = extract<std::vector<IndexedTensor<Tensor>*>>(_lhs);
			_rhs(tmp);
		})
	;
	class_<SVD, bases<TensorFactorisation>, boost::noncopyable>("SVD_temporary", boost::python::no_init);
	def("SVD", +[](IndexedTensor<Tensor> &_rhs, size_t _maxRank, double _eps)->TensorFactorisation*{
		return new SVD(std::move(_rhs), _maxRank, _eps);
	}, return_value_policy<manage_new_object,  // result is treated as a new object
			with_custodian_and_ward_postcall<0,1>>(), // but the argument will not be destroyed before the result is destroyed
		(arg("source"), arg("maxRank")=std::numeric_limits<size_t>::max(), arg("eps")=EPSILON)
	);


	class_<QR, bases<TensorFactorisation>, boost::noncopyable>("QR_temporary", boost::python::no_init);
	def("QR", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new QR(std::move(_rhs));
	}, return_value_policy<manage_new_object,  // result is treated as a new object
	   with_custodian_and_ward_postcall<0,1>>()); // but the argument will not be destroyed before the result is destroyed

	class_<RQ, bases<TensorFactorisation>, boost::noncopyable>("RQ_temporary", boost::python::no_init);
	def("RQ", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new RQ(std::move(_rhs));
	}, return_value_policy<manage_new_object,  // result is treated as a new object
	   with_custodian_and_ward_postcall<0,1>>()); // but the argument will not be destroyed before the result is destroyed

	class_<QC, bases<TensorFactorisation>, boost::noncopyable>("QC_temporary", boost::python::no_init);
	def("QC", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new QC(std::move(_rhs));
	}, return_value_policy<manage_new_object,  // result is treated as a new object
	   with_custodian_and_ward_postcall<0,1>>()); // but the argument will not be destroyed before the result is destroyed

	class_<CQ, bases<TensorFactorisation>, boost::noncopyable>("CQ_temporary", boost::python::no_init);
	def("CQ", +[](IndexedTensor<Tensor> &_rhs)->TensorFactorisation*{
		return new CQ(std::move(_rhs));
	}, return_value_policy<manage_new_object,  // result is treated as a new object
	   with_custodian_and_ward_postcall<0,1>>()); // but the argument will not be destroyed before the result is destroyed

}
