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
 * @brief Definition of the Tensor python bindings.
 */


#include "misc.h"

void expose_indexedTensors(module& m) {
	// --------------------------------------------------------------- index
	class_<Index>(m,"Index",
		"helper class to define objects to be used in indexed expressions"
	)
		.def(init())
		.def(init<int64_t>())
		.def("__pow__",     &Index::operator^, "i**d changes the index i to span d indices in the current expression")
		.def("__xor__",     &Index::operator^, "i^d changes the index i to span d indices in the current expression")
		.def("__truediv__", &Index::operator/, "i/n changes the index i to span 1/n of all the indices of the current object")
		.def("__and__",     &Index::operator&, "i&d changes the index i to span all but d indices of the current object")
		.def("__repr__", static_cast<std::string (*)(const Index &)>(&misc::to_string<Index>))
	;
	implicitly_convertible<int64_t, Index>();
	m.def("indices", [](const size_t n) -> std::vector<Index> { return std::vector<Index>(n); });

	// NOTE In the follwoing all magic methods are defined only for the ReadOnly indexed Tensors.
	//      Since Writable, Movable and the generic IndexedTensor are subclasses they inherit the methods and since the
	//      first argument (except self) is always ReadOnly these methods work for arbitrary combinations.
	//      pybind11 will take care of the proper matching.
	using namespace internal;
#define ADD_MOVE_AND_RESULT_PTR(name, op, lhs_type, rhs_type, res_type) \
	.def(name, \
			+[](lhs_type &_l, rhs_type &_r) -> res_type* { \
				LOG(pydebug, "python wrapper: " name "(" #lhs_type ", " #rhs_type ")");\
				return new res_type(std::move(_l) op std::move(_r)); \
			}, keep_alive<0, 1>(), keep_alive<0, 2>(), return_value_policy::take_ownership)

	class_<internal::IndexedTensorReadOnly<TensorNetwork>>(m,"IndexedTensorNetworkReadOnly")
		ADD_MOVE_AND_RESULT_PTR("__add__",      +, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__add__",      +, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__sub__",      -, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__sub__",      -, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",      *, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",      *, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",      *, IndexedTensorReadOnly<TensorNetwork>, value_t,                              IndexedTensorReadOnly<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__rmul__",     *, IndexedTensorReadOnly<TensorNetwork>, value_t,                              IndexedTensorReadOnly<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__truediv__",  /, IndexedTensorReadOnly<TensorNetwork>, value_t,                              IndexedTensorReadOnly<TensorNetwork>)
		.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<TensorNetwork> &)>(&frob_norm<TensorNetwork>))
		.def("__float__", [](const IndexedTensorReadOnly<TensorNetwork> &_self){ return value_t(_self); })
	;
	class_<internal::IndexedTensorWritable<TensorNetwork>, internal::IndexedTensorReadOnly<TensorNetwork>>(m,"IndexedTensorNetworkWriteable");
	class_<internal::IndexedTensorMoveable<TensorNetwork>, internal::IndexedTensorWritable<TensorNetwork>>(m,"IndexedTensorNetworkMoveable");
	class_<internal::IndexedTensor<TensorNetwork>,         internal::IndexedTensorWritable<TensorNetwork>>(m,"IndexedTensorNetwork")
		.def("__lshift__",
			+[](internal::IndexedTensor<TensorNetwork> &_lhs, internal::IndexedTensorReadOnly<Tensor> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
		.def("__lshift__",
			+[](internal::IndexedTensor<TensorNetwork> &_lhs, internal::IndexedTensorReadOnly<TensorNetwork> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
	;

	class_<internal::IndexedTensorReadOnly<Tensor>>(m,"IndexedTensorReadOnly")
		ADD_MOVE_AND_RESULT_PTR("__add__",  +, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__sub__",  -, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",  *, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",  *, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__",  *, IndexedTensorReadOnly<Tensor>, value_t,                              IndexedTensorReadOnly<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__rmul__", *, IndexedTensorReadOnly<Tensor>, value_t,                              IndexedTensorReadOnly<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__truediv__",  /, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>,        IndexedTensorMoveable<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__truediv__",  /, IndexedTensorReadOnly<Tensor>, value_t,                              IndexedTensorMoveable<Tensor>)
		.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<Tensor> &)>(&frob_norm<Tensor>))
		.def("__float__", [](const IndexedTensorReadOnly<Tensor> &_self){ return value_t(_self); })
	;
	class_<internal::IndexedTensorWritable<Tensor>, internal::IndexedTensorReadOnly<Tensor>>(m,"IndexedTensorWriteable");
	class_<internal::IndexedTensorMoveable<Tensor>, internal::IndexedTensorWritable<Tensor>>(m,"IndexedTensorMoveable");
	class_<internal::IndexedTensor<Tensor>, internal::IndexedTensorWritable<Tensor>>(m,"IndexedTensor")
		.def("__lshift__",
			+[](internal::IndexedTensor<Tensor> &_lhs, internal::IndexedTensorReadOnly<Tensor> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
		.def("__lshift__",
			+[](internal::IndexedTensor<Tensor> &_lhs, internal::IndexedTensorReadOnly<TensorNetwork> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
		.def_readonly("indices", &internal::IndexedTensor<Tensor>::indices)
	;

	implicitly_convertible<internal::IndexedTensorReadOnly<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensorWritable<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensorMoveable<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensor<Tensor>,         internal::IndexedTensorMoveable<TensorNetwork>>();
}
