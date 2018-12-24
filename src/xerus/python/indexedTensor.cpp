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
 * @brief Definition of the Tensor python bindings.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"

void expose_indexedTensors() {
	// --------------------------------------------------------------- index
	class_<Index>("Index",
		"helper class to define objects to be used in indexed expressions"
	)
		.def(init<int64_t>())
		.def("__pow__", &Index::operator^, "i**d changes the index i to span d indices in the current expression")
		.def("__xor__", &Index::operator^, "i^d changes the index i to span d indices in the current expression")
		.def("__div__", &Index::operator/, "i/n changes the index i to span 1/n of all the indices of the current object")
		.def("__and__", &Index::operator&, "i&d changes the index i to span all but d indices of the current object")
		.def("__str__", static_cast<std::string (*)(const Index &)>(&misc::to_string<Index>))
	;
	implicitly_convertible<int64_t, Index>();
	exec(
		"def indices(n=1):\n"
		"  \"\"\"Create n distinct indices.\"\"\"\n"
		"  i = 0\n"
		"  while i<n:\n"
		"    yield Index()\n"
		"    i += 1\n"
	, scope().attr("__dict__"));
	VECTOR_TO_PY(Index, "IndexVector");

	implicitly_convertible<internal::IndexedTensorReadOnly<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensorWritable<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensorMoveable<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	implicitly_convertible<internal::IndexedTensor<Tensor>, internal::IndexedTensorMoveable<TensorNetwork>>();
	// NOTE in the following all __mul__ variants are defined for the ReadOnly indexed Tensors, even if they are meant for
	//		the moveable indexed tensors. boost will take care of the proper matching that way. if IndexedTensorMoveable
	//		defined an __mul__ function on its own it would overwrite all overloaded variants of the readonly indexed tensors
	//		and thus loose a lot of functionality.
	// ---------------------------------------------- indexedTensor<TN>
	using namespace internal;
#define ADD_MOVE_AND_RESULT_PTR(name, op, lhs_type, rhs_type, res_type) \
	.def(name, \
			+[](lhs_type &_l, rhs_type &_r) -> res_type* { \
				LOG(pydebug, "python wrapper: " name);\
				return new res_type(std::move(_l) op std::move(_r)); \
			}, return_value_policy<manage_new_object>())

	class_<internal::IndexedTensorReadOnly<TensorNetwork>, boost::noncopyable>("IndexedTensorNetworkReadOnly", no_init)
		ADD_MOVE_AND_RESULT_PTR("__add__", +, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__add__", +, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__sub__", -, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__sub__", -, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorMoveable<TensorNetwork>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorMoveable<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<TensorNetwork>)
		.def("__mul__",
			+[](internal::IndexedTensorReadOnly<TensorNetwork> &_l, value_t _r) -> internal::IndexedTensorReadOnly<TensorNetwork>* {
				LOG(pydebug, "mul TN ro * scalar");
				return new internal::IndexedTensorMoveable<TensorNetwork>(std::move(_l) * _r);
			}, return_value_policy<manage_new_object>())
		.def("__rmul__",
			+[](value_t _r, internal::IndexedTensorReadOnly<TensorNetwork> &_l) -> internal::IndexedTensorReadOnly<TensorNetwork>* {
				LOG(pydebug, "mul TN scalar * ro");
				return new internal::IndexedTensorMoveable<TensorNetwork>(std::move(_l) * _r);
			}, return_value_policy<manage_new_object>())
		.def("__div__",
			+[](internal::IndexedTensorReadOnly<TensorNetwork> &_l, value_t _r) -> internal::IndexedTensorReadOnly<TensorNetwork>* {
				LOG(pydebug, "div TN ro / scalar");
				return new internal::IndexedTensorMoveable<TensorNetwork>(std::move(_l) / _r);
			}, return_value_policy<manage_new_object>())
		.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<TensorNetwork> &)>(&frob_norm<TensorNetwork>))
		.def(float_(self)) // cast to double
	;

	class_<internal::IndexedTensorWritable<TensorNetwork>, boost::noncopyable, bases<internal::IndexedTensorReadOnly<TensorNetwork>>>("IndexedTensorNetworkWriteable", no_init)
	;
	class_<internal::IndexedTensorMoveable<TensorNetwork>, boost::noncopyable, bases<internal::IndexedTensorWritable<TensorNetwork>>>("IndexedTensorNetworkMoveable", no_init)
	;
	class_<internal::IndexedTensor<TensorNetwork>, boost::noncopyable, bases<internal::IndexedTensorWritable<TensorNetwork>>>("IndexedTensorNetwork", no_init)
		.def("__lshift__",
			+[](internal::IndexedTensor<TensorNetwork> &_lhs, internal::IndexedTensorReadOnly<Tensor> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
		.def("__lshift__",
			+[](internal::IndexedTensor<TensorNetwork> &_lhs, internal::IndexedTensorReadOnly<TensorNetwork> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
	;

	// --------------------------------------------- indexedTensor<Tensor>

	class_<internal::IndexedTensorReadOnly<Tensor>, boost::noncopyable>("IndexedTensorReadOnly", no_init)
		ADD_MOVE_AND_RESULT_PTR("__add__", +, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__sub__", -, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<Tensor>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__mul__", *, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<TensorNetwork>, IndexedTensorMoveable<TensorNetwork>)
		ADD_MOVE_AND_RESULT_PTR("__div__", /, IndexedTensorReadOnly<Tensor>, IndexedTensorReadOnly<Tensor>, IndexedTensorMoveable<Tensor>)
		.def("__mul__",
			+[](internal::IndexedTensorReadOnly<Tensor> &_l, value_t _r) -> internal::IndexedTensorReadOnly<Tensor>* {
				LOG(pydebug, "mul ro * scalar");
				return new internal::IndexedTensorMoveable<Tensor>(std::move(_l) * _r);
			}, return_value_policy<manage_new_object>())
		.def("__rmul__",
			+[](value_t _r, internal::IndexedTensorReadOnly<Tensor> &_l) -> internal::IndexedTensorReadOnly<Tensor>* {
				LOG(pydebug, "mul scalar * ro");
				return new internal::IndexedTensorMoveable<Tensor>(std::move(_l) * _r);
			}, return_value_policy<manage_new_object>())
		.def("__div__",
			+[](internal::IndexedTensorReadOnly<Tensor> &_l, value_t _r) -> internal::IndexedTensorReadOnly<Tensor>* {
				LOG(pydebug, "div ro / scalar");
				return new internal::IndexedTensorMoveable<Tensor>(std::move(_l) / _r);
			}, return_value_policy<manage_new_object>())
		.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<Tensor> &)>(&frob_norm<Tensor>))
		.def(float_(self)) // cast to double
	;
	class_<internal::IndexedTensorWritable<Tensor>, boost::noncopyable, bases<internal::IndexedTensorReadOnly<Tensor>>>("IndexedTensorWriteable", no_init)
	;
	class_<internal::IndexedTensorMoveable<Tensor>, boost::noncopyable, bases<internal::IndexedTensorWritable<Tensor>>>("IndexedTensorMoveable", no_init)
	;
	class_<internal::IndexedTensor<Tensor>, boost::noncopyable, bases<internal::IndexedTensorWritable<Tensor>>>("IndexedTensor", no_init)
		.def("__lshift__",
			+[](internal::IndexedTensor<Tensor> &_lhs, internal::IndexedTensorReadOnly<Tensor> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
		.def("__lshift__",
			+[](internal::IndexedTensor<Tensor> &_lhs, internal::IndexedTensorReadOnly<TensorNetwork> &_rhs) {
				std::move(_lhs) = std::move(_rhs);
			})
	;
	VECTOR_TO_PY(IndexedTensor<Tensor>*, "IndexedTensorList");
}
