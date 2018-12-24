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
 * @brief Definition of the Tensor Network python bindings.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"

void expose_tensorNetwork() {
	VECTOR_TO_PY(TensorNetwork::Link, "LinkVector");
	VECTOR_TO_PY(TensorNetwork::TensorNode, "TensorNetworkNodeVector");
	// TODO allow modification of the network
	{ scope TN_scope =
		class_<TensorNetwork>("TensorNetwork")
			.def(init<Tensor>())
			.def(init<const TensorNetwork &>())
			.add_property("dimensions", +[](TensorNetwork &_A) {
				return _A.dimensions;
			})
			.def("degree", &TensorNetwork::degree)
			.def("datasize", &TensorNetwork::datasize)
			.add_property("nodes", +[](TensorNetwork &_this){
				return _this.nodes;
			})
			.def("node", +[](TensorNetwork &_this, size_t _i) {
				return _this.nodes[_i];
			})
			.add_property("externalLinks", +[](TensorNetwork &_this){
				return _this.externalLinks;
			})
			.def("__call__", +[](TensorNetwork &_this, const std::vector<Index> &_idx){
				return  new xerus::internal::IndexedTensor<TensorNetwork>(std::move(_this(_idx)));
			}, return_value_policy<manage_new_object, with_custodian_and_ward_postcall<0, 1>>() )
			.def(self * other<value_t>())
			.def(other<value_t>() * self)
			.def(self / other<value_t>())
			.def("__getitem__", +[](TensorNetwork &_this, size_t _i) {
				if (_i >= misc::product(_this.dimensions)) {
					PyErr_SetString(PyExc_IndexError, "Index out of range");
					throw_error_already_set();
				}
				return _this[_i];
			})
			.def("__getitem__", +[](TensorNetwork &_this, std::vector<size_t> _idx) {
				return _this[_idx];
			})
// 			.def("reshuffle_nodes", +[](TensorNetwork &_this, boost::python::object _f) { //TODO
// 				_this.reshuffle_nodes(_f);
// 			})
			.def("require_valid_network", +[](TensorNetwork &_this) {
				_this.require_valid_network();
			})
			.def("require_correct_format", &TensorNetwork::require_correct_format)
			.def("swap_external_links", &TensorNetwork::swap_external_links)
			.def("round_edge", &TensorNetwork::round_edge)
			.def("transfer_core", &TensorNetwork::transfer_core,
				(arg("from"), arg("to"), arg("allowRankReduction")=true)
			)
			.def("reduce_representation", &TensorNetwork::reduce_representation)
			.def("find_common_edge",
				+[](TensorNetwork &_this, size_t _nodeA, size_t _nodeB){
					const auto result = _this.find_common_edge(_nodeA, _nodeB);
					return boost::python::make_tuple(result.first, result.second);
				}
			)
			.def("sanitize", &TensorNetwork::sanitize)
			.def("fix_mode", &TensorNetwork::fix_mode)
			.def("remove_slate", &TensorNetwork::remove_slate)
			.def("resize_mode", &TensorNetwork::resize_mode,
				(arg("mode"), arg("newDimension"), arg("cutPosition")=~0ul)
			)
			.def("contract", static_cast<void (TensorNetwork::*)(size_t, size_t)>(&TensorNetwork::contract))
			.def("contract", static_cast<size_t (TensorNetwork::*)(const std::set<size_t>&)>(&TensorNetwork::contract)) //TODO write converter
			.def("contraction_cost", &TensorNetwork::contraction_cost)
			.def("draw", &TensorNetwork::draw)
			.def("frob_norm", &TensorNetwork::frob_norm)
		;

		class_<TensorNetwork::TensorNode>("TensorNode")
			.def("size", &TensorNetwork::TensorNode::size)
			.def("degree", &TensorNetwork::TensorNode::degree)
// 			.def_readonly("erased", &TensorNetwork::TensorNode::erased) // internal
// 			.def("erase", &TensorNetwork::TensorNode::erase) // internal
			.add_property("tensorObject", +[](TensorNetwork::TensorNode &_this)->object{
				if (_this.tensorObject) {
					return object(_this.tensorObject.get());
				} else {
					return object(); // aka none
				}
			})
			.add_property("neighbors", +[](TensorNetwork::TensorNode &_this){
				return _this.neighbors;
			})
		;

		class_<TensorNetwork::Link>("TensorNetworkLink")
			.def_readonly("other", &TensorNetwork::Link::other)
			.def_readonly("indexPosition", &TensorNetwork::Link::indexPosition)
			.def_readonly("dimension", &TensorNetwork::Link::dimension)
			.def_readonly("external", &TensorNetwork::Link::external)
			.def("links", &TensorNetwork::Link::links)
		;
	} // closes TN_scope
	variable_argument_member_to_tuple_wrapper("TensorNetwork.__call__", "TensorNetworkCallOperator");
}
