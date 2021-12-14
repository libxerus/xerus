#include "misc.h"

void expose_tensorNetwork(module& m) {
	class_<TensorNetwork>(m, "TensorNetwork")
		.def(pickle(
			+[](const TensorNetwork &_self) { // __getstate__
				return bytes(misc::serialize(_self));
			},
			+[](bytes _bytes) { // __setstate__
				return misc::deserialize<TensorNetwork>(_bytes);
			}
		))
		.def(init<>())
		.def(init<Tensor>())
		.def(init<const TensorNetwork &>())
		.def("__float__", [](const TensorNetwork &_self){
				if (_self.order() != 0) {
					throw value_error("order must be 0");
				}
				return value_t(_self());
		})
		.def_readonly("dimensions", &TensorNetwork::dimensions)
		.def("order", &TensorNetwork::order)
		.def("datasize", &TensorNetwork::datasize)
		.def_readonly("nodes", &TensorNetwork::nodes)
		.def("node", +[](TensorNetwork &_this, size_t _i) {
			return _this.nodes[_i];
		})
		.def_readonly("externalLinks", &TensorNetwork::externalLinks)
		/* .def("__call__", +[](TensorNetwork &_this, const std::vector<Index> &_idx){ */
		/*     return  new xerus::internal::IndexedTensor<TensorNetwork>(std::move(_this(_idx))); */
		/* }, return_value_policy<manage_new_object, with_custodian_and_ward_postcall<0, 1>>() ) */
		.def("__call__", +[](TensorNetwork& _this, args _args){
			std::vector<Index> idx;
			idx.reserve(_args.size());
			for (size_t i=0; i<_args.size(); ++i) {
				idx.push_back(*(_args[i].cast<Index *>()));
			}
			return new xerus::internal::IndexedTensor<TensorNetwork>(std::move(_this(idx)));
		}, keep_alive<0, 1>(), return_value_policy::take_ownership )
		.def(self * value_t())
		.def(value_t() * self)
		.def(self / value_t())
		.def("__getitem__", +[](TensorNetwork &_this, size_t _i) {
			if (_i >= misc::product(_this.dimensions)) {
				throw index_error("Index out of range");
			}
			return _this[_i];
		})
		.def("__getitem__", +[](TensorNetwork &_this, std::vector<size_t> _idx) {
			return _this[_idx];
		})
		.def("reshuffle_nodes", &TensorNetwork::reshuffle_nodes, arg("function"), "reshuffle the nodes according to the given function")
		.def("require_valid_network", +[](TensorNetwork &_this) {
			_this.require_valid_network();
		})
		.def("require_correct_format", &TensorNetwork::require_correct_format)
		.def("swap_external_links", &TensorNetwork::swap_external_links)
		.def("round_edge", &TensorNetwork::round_edge)
		.def("transfer_core", &TensorNetwork::transfer_core, arg("from"), arg("to"), arg("allowRankReduction")=true)
		.def("reduce_representation", &TensorNetwork::reduce_representation)
		.def("find_common_edge", &TensorNetwork::find_common_edge)
		.def("sanitize", &TensorNetwork::sanitize)
		.def("fix_mode", &TensorNetwork::fix_mode)
		.def("remove_slate", &TensorNetwork::remove_slate)
		.def("resize_mode", &TensorNetwork::resize_mode, arg("mode"), arg("newDimension"), arg("cutPosition")=~0ul)
		.def("contract", static_cast<void (TensorNetwork::*)(size_t, size_t)>(&TensorNetwork::contract))
		.def("contract", static_cast<size_t (TensorNetwork::*)(const std::set<size_t>&)>(&TensorNetwork::contract)) //TODO write converter
		.def("contraction_cost", &TensorNetwork::contraction_cost)
		.def("draw", &TensorNetwork::draw)
		.def("frob_norm", &TensorNetwork::frob_norm)
	;

	class_<TensorNetwork::TensorNode>(m, "TensorNode")
		.def("size", &TensorNetwork::TensorNode::size)
		.def("order", &TensorNetwork::TensorNode::order)
		/* .def_readonly("erased", &TensorNetwork::TensorNode::erased) // internal */
		/* .def("erase", &TensorNetwork::TensorNode::erase) // internal */
		.def_property_readonly("tensorObject", +[](TensorNetwork::TensorNode &_this)->object {
			if (_this.tensorObject) {
				return cast(_this.tensorObject.get());
			} else {
				return none();
			}
		})
		.def_readonly("neighbors", &TensorNetwork::TensorNode::neighbors);
	;

	class_<TensorNetwork::Link>(m, "TensorNetworkLink")
		.def_readonly("other", &TensorNetwork::Link::other)
		.def_readonly("indexPosition", &TensorNetwork::Link::indexPosition)
		.def_readonly("dimension", &TensorNetwork::Link::dimension)
		.def_readonly("external", &TensorNetwork::Link::external)
		.def("links", &TensorNetwork::Link::links)
	;
}
