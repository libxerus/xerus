#include "misc.h"

void expose_ttnetwork(module& m) {
	class_<TTTensor, TensorNetwork>(m, "TTTensor")
		.def(pickle(
			[](const TTTensor &_self) { // __getstate__
				return bytes(misc::serialize(_self));
			},
			[](bytes _bytes) { // __setstate__
				return misc::deserialize<TTTensor>(_bytes);
			}
		))
		.def(init<>(), "constructs an empty TTTensor")
		.def(init<const TTTensor &>())
		.def(init<const Tensor&>())
		.def(init<const Tensor&, value_t>())
		.def(init<const Tensor&, value_t, size_t>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def("get_component", &TTTensor::get_component)
		.def("set_component", &TTTensor::set_component)
		.def_readonly("canonicalized", &TTTensor::canonicalized)
		.def_readonly("corePosition", &TTTensor::corePosition)
		.def("ranks", &TTTensor::ranks)
		.def("rank", &TTTensor::rank)
		/* .def("frob_norm", &TTTensor::frob_norm) // NOTE unneccessary because correct call is inherited */
		.def_static("random",
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTTensor::random(_dim, _rank);
			})
		.def_static("ones", &TTTensor::ones)
		.def_static("kronecker", &TTTensor::kronecker)
		/* .def_static("dirac", &TTTensor::dirac) */
		.def_static("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTTensor::dirac))
		.def_static("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const size_t)>(&TTTensor::dirac))

		.def("use_dense_representations", &TTTensor::use_dense_representations)
		.def_static("reduce_to_maximal_ranks", &TTTensor::reduce_to_maximal_ranks)
		/* .def("degrees_of_freedom", static_cast<size_t (TTTensor::*)() const>(&TTTensor::degrees_of_freedom))  // NOTE overloading a method with both static and instance methods is not supported */
		.def_static("degrees_of_freedom", static_cast<size_t (*)(const std::vector<size_t>&, const std::vector<size_t>&)>(&TTTensor::degrees_of_freedom))
		.def("chop", &TTTensor::chop, arg("position"))

		.def("round", static_cast<void (TTTensor::*)(const std::vector<size_t>&, double)>(&TTTensor::round),
			arg("ranks"), arg("epsilon")=EPSILON
		)
		.def("round", static_cast<void (TTTensor::*)(double)>(&TTTensor::round))
		.def("round", static_cast<void (TTTensor::*)(size_t)>(&TTTensor::round))
		.def("soft_threshold", static_cast<void (TTTensor::*)(const double)>(&TTTensor::soft_threshold), arg("tau"))
		.def("soft_threshold", static_cast<void (TTTensor::*)(const std::vector<double>&)>(&TTTensor::soft_threshold), arg("tau"))

		.def("move_core", &TTTensor::move_core, arg("position"), arg("keepRank")=false)

		.def("assume_core_position", &TTTensor::assume_core_position)
		.def("canonicalize_left", &TTTensor::canonicalize_left)
		.def("canonicalize_right", &TTTensor::canonicalize_right)
		/* .def(-self) */
		.def("__neg__",
			+[](TTTensor& _self) {
				return (-1)*_self;
			})
		.def(self + self)
		.def(self - self)
		.def(self += self)
		.def(self -= self)
		.def(self * value_t())
		.def(value_t() * self)
		.def(self *= value_t())
		.def(self / value_t())
		/* .def(self /= self) */
		.def("__itruediv__",
			+[](TTTensor& _self, value_t _other) {
				return (_self *= (1/_other));
			})
	;

	m.def("entrywise_product", static_cast<TTTensor (*)(const TTTensor&, const TTTensor&)>(&entrywise_product));
	m.def("find_largest_entry", static_cast<size_t (*)(const TTTensor&, value_t, value_t)>(&find_largest_entry));
	m.def("dyadic_product", static_cast<TTTensor (*)(const std::vector<TTTensor> &)>(&dyadic_product));

	class_<TTOperator, TensorNetwork>(m, "TTOperator")
		.def(init<const Tensor&>())
		.def(init<const Tensor&, value_t>())
		.def(init<const Tensor&, value_t, size_t>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def(init<const TTOperator &>())
		.def("get_component", &TTOperator::get_component)
		.def("set_component", &TTOperator::set_component)
		.def_readonly("canonicalized", &TTOperator::canonicalized)
		.def_readonly("corePosition", &TTOperator::corePosition)
		.def("ranks", &TTOperator::ranks)
		.def("rank", &TTOperator::rank)
		/* .def("frob_norm", &TTOperator::frob_norm) // NOTE unneccessary because correct call is inherited */
		.def_static("random", //TODO check error throwing python crashes when error from xerus is thrown
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTOperator::random(_dim, _rank);
			})
		.def_static("ones", &TTOperator::ones)
		.def_static("kronecker", &TTOperator::kronecker)
		/* .def_static("dirac", &TTOperator::dirac) */
		.def_static("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTOperator::dirac))
		.def_static("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const size_t)>(&TTOperator::dirac))

		.def("use_dense_representations", &TTOperator::use_dense_representations)
		.def_static("reduce_to_maximal_ranks", &TTOperator::reduce_to_maximal_ranks)
		/* .def("degrees_of_freedom", static_cast<size_t (TTOperator::*)()>(&TTOperator::degrees_of_freedom)) */
		.def_static("degrees_of_freedom", static_cast<size_t (*)(const std::vector<size_t>&, const std::vector<size_t>&)>(&TTOperator::degrees_of_freedom))
		.def("chop", &TTOperator::chop, arg("position"))

		.def("round", static_cast<void (TTOperator::*)(const std::vector<size_t>&, double)>(&TTOperator::round), arg("ranks"), arg("epsilon")=EPSILON)
		.def("round", static_cast<void (TTOperator::*)(double)>(&TTOperator::round))
		.def("round", static_cast<void (TTOperator::*)(size_t)>(&TTOperator::round))
		.def("soft_threshold", static_cast<void (TTOperator::*)(const double)>(&TTOperator::soft_threshold), arg("tau"))
		.def("soft_threshold", static_cast<void (TTOperator::*)(const std::vector<double>&)>(&TTOperator::soft_threshold), arg("tau"))

		.def("move_core", &TTOperator::move_core, arg("position"), arg("keepRank")=false)

		.def("assume_core_position", &TTOperator::assume_core_position)
		.def("canonicalize_left", &TTOperator::canonicalize_left)
		.def("canonicalize_right", &TTOperator::canonicalize_right)
		/* .def(-self) */
		.def("__neg__",
			+[](TTTensor& _self) {
				return (-1)*_self;
			})
		.def(self + self)
		.def(self - self)
		.def(self += self)
		.def(self -= self)
		.def(self * value_t())
		.def(value_t() * self)
		.def(self *= value_t())
		.def(self / value_t())
		/* .def(self /= self) */
		.def("__itruediv__",
			+[](TTTensor& _self, value_t _other) {
				return (_self *= (1/_other));
			})

		// for  TTOperator only:
		.def_static("identity", &TTOperator::identity<>)
		.def("transpose", &TTOperator::transpose<>)
	;

	m.def("entrywise_product", static_cast<TTOperator (*)(const TTOperator&, const TTOperator&)>(&entrywise_product));
	m.def("find_largest_entry", static_cast<size_t (*)(const TTOperator&, value_t, value_t)>(&find_largest_entry));
	m.def("dyadic_product", static_cast<TTOperator (*)(const std::vector<TTOperator> &)>(&dyadic_product));
}
