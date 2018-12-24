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
 * @brief Definition of the TT-Network python bindings.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"

void expose_ttnetwork() {
	VECTOR_TO_PY(TTTensor, "TTTensorVector");
	VECTOR_TO_PY(TTOperator, "TTOperatorVector");

	class_<TTTensor, bases<TensorNetwork>>("TTTensor")
		.def(init<const Tensor&, optional<value_t, size_t>>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def(init<const TTTensor &>())
		.def("get_component", &TTTensor::get_component, return_value_policy<copy_const_reference>())
		.def("set_component", &TTTensor::set_component)
		.def_readonly("canonicalized", &TTTensor::canonicalized)
		.def_readonly("corePosition", &TTTensor::corePosition)
		.def("ranks", &TTTensor::ranks)
		.def("rank", &TTTensor::rank)
// 		.def("frob_norm", &TTTensor::frob_norm) // NOTE unneccessary because correct call is inherited
		.def("random",
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTTensor::random(_dim, _rank);
			}).staticmethod("random")
		.def("constant",
			+[](value_t _constant, std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTTensor::constant(_constant, _dim, _rank);
			}).staticmethod("constant")
		.def("ones", &TTTensor::ones).staticmethod("ones")
		.def("kronecker", &TTTensor::kronecker).staticmethod("kronecker")
		.def("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTTensor::dirac))
		.def("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const size_t)>(&TTTensor::dirac)).staticmethod("dirac")

		.def("use_dense_representations", &TTTensor::use_dense_representations)
		.def("reduce_to_maximal_ranks", &TTTensor::reduce_to_maximal_ranks).staticmethod("reduce_to_maximal_ranks")
// 		.def("degrees_of_freedom", static_cast<size_t (TTTensor::*)()>(&TTTensor::degrees_of_freedom))
		.def("degrees_of_freedom", static_cast<size_t (*)(const std::vector<size_t>&, const std::vector<size_t>&)>(&TTTensor::degrees_of_freedom)).staticmethod("degrees_of_freedom")
		.def("chop",
			+[](TTTensor &_this, size_t _pos) {
				const auto result = _this.chop(_pos);
				return boost::python::make_tuple(result.first, result.second);
			}, arg("position"))

 		.def("round", static_cast<void (TTTensor::*)(const std::vector<size_t>&, double)>(&TTTensor::round),
 			(arg("ranks"), arg("epsilon")=EPSILON)
 		)
		.def("round", static_cast<void (TTTensor::*)(double)>(&TTTensor::round))
 		.def("round", static_cast<void (TTTensor::*)(size_t)>(&TTTensor::round))
		.def("soft_threshold", static_cast<void (TTTensor::*)(const double, const bool)>(&TTTensor::soft_threshold),
			(arg("tau"), arg("preventZero")=false)
		)
		.def("soft_threshold", static_cast<void (TTTensor::*)(const std::vector<double>&, const bool)>(&TTTensor::soft_threshold),
			(arg("tau"), arg("preventZero")=false)
		)

		.def("move_core", &TTTensor::move_core,
			(arg("position"), arg("keepRank")=false)
		)

		.def("assume_core_position", &TTTensor::assume_core_position)
		.def("canonicalize_left", &TTTensor::canonicalize_left)
		.def("canonicalize_right", &TTTensor::canonicalize_right)
		.def(self + self)
		.def(self - self)
		.def(self * other<value_t>())
		.def(other<value_t>() * self)
		.def(self / other<value_t>())
		.def(self += self)
		.def(self -= self)
	;

	def("entrywise_product", static_cast<TTTensor (*)(const TTTensor&, const TTTensor&)>(&entrywise_product));
	def("find_largest_entry", static_cast<size_t (*)(const TTTensor&, value_t, value_t)>(&find_largest_entry));
	def("dyadic_product", static_cast<TTTensor (*)(const std::vector<TTTensor> &)>(&dyadic_product));

	class_<TTOperator, bases<TensorNetwork>>("TTOperator")
		.def(init<const Tensor&, optional<value_t, size_t>>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def(init<const TTOperator &>())
		.def("get_component", &TTOperator::get_component, return_value_policy<copy_const_reference>())
		.def("set_component", &TTOperator::set_component)
		.def_readonly("canonicalized", &TTOperator::canonicalized)
		.def_readonly("corePosition", &TTOperator::corePosition)
		.def("ranks", &TTOperator::ranks)
		.def("rank", &TTOperator::rank)
// 		.def("frob_norm", &TTOperator::frob_norm) // NOTE unneccessary because correct call is inherited
		.def("random",
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTOperator::random(_dim, _rank);
			}).staticmethod("random")
		.def("constant",
			+[](value_t _constant, std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::TTOperator::constant(_constant, _dim, _rank);
			}).staticmethod("constant")
		.def("ones", &TTOperator::ones).staticmethod("ones")
		.def("kronecker", &TTOperator::kronecker).staticmethod("kronecker")
		.def("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTOperator::dirac))
		.def("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const size_t)>(&TTOperator::dirac)).staticmethod("dirac")

		.def("use_dense_representations", &TTOperator::use_dense_representations)
		.def("reduce_to_maximal_ranks", &TTOperator::reduce_to_maximal_ranks).staticmethod("reduce_to_maximal_ranks")
// 		.def("degrees_of_freedom", static_cast<size_t (TTOperator::*)()>(&TTOperator::degrees_of_freedom))
		.def("degrees_of_freedom", static_cast<size_t (*)(const std::vector<size_t>&, const std::vector<size_t>&)>(&TTOperator::degrees_of_freedom)).staticmethod("degrees_of_freedom")
		.def("chop",
			+[](TTOperator &_this, size_t _pos) {
				const auto result = _this.chop(_pos);
				return boost::python::make_tuple(result.first, result.second);
			}, arg("position"))

		.def("round", static_cast<void (TTOperator::*)(const std::vector<size_t>&, double)>(&TTOperator::round),
			(arg("ranks"), arg("epsilon")=EPSILON)
		)
		.def("round", static_cast<void (TTOperator::*)(double)>(&TTOperator::round))
		.def("round", static_cast<void (TTOperator::*)(size_t)>(&TTOperator::round))
		.def("soft_threshold", static_cast<void (TTOperator::*)(const double, const bool)>(&TTOperator::soft_threshold),
			(arg("tau"), arg("preventZero")=false)
		)
		.def("soft_threshold", static_cast<void (TTOperator::*)(const std::vector<double>&, const bool)>(&TTOperator::soft_threshold),
			(arg("tau"), arg("preventZero")=false)
		)

		.def("move_core", &TTOperator::move_core,
			(arg("position"), arg("keepRank")=false)
		)

		.def("assume_core_position", &TTOperator::assume_core_position)
		.def("canonicalize_left", &TTOperator::canonicalize_left)
		.def("canonicalize_right", &TTOperator::canonicalize_right)
		.def(self + self)
		.def(self - self)
		.def(self += self)
		.def(self -= self)
		.def(self * other<value_t>())
		.def(other<value_t>() * self)
		.def(self / other<value_t>())


		// for  TTOperator only:
		.def("identity", &TTOperator::identity<>).staticmethod("identity")
		.def("transpose", &TTOperator::transpose<>)
	;
	def("entrywise_product", static_cast<TTOperator (*)(const TTOperator&, const TTOperator&)>(&entrywise_product));
	def("find_largest_entry", static_cast<size_t (*)(const TTOperator&, value_t, value_t)>(&find_largest_entry));
	def("dyadic_product", static_cast<TTOperator (*)(const std::vector<TTOperator> &)>(&dyadic_product));
}
