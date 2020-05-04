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
 * @brief Definition of the hT-Network python bindings.
 */


#include "misc.h"

void expose_htnetwork(module& m) {
	class_<HTTensor, TensorNetwork>(m,"HTTensor")
		.def(pickle(
			[](const HTTensor &_self) { // __getstate__
				return bytes(misc::serialize(_self));
			},
			[](bytes _bytes) { // __setstate__
				return misc::deserialize<HTTensor>(_bytes);
			}
		))
		.def(init<const HTTensor &>())
		.def(init<const Tensor&>())
		.def(init<const Tensor&, value_t>())
		.def(init<const Tensor&, value_t, size_t>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def("get_component", &HTTensor::get_component)
		.def("set_component", &HTTensor::set_component)
		.def_readonly("canonicalized", &HTTensor::canonicalized)
		.def_readonly("corePosition", &HTTensor::corePosition)
		.def("ranks", &HTTensor::ranks)
		.def("rank", &HTTensor::rank)
		//.def("frob_norm", &TTTensor::frob_norm) // NOTE unneccessary because correct call is inherited
		.def_static("random",
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::HTTensor::random(_dim, _rank);
			})
		.def_static("ones", &HTTensor::ones)
		//.def_static("kronecker", &TTTensor::kronecker)
		//.def_static("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTTensor::dirac))
		//.def_static("dirac", static_cast<TTTensor (*)(Tensor::DimensionTuple, const size_t)>(&TTTensor::dirac))

		//.def("use_dense_representations", &TTTensor::use_dense_representations)
		//.def_static("reduce_to_maximal_ranks", &TTTensor::reduce_to_maximal_ranks)
		//.def("degrees_of_freedom", static_cast<size_t (TTTensor::*)()>(&TTTensor::degrees_of_freedom))

		.def("round", static_cast<void (HTTensor::*)(const std::vector<size_t>&, double)>(&HTTensor::round),
				arg("ranks"), arg("epsilon")=EPSILON
		)
		.def("round", static_cast<void (HTTensor::*)(double)>(&HTTensor::round))
		.def("round", static_cast<void (HTTensor::*)(size_t)>(&HTTensor::round))

		.def("soft_threshold", static_cast<void (HTTensor::*)(const double, const bool)>(&HTTensor::soft_threshold),
			arg("tau"), arg("preventZero")=false
		)
		.def("soft_threshold", static_cast<void (HTTensor::*)(const std::vector<double>&, const bool)>(&HTTensor::soft_threshold),
			arg("tau"), arg("preventZero")=false
		)
		.def("move_core", &HTTensor::move_core,arg("position"), arg("keepRank")=false)
		.def("assume_core_position", &HTTensor::assume_core_position)
		.def("canonicalize_root", &HTTensor::canonicalize_root)
	.def(self + self)
	.def(self - self)
	.def(self * value_t())
	.def(value_t() * self)
	.def(self / value_t())
	.def(self += self)
	.def(self -= self)
	;

//	def("entrywise_product", static_cast<TTTensor (*)(const TTTensor&, const TTTensor&)>(&entrywise_product));
//	def("find_largest_entry", static_cast<size_t (*)(const TTTensor&, value_t, value_t)>(&find_largest_entry));
//	def("dyadic_product", static_cast<TTTensor (*)(const std::vector<TTTensor> &)>(&dyadic_product));

	class_<HTOperator, TensorNetwork>(m,"HTOperator")
		 .def(init<const Tensor&>())
		.def(init<const Tensor&, value_t>())
		.def(init<const Tensor&, value_t, size_t>())
		.def(init<const Tensor&, value_t, TensorNetwork::RankTuple>())
		.def(init<Tensor::DimensionTuple>())
		.def(init<size_t>())
		.def(init<const HTOperator &>())
	.def("get_component", &HTOperator::get_component)
		.def("set_component", &HTOperator::set_component)
		.def_readonly("canonicalized", &HTOperator::canonicalized)
		.def_readonly("corePosition", &HTOperator::corePosition)
		.def("ranks", &HTOperator::ranks)
		.def("rank", &HTOperator::rank)
		//.def("frob_norm", &TTOperator::frob_norm) // NOTE unneccessary because correct call is inherited
		.def_static("random",
			+[](std::vector<size_t> _dim, std::vector<size_t> _rank) {
				return xerus::HTOperator::random(_dim, _rank);
			})
		.def_static("ones", &HTOperator::ones)
		//.def("kronecker", &TTOperator::kronecker).staticmethod("kronecker")
		//.def("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&TTOperator::dirac))
		//.def("dirac", static_cast<TTOperator (*)(Tensor::DimensionTuple, const size_t)>(&TTOperator::dirac)).staticmethod("dirac")

		//.def("use_dense_representations", &TTOperator::use_dense_representations)
		//.def("reduce_to_maximal_ranks", &TTOperator::reduce_to_maximal_ranks).staticmethod("reduce_to_maximal_ranks")
		//.def("degrees_of_freedom", static_cast<size_t (TTOperator::*)()>(&TTOperator::degrees_of_freedom))
		//.def("degrees_of_freedom", static_cast<size_t (*)(const std::vector<size_t>&, const std::vector<size_t>&)>(&TTOperator::degrees_of_freedom)).staticmethod("degrees_of_freedom")
		//.def("chop", ...)

		.def("round", static_cast<void (HTOperator::*)(const std::vector<size_t>&, double)>(&HTOperator::round),
			arg("ranks"), arg("epsilon")=EPSILON
		)
		.def("round", static_cast<void (HTOperator::*)(double)>(&HTOperator::round))
		.def("round", static_cast<void (HTOperator::*)(size_t)>(&HTOperator::round))

		.def("soft_threshold", static_cast<void (HTOperator::*)(const double, const bool)>(&HTOperator::soft_threshold),
			arg("tau"), arg("preventZero")=false
		)
		.def("soft_threshold", static_cast<void (HTOperator::*)(const std::vector<double>&, const bool)>(&HTOperator::soft_threshold),
			arg("tau"), arg("preventZero")=false
		)

		.def("move_core", &HTOperator::move_core,
			arg("position"), arg("keepRank")=false
		)

		.def("assume_core_position", &HTOperator::assume_core_position)
		.def("canonicalize_root", &HTOperator::canonicalize_root)
		.def(self + self)
		.def(self - self)
		.def(self += self)
		.def(self -= self)
		.def(self * value_t())
		.def(value_t() * self)
		.def(self / value_t())
//
//
//		// for  TTOperator only:
		.def_static("identity", &HTOperator::identity<>)
		.def("transpose", &HTOperator::transpose<>)
	;
//	def("entrywise_product", static_cast<TTOperator (*)(const TTOperator&, const TTOperator&)>(&entrywise_product));
//	def("find_largest_entry", static_cast<size_t (*)(const TTOperator&, value_t, value_t)>(&find_largest_entry));
//	def("dyadic_product", static_cast<TTOperator (*)(const std::vector<TTOperator> &)>(&dyadic_product));
}
