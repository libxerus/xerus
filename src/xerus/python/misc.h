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
 * @brief Definition of common functions for the python bindings.
 */

#include <boost/function.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/call.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wpedantic"
#define PY_ARRAY_UNIQUE_SYMBOL XERUS_ARRAY_API
#include <numpy/ndarrayobject.h>
#pragma GCC diagnostic pop
#include "xerus.h"
#include "xerus/misc/internal.h"

using namespace boost::python;
using namespace xerus;


#define parametersDocstr "\n\nParameters\n----------\n"
#define	returnsDocstr "\n\nReturns\n-------\n"

#define VECTOR_TO_PY(type, name) class_<std::vector<type>, boost::noncopyable>(name, no_init); \
	custom_vector_from_seq<type>(); \
	to_python_converter<std::vector<type>, custom_vector_to_list<type>>(); void(0)

#include "vectorAndPair.h"

void variable_argument_member_to_tuple_wrapper(const std::string &_name, const std::string &_tmpName = "new_fn");

void expose_tensor();
void expose_factorizations();
void expose_indexedTensors();
void expose_tensorNetwork();
void expose_ttnetwork();
void expose_htnetwork();

void expose_leastSquaresAlgorithms();
void expose_recoveryAlgorithms();

void expose_misc();
