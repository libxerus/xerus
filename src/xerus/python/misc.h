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
 * @brief Definition of common functions for the python bindings.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eval.h>
// #include <pybind11/numpy.h>
// #undef NDEBUG
#include "xerus.h"
#include "xerus/misc/internal.h"  // REQUIRE
#pragma GCC diagnostic pop


using namespace pybind11;
using namespace xerus;


#define parametersDocstr "\n\nParameters\n----------\n"
#define returnsDocstr "\n\nReturns\n-------\n"


void expose_indexedTensors(module& m);
void expose_factorizations(module& m);

void expose_tensor(module& m);
void expose_tensorNetwork(module& m);
void expose_ttnetwork(module& m);
void expose_htnetwork(module& m);

void expose_leastSquaresAlgorithms(module& m);
void expose_recoveryAlgorithms(module& m);

void expose_misc(module& m);
