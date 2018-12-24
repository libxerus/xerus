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
 * @brief Header file for the ADF algorithm and its variants.
 */

#pragma once

#include "uq.h"
#include "../blockTT.h"

namespace xerus { namespace uq {

    void uq_adf(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0);

    TTTensor uq_adf(const UQMeasurementSet& _initalMeasurments, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

    void uq_ra_adf(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0, const double _initalRankEps = 1e-2);

    TTTensor uq_ra_adf(const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

    TTTensor uq_ra_adf_iv(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0);
}}
