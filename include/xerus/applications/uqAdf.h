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
 * @brief Header file for the ADF algorithm and its variants.
 */

#pragma once

#include "uq.h"
#include "../blockTT.h"

namespace xerus { namespace uq {

	///@brief Inplace variant of the UQ ADF to find a solution @a _x (with the inital dimensions and rank) for a given set of @a _measurements.
    void uq_adf(TTTensor& _x, const UQMeasurementSet& _measurements, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0);

	
	///@brief Inplace variant of the rank-adaptive UQ ADF, to find a solution @a _x (with the inital dimensions) for a given set of @a _measurements.
    void uq_ra_adf(TTTensor& _x, const UQMeasurementSet& _measurements, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0, const double _initalRankEps = 1e-2);

	
	///@brief Rank-adaptive UQ ADF to find a solution with given @a _dimensions for a given set of @a _measurements.
    TTTensor uq_ra_adf(const UQMeasurementSet& _measurements, const PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

	/**
	 * @brief Rank adaptive ADF to calculate the UQ solution for given measurements.
	 * @param _positions The positions of the measurements.
	 * @param _solutions The measured solutions corresponding to the @a _positions.
	 * @param _dimensions The dimensions of the final solution tensor.
	 * @param _targetEps TODO describe effect
	 * @param _maxItr Maximal number of iterations to perform.
	 */
    TTTensor uq_ra_adf(const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

	/**
	 * @brief Rank adaptive ADF to calculate the UQ solution for given measurements.
	 * @param _x Initial value.
	 * @param _positions The positions of the measurements.
	 * @param _solutions The measured solutions corresponding to the @a _positions.
	 * @param _dimensions The dimensions of the final solution tensor.
	 * @param _targetEps TODO describe effect
	 * @param _maxItr Maximal number of iterations to perform.
	 */
    TTTensor uq_ra_adf(const TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

	/**
	 * @brief Rank adaptive ADF to calculate the UQ solution for given weighted measurements.
	 * @param _positions The positions of the measurements.
	 * @param _solutions The measured solutions corresponding to the @a _positions.
	 * @param _weights Weights for the individual measurements.
	 * @param _dimensions The dimensions of the final solution tensor.
	 * @param _targetEps TODO describe effect
	 * @param _maxItr Maximal number of iterations to perform.
	 */
    TTTensor uq_ra_adf(const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<double>& _weights, const std::vector<size_t>& _dimensions, const double _targetEps = 1e-8, const size_t _maxItr = 0);

	
    TTTensor uq_ra_adf(const TTTensor& _x, const UQMeasurementSet& _measurements, const PolynomBasis _basisType, const double _targetEps = 1e-8, const size_t _maxItr = 0);
}}
