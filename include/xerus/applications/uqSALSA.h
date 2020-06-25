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

#include "../ttNetwork.h"

namespace xerus { namespace uq {

    class SALSA {
        TTTensor x;
        std::vector<std::vector<Tensor>> measures;
        std::vector<Tensor> values;

        const size_t M;
        const size_t N;
        const size_t P;

        const std::vector<size_t> maxTheoreticalRanks;

        double alpha, omega, smin;

        std::pair<size_t, size_t> trainingSet;
        std::pair<size_t, size_t> validationSet;

        double valueNorm_trainingSet;
        double valueNorm_validationSet;

        std::vector<std::vector<Tensor>> leftLHSStack;  // contains successive contractions of x.T@A.T@A@x
        std::vector<std::vector<Tensor>> leftRHSStack;  // contains successive contractions of x.T@A.T@b
        std::vector<std::vector<Tensor>> rightStack;    // contains successive contractions of A@x
        std::vector<Tensor> leftRegularizationStack;
        std::vector<Tensor> rightRegularizationStack;

        std::vector<std::vector<double>> singularValues;
        std::vector<double> weightedNorms;  //TODO: rename: densities

    public:
        double controlSetFraction = 0.1;

        // Convergence parameters
        double targetResidual = 1e-8;

        // Stagnation/Divergence parameters
        double minDecrease = 1e-3;
        size_t maxIterations = 1000;
        size_t trackingPeriodLength = 10;
        size_t maxNonImprovingAlphaCycles = 10;

        // Inactive rank parameters
        size_t kmin = 2;
        std::vector<size_t> maxRanks;

        // IRLS parameters
        size_t maxIRsteps = 3;
        double IRtolerance = 0.05;
        double sparsityThreshold = 1e-4;

        // SALSA parameters
        double fomega = 1.05;
        double omega_factor = 1;

        // LASSO parameters
        double falpha = 1.05;
        double alpha_factor = 1;
        std::vector<Tensor> basisWeights;

        /* // Reweighting parameters */
        /* std::vector<double> weights; */

        double initialResidual;  //TODO: rename
        size_t bestIteration;
        TTTensor bestX;
        double bestTrainingResidual;
        double bestValidationResidual;

        SALSA(const TTTensor& _x, const std::vector<Tensor>& _measures, const Tensor& _values);
        void run();

    private:
        void move_core_left(const bool adapt);
        void move_core_right(const bool adapt);
        void calc_left_stack(const size_t _position);
        void calc_right_stack(const size_t _position);
        void adapt_rank(Tensor& _U, Tensor& _S, Tensor& _Vt, const size_t _maxRank, const double _threshold) const;
        double residual(const std::pair<size_t, size_t>& _slice) const;
        double slow_residual(const std::pair<size_t, size_t>& _slice) const;
        Tensor omega_operator() const;
        Tensor alpha_operator() const;
        void solve_local();
        void print_parameters() const;
        void initialize();
        std::string print_fractional_ranks() const;
        std::string print_densities() const;
    };

}}
