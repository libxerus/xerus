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
 * @brief Implementation of the ADF variants.
 */

#include <xerus/applications/uqAdf.h>

#include <xerus/blockTT.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <boost/circular_buffer.hpp>
#pragma GCC diagnostic pop

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace uq { namespace impl_uqRaAdf {
	const size_t tracking = 10;

	template<size_t P>
	class InternalSolver {
		// Minimizes \sum_i w_i 0.5 \norm{A_i x - b_i}^2
		// where w_i are weights, A_i are evaluations at positions and b_i are solutions.
		const size_t N;
		const size_t d;

		double initialResiduum = 0.0;
		const double targetResidual;
		const size_t maxRank = 50;
		const double minRankEps = 1e-10;
		const double epsDecay = 0.8;

		const double convergenceFactor = 0.995;
		const size_t maxIterations;

		const double controlSetFraction = 0.1;

		const std::vector<std::vector<Tensor>> positions;
		const std::vector<Tensor>& solutions;
		const std::vector<double> weights;

		TTTensor& outX;

		std::vector<std::vector<size_t>> sets;
		std::vector<size_t> controlSet;

		double optNorm;
		double testNorm;
		std::vector<double> setNorms = std::vector<double>(P);

		double bestTestResidual = std::numeric_limits<double>::max();
		internal::BlockTT bestX;

		internal::BlockTT x;

		std::vector<std::vector<Tensor>> rightStack;  // From corePosition 1 to d-1
		std::vector<std::vector<Tensor>> leftIsStack;  // The "left part" of (A_i^T (A_i x))
		std::vector<std::vector<Tensor>> leftOughtStack;  // The "left part" of (solutions[i] * (A_i x))

		double rankEps;
		boost::circular_buffer<std::vector<size_t>> prevRanks;
		boost::circular_buffer<double> residuals {tracking, std::numeric_limits<double>::max()};


	public:
		static std::vector<std::vector<Tensor>> create_positions(const TTTensor& _x, const PolynomBasis _basisType, const std::vector<std::vector<double>>& _randomVariables) {
			std::vector<std::vector<Tensor>> positions(_x.order());

			for(size_t corePosition = 1; corePosition < _x.order(); ++corePosition) {
				positions[corePosition].reserve(_randomVariables.size());
				for(size_t j = 0; j < _randomVariables.size(); ++j) {
					positions[corePosition].push_back(polynomial_basis_evaluation(_randomVariables[j][corePosition-1], _basisType, _x.dimensions[corePosition]));
				}
			}

			return positions;
		}


		static std::vector<std::vector<Tensor>> transpose_positions(const TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions) {
			REQUIRE(_positions.size() == _solutions.size(), "Incompatible positions and solutions vector: " << _positions.size() << " vs " << _solutions.size());
			for(size_t sample=0; sample < _positions.size(); ++sample) {
				REQUIRE(_positions[sample].size() == _x.order()-1, "Invalid measurement: " << _positions[sample].size() << " vs " <<  _x.order()-1 << " (at sample " << sample << ")");
			}

			std::vector<std::vector<Tensor>> positions(_x.order());
			for(size_t corePosition=1; corePosition < _x.order(); ++corePosition) {
				positions[corePosition].reserve(_positions.size());
				for(size_t sample=0; sample < _positions.size(); ++sample) {
					REQUIRE(_positions[sample][corePosition-1].dimensions.size() == 1, "Invalid measurement component: len(" << _positions[sample][corePosition-1].dimensions << ") != 1 (at sample " << sample << " & position " << corePosition << ")");
					REQUIRE(_positions[sample][corePosition-1].size == _x.dimensions[corePosition], "Invalid measurement component"); // << _positions[sample][corePosition-1].size == _x.dimensions[corePosition]);
					positions[corePosition].push_back(_positions[sample][corePosition-1]);
				}
			}

			return positions;
		}


		void shuffle_sets() {
			LOG(uqADF, "Shuffle sets");
			sets = std::vector<std::vector<size_t>>(P);
			controlSet.clear();

			std::uniform_real_distribution<double> stochDist(0.0, 1.0);
			std::uniform_int_distribution<size_t> setDist(0, P-1);

			for(size_t j = 0; j < N; ++j) {
				if(stochDist(misc::randomEngine) > controlSetFraction) {
					sets[setDist(misc::randomEngine)].push_back(j);
				} else {
					controlSet.push_back(j);
				}
			}

			std::vector<double> setWeights(2,0);
			for(size_t k = 0; k < sets.size(); ++k) {
				for(const auto j : sets[k]) {
					setWeights[k] += weights[j];  // HIER
				}
			}
			LOG(uqADF, "Set weights " << std::scientific << setWeights);

			double testWeight = 0.0;
			for(const auto j : controlSet) {
				testWeight += weights[j];  // HIER
			}
			LOG(uqADF, "Test weights " << std::scientific << testWeight);

			calc_solution_norms();
		}


		void calc_solution_norms() {
			optNorm = 0.0;
			for(size_t k = 0; k < sets.size(); ++k) {
				setNorms[k] = 0.0;
				for(const auto j : sets[k]) {
					const double sqrNorm = misc::sqr(frob_norm(solutions[j]));
					optNorm += weights[j] * sqrNorm;  // HIER
					setNorms[k] += weights[j] * sqrNorm;  // HIER
				}
				setNorms[k] = std::sqrt(setNorms[k]);
			}
			optNorm = std::sqrt(optNorm);

			testNorm = 0.0;
			for(const auto j : controlSet) {
				const double sqrNorm = misc::sqr(frob_norm(solutions[j]));
				testNorm += weights[j] * sqrNorm;  // HIER
			}
			testNorm = std::sqrt(testNorm);
		}


		InternalSolver(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const size_t _maxItr, const double _targetEps, const double _initalRankEps) :
			N(_measurments.size()),
			d(_x.order()),
			targetResidual(_targetEps),
			maxIterations(_maxItr),
			positions(create_positions(_x, _basisType, _measurments.parameterVectors)),
			solutions(_measurments.solutions),
			weights(std::vector<double>(N, 1.0)),
			outX(_x),
			x(_x, 0, P),
			rightStack(d, std::vector<Tensor>(N)),
			leftIsStack(d, std::vector<Tensor>(N)),
			leftOughtStack(d, std::vector<Tensor>(N)),
			rankEps(_initalRankEps),
			prevRanks(tracking+1, _x.ranks())
			{
				LOG(uqADF, "Set size: " << _measurments.size());

				shuffle_sets();
		}


		InternalSolver(TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const size_t _maxItr, const double _targetEps, const double _initalRankEps) :
			N(_solutions.size()),
			d(_x.order()),
			targetResidual(_targetEps),
			maxIterations(_maxItr),
			positions(transpose_positions(_x, _positions, _solutions)),
			solutions(_solutions),
			weights(std::vector<double>(N, 1.0)),
			outX(_x),
			x(_x, 0, P),
			rightStack(d, std::vector<Tensor>(N)),
			leftIsStack(d, std::vector<Tensor>(N)),
			leftOughtStack(d, std::vector<Tensor>(N)),
			rankEps(_initalRankEps),
			prevRanks(tracking+1, _x.ranks())
			{
				LOG(uqADF, "Set size: " << N);

				shuffle_sets();
		}

		InternalSolver(TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<double>& _weights, const size_t _maxItr, const double _targetEps, const double _initalRankEps) :
			N(_solutions.size()),
			d(_x.order()),
			targetResidual(_targetEps),
			maxIterations(_maxItr),
			positions(transpose_positions(_x, _positions, _solutions)),
			solutions(_solutions),
			weights(_weights),
			outX(_x),
			x(_x, 0, P),
			rightStack(d, std::vector<Tensor>(N)),
			leftIsStack(d, std::vector<Tensor>(N)),
			leftOughtStack(d, std::vector<Tensor>(N)),
			rankEps(_initalRankEps),
			prevRanks(tracking+1, _x.ranks())
			{
				LOG(uqADF, "Set size: " << N);

				shuffle_sets();
			}


		void calc_left_stack(const size_t _position) {
			REQUIRE(_position+1 < d, "Invalid corePosition");

			if(_position == 0) {
				Tensor shuffledX = x.get_component(0);
				shuffledX.reinterpret_dimensions({x.dimensions[0], x.rank(0)}); // Remove dangling 1-mode

				#pragma omp parallel for
				for(size_t j = 0; j < N; ++j) {
					// NOTE: leftIsStack[0] is always an identity
					contract(leftOughtStack[_position][j], solutions[j], shuffledX, 1);
				}

			} else { // _position > 0
				const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
				Tensor measCmp, tmp;
				#pragma omp parallel for firstprivate(measCmp, tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(measCmp, positions[_position][j], shuffledX, 1);

					if(_position > 1) {
						contract(tmp, measCmp, true, leftIsStack[_position-1][j], false,  1);
						contract(leftIsStack[_position][j], tmp, measCmp, 1);
					} else { // _position == 1
						contract(leftIsStack[_position][j], measCmp, true, measCmp, false, 1);
					}

					contract(leftOughtStack[_position][j], leftOughtStack[_position-1][j], measCmp, 1);
				}
			}
		}


		void calc_right_stack(const size_t _position) {
			REQUIRE(_position > 0 && _position < d, "Invalid corePosition");
			Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});

			if(_position+1 < d) {
				Tensor tmp;
				#pragma omp parallel for firstprivate(tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(tmp, positions[_position][j], shuffledX, 1);
					contract(rightStack[_position][j], tmp, rightStack[_position+1][j], 1);
				}
			} else { // _position == d-1
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				#pragma omp parallel for
				for(size_t j = 0; j < N; ++j) {
					contract(rightStack[_position][j], positions[_position][j], shuffledX, 1);
				}
			}
		}


		Tensor calculate_delta(const size_t _corePosition, const size_t _setId) const {
			REQUIRE(x.corePosition == _corePosition, "IE");

			Tensor delta(x.get_core(_setId).dimensions);
			Tensor dyadComp, tmp;

			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))

			if(_corePosition > 0) {
				const Tensor shuffledX = reshuffle(x.get_core(_setId), {1, 0, 2});

				#pragma omp parallel for reduction(+: delta) firstprivate(_corePosition, _setId, dyadComp, tmp, shuffledX) default(none)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];

					// Calculate common "dyadic part"
					Tensor dyadicPart;
					if(_corePosition < d-1) {
						contract(dyadicPart, positions[_corePosition][j], rightStack[_corePosition+1][j], 0);
					} else {
						dyadicPart = positions[_corePosition][j];
						dyadicPart.reinterpret_dimensions({dyadicPart.dimensions[0], 1}); // Add dangling 1-mode
					}

					// Calculate "is"
					Tensor isPart;
					contract(isPart, positions[_corePosition][j], shuffledX, 1);

					if(_corePosition < d-1) {
						contract(isPart, isPart, rightStack[_corePosition+1][j], 1);
					} else {
						isPart.reinterpret_dimensions({isPart.dimensions[0]});
					}

					if(_corePosition > 1) { // NOTE: For _corePosition == 1 leftIsStack is the identity
						contract(isPart, leftIsStack[_corePosition-1][j], isPart, 1);
					}


					// Combine with ought part
					contract(dyadComp, isPart - leftOughtStack[_corePosition-1][j], dyadicPart, 0);

					delta += weights[j] * dyadComp;  // HIER
				}
			} else { // _corePosition == 0
				Tensor shuffledX = x.get_core(_setId);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]});

				#pragma omp parallel for reduction(+: delta) firstprivate(_corePosition, _setId, dyadComp, tmp, shuffledX) default(none)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
					contract(dyadComp, shuffledX, rightStack[_corePosition+1][j], 1);
					contract(dyadComp, dyadComp - solutions[j], rightStack[_corePosition+1][j], 0);
					dyadComp.reinterpret_dimensions({1, dyadComp.dimensions[0], dyadComp.dimensions[1]});

					delta += weights[j] * dyadComp;  // HIER
				}
			}

			return delta;
		}


		double calculate_norm_A_projGrad(const Tensor& _delta, const size_t _corePosition, const size_t _setId) const {
			double norm = 0.0;
			Tensor tmp;

			if(_corePosition == 0) {
				#pragma omp parallel for firstprivate(tmp) reduction(+:norm)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
					contract(tmp, _delta, rightStack[1][j], 1);
					const double normPart = misc::sqr(frob_norm(tmp));
					norm += weights[j] * normPart;  // HIER
				}
			} else { // _corePosition > 0
				Tensor shuffledDelta = reshuffle(_delta, {1, 0, 2});
				if(_corePosition+1 == d) {
					shuffledDelta.reinterpret_dimensions({shuffledDelta.dimensions[0], shuffledDelta.dimensions[1]}); // Remove dangling 1-mode
				}

				Tensor rightPart;
				#pragma omp parallel for firstprivate(tmp, rightPart) reduction(+:norm)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];

					// Current node
					contract(tmp, positions[_corePosition][j], shuffledDelta, 1);

					if(_corePosition+1 < d) {
						contract(rightPart, tmp, rightStack[_corePosition+1][j], 1);
					} else {
						rightPart = tmp;
					}

					if(_corePosition > 1) {
						contract(tmp, rightPart, leftIsStack[_corePosition-1][j], 1);
						contract(tmp, tmp, rightPart, 1);
					} else { // NOTE: For _corePosition == 1 leftIsStack is the identity
						contract(tmp, rightPart, rightPart, 1);
					}

					REQUIRE(tmp.size == 1, "IE");
					norm += weights[j] * tmp[0];  // HIER
				}
			}

			return std::sqrt(norm);
		}



		std::tuple<double, double, std::vector<double>> calc_residuals(const size_t _corePosition) const {
			REQUIRE(_corePosition == 0, "Invalid corePosition");

			// TODO paralell

			const auto avgCore = x.get_average_core();
			Tensor tmp;

			double optResidual = 0.0;
			std::vector<double> setResiduals(sets.size(), 0.0);
			for(size_t k = 0; k < sets.size(); ++k) {
				for(const auto j : sets[k]) {
					contract(tmp, avgCore, rightStack[1][j], 1);
					tmp.reinterpret_dimensions({x.dimensions[0]});
					tmp -= solutions[j];
					const double resSqr = misc::sqr(frob_norm(tmp));

					optResidual += weights[j] * resSqr;  // HIER
					setResiduals[k] += weights[j] * resSqr;  // HIER
				}
				setResiduals[k] = std::sqrt(setResiduals[k])/setNorms[k];
			}
			optResidual = std::sqrt(optResidual)/optNorm;

			double testResidual = 0.0;
			for(const auto j : controlSet) {
				contract(tmp, avgCore, rightStack[1][j], 1);
				tmp.reinterpret_dimensions({x.dimensions[0]});
				tmp -= solutions[j];
				const double resSqr = misc::sqr(frob_norm(tmp));

				testResidual += weights[j] * resSqr;  // HIER
			}
			testResidual = std::sqrt(testResidual)/testNorm;

			return std::make_tuple(optResidual, testResidual, setResiduals);
		}

		double calc_res(const size_t _corePosition, const size_t _setId) const {
			REQUIRE(_corePosition == x.corePosition, "Invalid corePosition");
			REQUIRE(_corePosition == 0, "Invalid corePosition");
			double res = 0.0;
			auto core = x.get_component(_corePosition);
			core.fix_mode(2, _setId);

			Tensor tmp;
			for(const auto j : sets[_setId]) {
				contract(tmp, core, rightStack[1][j], 1);
				tmp.reinterpret_dimensions({x.dimensions[0]});
				tmp -= solutions[j];
				const double resSqr = misc::sqr(frob_norm(tmp));

				res += weights[j] * resSqr;  // HIER
			}
			return std::sqrt(res)/optNorm;
		}

		void update_core(const size_t _corePosition) {
			const Index left, right, ext, p;
			REQUIRE(x.corePosition == _corePosition, "IE");

			for(size_t setId = 0; setId < P; ++setId) {
				const auto delta = calculate_delta(_corePosition, setId);
				const auto normAProjGrad = calculate_norm_A_projGrad(delta, _corePosition, setId);
				const value_t PyR = misc::sqr(frob_norm(delta));

				value_t res = 0.0, new_res = 0.0;
				if (_corePosition == 0) {
					res = calc_res(_corePosition, setId);
				}
				if (PyR > 0) {
					// Actual update
					x.component(_corePosition)(left, ext, p, right) = x.component(_corePosition)(left, ext, p, right) - ((PyR/misc::sqr(normAProjGrad))*delta)(left, ext, right)*Tensor::dirac({P}, setId)(p);
				} else {
					LOG(uqADF, "Warning: vanishing gradient on set " << setId);
				}
				if (_corePosition == 0) {
					new_res = calc_res(_corePosition, setId);
					if (new_res > res) {
						LOG(uqADF, "Warning: relative residuum increased by " << std::scientific << (new_res - res)/res);
					}
				}
			}
		}


		void finish(const size_t _iteration) {
			for(size_t i = 0; i < bestX.order(); i++) {
				if(i == bestX.corePosition) {
					outX.set_component(i, bestX.get_average_core());
				} else {
					outX.set_component(i, bestX.get_component(i));
				}
			}

			LOG(uqADF, "Residual decrease from " << std::scientific << initialResiduum << " to " << std::scientific << residuals.back() << " in " << _iteration << " iterations.");
		}


		void solve() {
			size_t nonImprovementCounter = 0;

			// Build inital right stack
			REQUIRE(x.corePosition == 0, "Expecting core position to be 0.");
			for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
				calc_right_stack(corePosition);
			}

			double optResidual, testResidual;
			std::vector<double> setResiduals;
			std::tie(optResidual, testResidual, setResiduals) = calc_residuals(0);
			initialResiduum = optResidual; /* TODO: inefficient */

			const Index l,e,r,p;
			const Tensor ones = Tensor::ones({P});
			Tensor newCore;
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				std::tie(optResidual, testResidual, setResiduals) = calc_residuals(0);
				residuals.push_back(optResidual);
				prevRanks.push_back(x.ranks());

				if(testResidual < 0.9999*bestTestResidual) {
					bestX = x;
					bestTestResidual = testResidual;
					nonImprovementCounter = 0;
				} else {
					nonImprovementCounter++;
				}


				LOG(uqADF, "Residual " << std::scientific << residuals.back() << " " << setResiduals << ". NonImpCnt: " << nonImprovementCounter << ", Controlset: " << testResidual << ". Ranks: " << x.ranks() << ". DOFs: " << x.dofs());

				if(residuals.back() < targetResidual || nonImprovementCounter >= 100) {
					finish(iteration);
					return;
				}

				if(residuals.back() > convergenceFactor*residuals[0]) {
					bool maxRankReached = false;
					bool rankMaxed = false;
					for(size_t i = 0; i < x.order()-1; ++i) {
						maxRankReached = maxRankReached || (x.rank(i) == maxRank);
						rankMaxed = rankMaxed || (x.rank(i) == prevRanks[0][i]+1);
					}

					if(misc::hard_equal(rankEps, minRankEps) || maxRankReached) {
						finish(iteration);
						return; // We are done!
					}

					if(!rankMaxed) {
						LOG(uqADF, "Reduce rankEps to " << std::max(minRankEps, epsDecay*rankEps));
						rankEps = std::max(minRankEps, epsDecay*rankEps);
					}
				}

				// Forward sweep
				for(size_t corePosition = 0; corePosition+1 < d; ++corePosition) {
					update_core(corePosition);

					x.move_core_right(rankEps, std::min(maxRank, prevRanks[1][corePosition]+1));

					REQUIRE(x.corePosition == corePosition+1, "IE");
					newCore(l,e,p,r) = x.get_average_core()(l,e,r) * ones(p);
					x.set_component(corePosition+1, newCore);

					calc_left_stack(corePosition);
				}

				update_core(d-1);

				// Backward sweep
				for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
					update_core(corePosition);

					x.move_core_left(rankEps, std::min(maxRank, prevRanks[1][corePosition-1]+1));

					REQUIRE(x.corePosition == corePosition-1, "IE");
					newCore(l,e,p,r) = x.get_average_core()(l,e,r) * ones(p);
					x.set_component(corePosition-1, newCore);

					calc_right_stack(corePosition);
				}

				update_core(0);
			}

			finish(maxIterations);
		}
	};
}

	void uq_adf(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_measurments.parameterVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_x.dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");

		impl_uqRaAdf::InternalSolver<1> solver(_x, _measurments, _basisType, _maxItr, _targetEps, 0.0);
		solver.solve();
	}


	void uq_ra_adf(TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps, const size_t _maxItr, const double _initalRankEps) {
		REQUIRE(_measurments.parameterVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_x.dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");

		impl_uqRaAdf::InternalSolver<2> solver(_x, _measurments, _basisType, _maxItr, _targetEps, _initalRankEps);
		solver.solve();
	}


	TTTensor uq_ra_adf(const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_measurments.parameterVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");

		LOG(UQ, "Calculating Average as start.");

		TTTensor x(_dimensions);

		Tensor mean = sample_mean(_measurments.solutions);  /* TODO: weights */

		// Set mean
		mean.reinterpret_dimensions({1, x.dimensions[0], 1});
		x.set_component(0, mean);
		for(size_t k = 1; k < x.order(); ++k) {
			x.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 0));
		}
		x.assume_core_position(0);

		impl_uqRaAdf::InternalSolver<2> solver(x, _measurments, _basisType, _maxItr, _targetEps, 1e-1);
		solver.solve();
		return x;
	}
	

	TTTensor uq_ra_adf(const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_positions.size() == _solutions.size(), "Invalid measurments");
		REQUIRE(_dimensions.front() == _solutions.front().size, "Inconsitent spacial dimension");

		LOG(UQ, "Calculating Average as start.");

		TTTensor x(_dimensions);

		Tensor mean = sample_mean(_solutions);  /* TODO: weights */

		// Set mean
		mean.reinterpret_dimensions({1, x.dimensions[0], 1});
		x.set_component(0, mean);
		for(size_t k = 1; k < x.order(); ++k) {
			x.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 0));
		}
		x.assume_core_position(0);

		impl_uqRaAdf::InternalSolver<2> solver(x, _positions, _solutions, _maxItr, _targetEps, 1e-1);
		solver.solve();
		return x;
	}

	TTTensor uq_ra_adf(const TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_positions.size() == _solutions.size(), "Invalid measurments");
		REQUIRE(_dimensions.front() == _solutions.front().size, "Inconsitent spacial dimension");

		TTTensor x(_x);
		x.canonicalize_left();
		impl_uqRaAdf::InternalSolver<2> solver(x, _positions, _solutions, _maxItr, _targetEps, 1e-1);
		solver.solve();
		return x;
	}
	

	TTTensor uq_ra_adf(const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<double>& _weights, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_positions.size() == _solutions.size(), "Invalid measurments");
		REQUIRE(_dimensions.front() == _solutions.front().size, "Inconsitent spacial dimension");

		LOG(UQ, "Calculating Average as start.");

		TTTensor x(_dimensions);

		Tensor mean = sample_mean(_solutions);  /* TODO: weights */

		// Set mean
		mean.reinterpret_dimensions({1, x.dimensions[0], 1});
		x.set_component(0, mean);
		for(size_t k = 1; k < x.order(); ++k) {
			x.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 0));
		}
		x.assume_core_position(0);

		impl_uqRaAdf::InternalSolver<2> solver(x, _positions, _solutions, _weights, _maxItr, _targetEps, 1e-1);
		solver.solve();
		return x;
	}
	

	TTTensor uq_ra_adf(const TTTensor& _x, const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_measurments.parameterVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_x.dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");

		for(size_t i=0; i < _measurments.parameterVectors.size(); ++i) {
		  REQUIRE(_x.order() <= _measurments.parameterVectors[i].size()+1, "Parameter vector for sample " << i << " to short: " << _measurments.parameterVectors[i]);
		  for(size_t j=0; j < _measurments.parameterVectors[i].size(); ++j) {
			if(_measurments.parameterVectors[i][j] > 1 || _measurments.parameterVectors[i][j] < -1) {
			  std::cout << "i=" << i << ", sample=" << _measurments.parameterVectors[i] << std::endl;
			  REQUIRE(false, "Sample parameter is not -1 <= x <= 1");
			}
		  }
		}

		TTTensor x(_x);
		impl_uqRaAdf::InternalSolver<2> solver(x, _measurments, _basisType, _maxItr, _targetEps, 1e-1);
		solver.solve();
		return x;
	}

}} // namespace  uq | xerus
