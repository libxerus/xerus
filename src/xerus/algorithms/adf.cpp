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

#include <xerus/algorithms/adf.h>

#include <xerus/misc/math.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/internal.h>

#include <xerus/indexedTensorMoveable.h>
#include <xerus/measurments.h>
#include <xerus/ttNetwork.h>
#include <xerus/performanceData.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus {
	template<class MeasurmentSet> 
		class InternalSolver : internal::OptimizationSolver {
			/*
			 * General notation for the ADF:
			 * The vector of the measured values is denoted as b
			 * The measurment operator is denoted as A. It holds by definition A(x) = b (for noiseless measurments).
			 * The Operator constructed from the x by removing the component at corePosition is denoted as E.
			 */
			
		protected:
			///@brief Indices for all internal functions.
			const Index r1, r2, i1;
			
			///@brief Reference to the current solution (external ownership)
			TTTensor& x;
			
			///@brief Degree of the solution.
			const size_t order;
			
			///@brief Maximally allowed ranks.
			const std::vector<size_t> maxRanks;
			
			///@brief Reference to the measurment set (external ownership)
			const MeasurmentSet& measurments;
			
			///@brief Number of measurments (i.e. measurments.size())
			const size_t numMeasurments;
			
			///@brief The two norm of the measured values
			const value_t normMeasuredValues;
			
			///@brief The current residual, saved as vector (instead of a order one tensor).
			std::vector<value_t> residual;
			
			///@brief The current projected Gradient component. That is E(A^T(Ax-b))
			Tensor projectedGradientComponent;
			
			///@brief Ownership holder for a (order+2)*numMeasurments array of Tensor pointers. (Not used directly)
			std::unique_ptr<Tensor*[]> forwardStackMem;
			
			/** @brief Array [numMeasurments][order]. For positions smaller than the current corePosition and for each measurment, this array contains the pre-computed
			* contraction of the first _ component tensors and the first _ components of the measurment operator. These tensors are deduplicated in the sense that for each unqiue
			* part of the position only one tensor is actually stored, which is why the is an array of pointers. The Tensors at the current corePosition are used as 
			* scatch space. For convinience the underlying array (forwardStackMem) is larger, wherefore also the positions -1 and order are allow, all poining to a {1} tensor
			* containing 1 as only entry. Note that position order-1 must not be used.
			**/
			Tensor* const * const forwardStack;
			
			/// @brief Ownership holder for the unqiue Tensors referenced in forwardStack.
			std::unique_ptr<Tensor[]> forwardStackSaveSlots;
			
			/// @brief Vector containing for each corePosition a vector of the smallest ids of each group of unique forwardStack entries.
			std::vector<std::vector<size_t>> forwardUpdates;
			
			
			///@brief Ownership holder for a (order+2)*numMeasurments array of Tensor pointers. (Not used directly)
			std::unique_ptr<Tensor*[]> backwardStackMem;
			
			/** @brief Array [numMeasurments][order]. For positions larger than the current corePosition and for each measurment, this array contains the pre-computed
			* contraction of the last _ component tensors and the last _ components of the measurment operator. These tensors are deduplicated in the sense that for each unqiue
			* part of the position only one tensor is actually stored, which is why the is an array of pointers. The Tensors at the current corePosition are used as 
			* scratch space. For convinience the underlying array (forwardStackMem) is larger, wherefore also the positions -1 and order are allow, all poining to a {1} tensor
			* containing 1 as only entry. Note that position zero must not be used.
			**/
			Tensor* const * const backwardStack;

			/// @brief Ownership holder for the unqiue Tensors referenced in backwardStack.
			std::unique_ptr<Tensor[]> backwardStackSaveSlots;
			
			/// @brief Vector containing for each corePosition a vector of the smallest ids of each group of unique backwardStack entries.
			std::vector<std::vector<size_t>> backwardUpdates;
			
			/// @brief: Norm of each rank one measurment operator
			std::unique_ptr<double[]> measurmentNorms;
			
			///@brief calculates the two-norm of the measured values.
			static double calculate_norm_of_measured_values(const MeasurmentSet& _measurments);
			
		public:
			InternalSolver(	const OptimizationAlgorithm& _optiAlgorithm, TTTensor& _x, const std::vector<size_t>& _maxRanks, const MeasurmentSet& _measurments, PerformanceData& _perfData);
			
		protected:
			///@brief Constructes either the forward or backward stack. That is, it determines the groups of partially equale measurments. Therby stetting (forward/backward)- Updates, StackMem and SaveSlot.
			void construct_stacks(std::unique_ptr< xerus::Tensor[] >& _stackSaveSlot, std::vector< std::vector< size_t > >& _updates, const std::unique_ptr<Tensor*[]>& _stackMem, const bool _forward);
			
			///@brief Resizes the unqiue stack tensors to correspond to the current ranks of x.
			void resize_stack_tensors();
			
			///@brief Returns a vector of tensors containing the slices of @a _component where the second dimension is fixed.
			std::vector<Tensor> get_fixed_components(const Tensor& _component);
			
			///@brief For each measurment sets the forwardStack at the given _corePosition to the contraction between the forwardStack at the previous corePosition (i.e. -1)
			/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
			/// this really updates the stack, otherwise it uses the stack as scratch space.
			void update_forward_stack(const size_t _corePosition, const Tensor& _currentComponent);
			
			///@brief For each measurment sets the backwardStack at the given _corePosition to the contraction between the backwardStack at the previous corePosition (i.e. +1)
			/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
			/// this really updates the stack, otherwise it uses the stack as scratch space.
			void update_backward_stack(const size_t _corePosition, const Tensor& _currentComponent);
			
			///@brief (Re-)Calculates the current residual, i.e. Ax-b.
			void calculate_residual( const size_t _corePosition );
			
			///@brief Calculates one internal step of calculate_projected_gradient. In particular the dyadic product of the leftStack, the rightStack and the position vector.
			template<class PositionType>
			void perform_dyadic_product(const size_t _localLeftRank, const size_t _localRightRank, const value_t* const _leftPtr,  const value_t* const _rightPtr,  value_t* const _deltaPtr, const value_t _residual, const PositionType& _position, value_t* const _scratchSpace );
	
			
			///@brief: Calculates the component at _corePosition of the projected gradient from the residual, i.e. E(A^T(b-Ax)).
			void calculate_projected_gradient(const size_t _corePosition);
			
			/**
			* @brief: Calculates ||P_n (A(E(A^T(b-Ax)))))|| = ||P_n (A(E(A^T(residual)))))|| =  ||P_n (A(E(gradient)))|| for each n, 
			* where P_n sets all entries equals zero except where the index at _corePosition is equals n. In case of RankOneMeasurments,
			* the calculation is not slicewise (only n=0 is set).
			*/
			std::vector<value_t> calculate_slicewise_norm_A_projGrad( const size_t _corePosition);
			
			///@brief Updates the current solution x. For SinglePointMeasurments the is done for each slice speratly, for RankOneMeasurments there is only one combined update.
			void update_x(const std::vector<value_t>& _normAProjGrad, const size_t _corePosition);
			
			///@brief Basically the complete algorithm, trying to reconstruct x using its current ranks.
			void solve_with_current_ranks();
			
		public:
			///@brief Tries to solve the reconstruction problem with the current settings.
			double solve();
			
		};
	
	template<class MeasurmentSet>
	InternalSolver<MeasurmentSet>::InternalSolver(
				const OptimizationAlgorithm& _optiAlgorithm,
				TTTensor& _x,
				const std::vector<size_t>& _maxRanks,
				const MeasurmentSet& _measurments,
				PerformanceData& _perfData ) : 
		OptimizationSolver(_optiAlgorithm, _perfData),
		x(_x),
		order(_x.order()),
		maxRanks(TTTensor::reduce_to_maximal_ranks(_maxRanks, _x.dimensions)),
		
		measurments(_measurments),
		numMeasurments(_measurments.size()),
		normMeasuredValues(calculate_norm_of_measured_values(_measurments)),
		
		residual(numMeasurments),
		
		forwardStackMem(new Tensor*[numMeasurments*(order+2)]),
		forwardStack(forwardStackMem.get()+numMeasurments),
		forwardUpdates(order),
			
		backwardStackMem(new Tensor*[numMeasurments*(order+2)]),
		backwardStack(backwardStackMem.get()+numMeasurments),
		backwardUpdates(order),
		
		measurmentNorms(new double[numMeasurments])
	{
		_x.require_correct_format();
		XERUS_REQUIRE(numMeasurments > 0, "Need at very least one measurment.");
		XERUS_REQUIRE(measurments.order() == order, "Measurment order must coincide with x order.");
	}
	
	
	
	template<class MeasurmentSet>
	double InternalSolver<MeasurmentSet>::calculate_norm_of_measured_values(const MeasurmentSet& _measurments) {
		value_t normMeasuredValues = 0;
		if (_measurments.weights.size() == 0) {
			for(const value_t measurement : _measurments.measuredValues) {
				normMeasuredValues += misc::sqr(measurement);
			}
		} else {
			for(size_t i=0; i<_measurments.size(); i++) {
				normMeasuredValues += _measurments.weights[i] * misc::sqr(_measurments.measuredValues[i]);
			}
		}
		return std::sqrt(normMeasuredValues);
	}


	template<class MeasurmentSet>
	class MeasurmentComparator {
		const bool forward;
		const size_t order;
		const MeasurmentSet& measurments;
	public:
		MeasurmentComparator(const MeasurmentSet& _measurments, const bool _forward);

		bool operator()(const size_t _a, const size_t _b) const;
	};

	template<>
	MeasurmentComparator<SinglePointMeasurementSet>::MeasurmentComparator(const SinglePointMeasurementSet& _measurments, const bool _forward) : forward(_forward), order(_measurments.order()), measurments(_measurments) { }

	template<>
	bool MeasurmentComparator<SinglePointMeasurementSet>::operator()(const size_t _a, const size_t _b) const {
		if(forward) {
			for (size_t j = 0; j < order; ++j) {
				if (measurments.positions[_a][j] < measurments.positions[_b][j]) { return true; }
				if (measurments.positions[_a][j] > measurments.positions[_b][j]) { return false; }
			}
		} else {
			for (size_t j = order; j > 0; --j) {
				if (measurments.positions[_a][j-1] < measurments.positions[_b][j-1]) { return true; }
				if (measurments.positions[_a][j-1] > measurments.positions[_b][j-1]) { return false; }
			}
		}
		LOG(fatal, "Measurments must not appear twice."); // NOTE that the algorithm works fine even if measurements appear twice.
		return false;
	}


	template<>
	MeasurmentComparator<RankOneMeasurementSet>::MeasurmentComparator(const RankOneMeasurementSet& _measurments, const bool _forward) : forward(_forward), order(_measurments.order()), measurments(_measurments) { }

	template<>
	bool MeasurmentComparator<RankOneMeasurementSet>::operator()(const size_t _a, const size_t _b) const {
		if(forward) {
			for (size_t j = 0; j < order; ++j) {
				const int res = internal::compare(measurments.positions[_a][j], measurments.positions[_b][j]);
				if(res == -1) { return true; }
				if(res == 1) { return false; }
			}
		} else {
			for (size_t j = order; j > 0; --j) {
				const int res = internal::compare(measurments.positions[_a][j-1], measurments.positions[_b][j-1]);
				if(res == -1) { return true; }
				if(res == 1) { return false; }
			}
		}

		/* LOG(fatal, "Measurments must not appear twice. "); // NOTE that the algorithm works fine even if measurements appear twice. */
		return false;
	}


	template<class MeasurmentSet>
	void InternalSolver<MeasurmentSet>::construct_stacks(std::unique_ptr<Tensor[]>& _stackSaveSlot, std::vector<std::vector<size_t>>& _updates, const std::unique_ptr<Tensor*[]>& _stackMem, const bool _forward) {
		using misc::approx_equal;

		// Direct reference to the stack (withou Mem)
		Tensor** const stack(_stackMem.get()+numMeasurments);

		// Temporary map. For each stack entry (i.e. measurement number + corePosition) gives a measurement number of a stack entry (at same corePosition) that shall have an equal value (or its own number otherwise).
		std::vector<size_t> calculationMap(order*numMeasurments);

		// Count how many Tensors we need for the stacks
		size_t numUniqueStackEntries = 0;

		// Create a reordering map
		perfData << "Start sorting";
		std::vector<size_t> reorderedMeasurments(numMeasurments);
		std::iota(reorderedMeasurments.begin(), reorderedMeasurments.end(), 0);
		std::sort(reorderedMeasurments.begin(), reorderedMeasurments.end(), MeasurmentComparator<MeasurmentSet>(measurments, _forward));
		perfData << "End sorting " << _forward ;

		// Create the entries for the first measurement (these are allways unqiue).
		for(size_t corePosition = 0; corePosition < order; ++corePosition) {
			const size_t realId = reorderedMeasurments[0];
			calculationMap[realId + corePosition*numMeasurments] = realId;
			++numUniqueStackEntries;
		}

		// Create the calculation map
		for(size_t i = 1; i < numMeasurments; ++i) {
			const size_t realId = reorderedMeasurments[i];
			const size_t realPreviousId = reorderedMeasurments[i-1];

			size_t position = 0;
			size_t corePosition = _forward ? position : order-1-position;

			for( ;
				position < order && approx_equal(measurments.positions[realId][corePosition], measurments.positions[realPreviousId][corePosition]);
				++position, corePosition = _forward ? position : order-1-position)
			{
				if( realPreviousId < realId ) {
					calculationMap[realId + corePosition*numMeasurments] = calculationMap[realPreviousId + corePosition*numMeasurments];
				} else if(realPreviousId == calculationMap[realPreviousId + corePosition*numMeasurments]) {
					calculationMap[realPreviousId + corePosition*numMeasurments] = realId;
					calculationMap[realId + corePosition*numMeasurments] = realId;
				} else if( realId < calculationMap[realPreviousId + corePosition*numMeasurments]) {
					const size_t nextOther = calculationMap[realPreviousId + corePosition*numMeasurments];
					INTERNAL_CHECK(calculationMap[nextOther + corePosition*numMeasurments] == nextOther, "IE");
					calculationMap[realPreviousId + corePosition*numMeasurments] = realId;
					calculationMap[nextOther + corePosition*numMeasurments] = realId;
					calculationMap[realId + corePosition*numMeasurments] = realId;
				} else {
					calculationMap[realId + corePosition*numMeasurments] = calculationMap[realPreviousId + corePosition*numMeasurments];
				}
			}

			for( ; position < order; ++position, corePosition = _forward ? position : order-1-position) {
				calculationMap[realId + corePosition*numMeasurments] = realId;
				++numUniqueStackEntries;
			}
		}

		// Create the stack
		numUniqueStackEntries++; // +1 for the special positions -1 and order.
		_stackSaveSlot.reset(new Tensor[numUniqueStackEntries]);
		size_t usedSlots = 0;
		_stackSaveSlot[usedSlots++] = Tensor::ones({1}); // Special slot reserved for the the position -1 and order stacks

		// NOTE that _stackMem contains (order+2)*numMeasurments entries and has an offset of numMeasurments (to have space for corePosition -1).

		// Set links for the special entries -1 and order
		for(size_t i = 0; i < numMeasurments; ++i) {
			stack[i - 1*numMeasurments] = &_stackSaveSlot[0];
			stack[i + order*numMeasurments] = &_stackSaveSlot[0];
		}

		for(size_t corePosition = 0; corePosition < order; ++corePosition) {
			for(size_t i = 0; i < numMeasurments; ++i) {
				if(calculationMap[i + corePosition*numMeasurments] == i) {
					_updates[corePosition].emplace_back(i);
					stack[i + corePosition*numMeasurments] = &_stackSaveSlot[usedSlots];
					usedSlots++;
				} else {
					stack[i + corePosition*numMeasurments] = stack[calculationMap[i + corePosition*numMeasurments] + corePosition*numMeasurments];
				}
			}
		}

		INTERNAL_CHECK(usedSlots == numUniqueStackEntries, "Internal Error.");
		perfData << "We have " << numUniqueStackEntries << " unique stack entries. There are " << numMeasurments*order+1 << " virtual stack entries.";
	}

	template<class MeasurmentSet>
	void InternalSolver<MeasurmentSet>::resize_stack_tensors() {
		#pragma omp parallel for schedule(static)
		for(size_t corePosition = 0; corePosition < order; ++corePosition) {
			for(const size_t i : forwardUpdates[corePosition]) {
				forwardStack[i + corePosition*numMeasurments]->reset({corePosition+1 == order ? 1 : x.rank(corePosition)}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			}
			for(const size_t i : backwardUpdates[corePosition]) {
				backwardStack[i + corePosition*numMeasurments]->reset({corePosition == 0 ? 1 :x.rank(corePosition - 1)}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			}
		}
	}

	template<class MeasurmentSet>
	std::vector<Tensor> InternalSolver<MeasurmentSet>::get_fixed_components(const Tensor& _component) {
		std::vector<Tensor> fixedComponents(_component.dimensions[1]);

		for(size_t i = 0; i < _component.dimensions[1]; ++i) {
			fixedComponents[i](r1, r2) = _component(r1, i, r2);
		}

		return fixedComponents;
	}

	template<>
	void InternalSolver<SinglePointMeasurementSet>::update_backward_stack(const size_t _corePosition, const Tensor& _currentComponent) {
		INTERNAL_CHECK(_currentComponent.dimensions[1] == x.dimensions[_corePosition], "IE");

		const size_t numUpdates = backwardUpdates[_corePosition].size();

		std::vector<Tensor> fixedComponents = get_fixed_components(_currentComponent);

		// Update the stack
		#pragma omp parallel for schedule(static)
		for(size_t u = 0; u < numUpdates; ++u) {
			const size_t i = backwardUpdates[_corePosition][u];
			contract(*backwardStack[i + _corePosition*numMeasurments], fixedComponents[measurments.positions[i][_corePosition]], false, *backwardStack[i + (_corePosition+1)*numMeasurments], false, 1);
		}
	}

	template<>
	void InternalSolver<RankOneMeasurementSet>::update_backward_stack(const size_t _corePosition, const Tensor& _currentComponent) {
		INTERNAL_CHECK(_currentComponent.dimensions[1] == x.dimensions[_corePosition], "IE");

		const size_t numUpdates = backwardUpdates[_corePosition].size();

		Tensor reshuffledComponent;
		reshuffledComponent(i1, r1, r2) =  _currentComponent(r1, i1, r2);

		Tensor mixedComponent({reshuffledComponent.dimensions[1], reshuffledComponent.dimensions[2]});

		// Update the stack
		#pragma omp parallel for firstprivate(mixedComponent) schedule(static)
		for(size_t u = 0; u < numUpdates; ++u) {
			const size_t i = backwardUpdates[_corePosition][u];
			contract(mixedComponent, measurments.positions[i][_corePosition], false, reshuffledComponent, false, 1);
			contract(*backwardStack[i + _corePosition*numMeasurments], mixedComponent, false, *backwardStack[i + (_corePosition+1)*numMeasurments], false, 1);
		}
	}


	template<>
	void InternalSolver<SinglePointMeasurementSet>::update_forward_stack( const size_t _corePosition, const Tensor& _currentComponent ) {
		INTERNAL_CHECK(_currentComponent.dimensions[1] == x.dimensions[_corePosition], "IE");

		const size_t numUpdates = forwardUpdates[_corePosition].size();

		const std::vector<Tensor> fixedComponents = get_fixed_components(_currentComponent);

		// Update the stack
		#pragma omp parallel for schedule(static)
		for(size_t u = 0; u < numUpdates; ++u) {
			const size_t i = forwardUpdates[_corePosition][u];
			contract(*forwardStack[i + _corePosition*numMeasurments] , *forwardStack[i + (_corePosition-1)*numMeasurments], false, fixedComponents[measurments.positions[i][_corePosition]], false, 1);
		}
	}

	template<>
	void InternalSolver<RankOneMeasurementSet>::update_forward_stack( const size_t _corePosition, const Tensor& _currentComponent ) {
		INTERNAL_CHECK(_currentComponent.dimensions[1] == x.dimensions[_corePosition], "IE");

		const size_t numUpdates = forwardUpdates[_corePosition].size();

		Tensor reshuffledComponent;
		reshuffledComponent(i1, r1, r2) =  _currentComponent(r1, i1, r2);

		Tensor mixedComponent({reshuffledComponent.dimensions[1], reshuffledComponent.dimensions[2]});

		// Update the stack
		#pragma omp parallel for firstprivate(mixedComponent) schedule(static)
		for(size_t u = 0; u < numUpdates; ++u) {
			const size_t i = forwardUpdates[_corePosition][u];
			contract(mixedComponent, measurments.positions[i][_corePosition], false, reshuffledComponent, false, 1);
			contract(*forwardStack[i + _corePosition*numMeasurments] , *forwardStack[i + (_corePosition-1)*numMeasurments], false, mixedComponent, false, 1);
		}
	}

	template<class MeasurmentSet>
	void InternalSolver<MeasurmentSet>::calculate_residual( const size_t _corePosition ) {
		Tensor currentValue({});

		// Look which side of the stack needs less calculations
		if(forwardUpdates[_corePosition].size() < backwardUpdates[_corePosition].size()) {
			update_forward_stack(_corePosition, x.get_component(_corePosition));

			if (measurments.weights.size() == 0) {
				#pragma omp parallel for firstprivate(currentValue) schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + _corePosition*numMeasurments], false, *backwardStack[i + (_corePosition+1)*numMeasurments], false, 1);
					residual[i] = (measurments.measuredValues[i]-currentValue[0]);
				}
			} else {
				#pragma omp parallel for firstprivate(currentValue) schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + _corePosition*numMeasurments], false, *backwardStack[i + (_corePosition+1)*numMeasurments], false, 1);
					residual[i] = measurments.weights[i] * (measurments.measuredValues[i]-currentValue[0]);
				}
			}
		} else {
			update_backward_stack(_corePosition, x.get_component(_corePosition));

			if (measurments.weights.size() == 0) {
				#pragma omp parallel for firstprivate(currentValue) schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + (_corePosition-1)*numMeasurments], false, *backwardStack[i + _corePosition*numMeasurments], false, 1);
					residual[i] = (measurments.measuredValues[i]-currentValue[0]);
				}
			} else {
				#pragma omp parallel for firstprivate(currentValue) schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + (_corePosition-1)*numMeasurments], false, *backwardStack[i + _corePosition*numMeasurments], false, 1);
					residual[i] = measurments.weights[i] * (measurments.measuredValues[i]-currentValue[0]);
				}
			}
		}
	}

	template<> template<>
	inline void InternalSolver<SinglePointMeasurementSet>::perform_dyadic_product(  const size_t _localLeftRank,
																					const size_t _localRightRank,
																					const value_t* const _leftPtr,
																					const value_t* const _rightPtr,
																					value_t* const _deltaPtr,
																					const value_t _residual,
																					const size_t& _position,
																					value_t* const
																				 /*unused*/) {
		value_t* const shiftedDeltaPtr = _deltaPtr + _position*_localLeftRank*_localRightRank;

		for(size_t k = 0; k < _localLeftRank; ++k) {
			for(size_t j = 0; j < _localRightRank; ++j) {
				shiftedDeltaPtr[k*_localRightRank+j] += _residual * _leftPtr[k] * _rightPtr[j];
			}
		}
	}

	template<> template<>
	inline void InternalSolver<RankOneMeasurementSet>::perform_dyadic_product(  const size_t _localLeftRank,
																					const size_t _localRightRank,
																					const value_t* const _leftPtr,
																					const value_t* const _rightPtr,
																					value_t* const _deltaPtr,
																					const value_t _residual,
																					const Tensor& _position,
																					value_t* const _scratchSpace
																				) {
		// Create dyadic product without factors in scratch space
		for(size_t k = 0; k < _localLeftRank; ++k) {
			for(size_t j = 0; j < _localRightRank; ++j) {
				_scratchSpace[k*_localRightRank+j] = _leftPtr[k] * _rightPtr[j];
			}
		}

		for(size_t n = 0; n < _position.size; ++n) {
			misc::add_scaled(_deltaPtr + n*_localLeftRank*_localRightRank,
				_position[n]*_residual,
				_scratchSpace,
				_localLeftRank*_localRightRank
			);
		}
	}

	template<class MeasurmentSet>
	inline void InternalSolver<MeasurmentSet>::calculate_projected_gradient( const size_t _corePosition ) {
		const size_t localLeftRank = x.get_component(_corePosition).dimensions[0];
		const size_t localRightRank = x.get_component(_corePosition).dimensions[2];

		projectedGradientComponent.reset({x.dimensions[_corePosition], localLeftRank, localRightRank});

		#pragma omp parallel
		{
			Tensor partialProjGradComp({x.dimensions[_corePosition], localLeftRank, localRightRank}, Tensor::Representation::Dense);

			std::unique_ptr<value_t[]> dyadicComponent(std::is_same<MeasurmentSet, RankOneMeasurementSet>::value ? new value_t[localLeftRank*localRightRank] : nullptr);

			#pragma omp for schedule(static)
			for(size_t i = 0; i < numMeasurments; ++i) {
				INTERNAL_CHECK(!forwardStack[i + (_corePosition-1)*numMeasurments]->has_factor() && !backwardStack[i + (_corePosition+1)*numMeasurments]->has_factor(), "IE");

				// Interestingly writing a dyadic product on our own turns out to be faster than blas...
				perform_dyadic_product( localLeftRank,
										localRightRank,
										forwardStack[i + (_corePosition-1)*numMeasurments]->get_dense_data(),
										backwardStack[i + (_corePosition+1)*numMeasurments]->get_dense_data(),
										partialProjGradComp.get_unsanitized_dense_data(),
										residual[i],
										measurments.positions[i][_corePosition],
										dyadicComponent.get()
									);
			}

			// Accumulate the partical components
			#pragma omp critical
			{
				projectedGradientComponent += partialProjGradComp;
			}
		}

		projectedGradientComponent(r1, i1, r2) = projectedGradientComponent(i1, r1, r2);
	}

	template<class MeasurmentSet>
	inline size_t position_or_zero(const MeasurmentSet& _measurments, const size_t _meas, const size_t _corePosition);

	template<>
	inline size_t position_or_zero<SinglePointMeasurementSet>(const SinglePointMeasurementSet& _measurments, const size_t _meas, const size_t _corePosition) {
		return _measurments.positions[_meas][_corePosition];
	}

	template<>
	inline size_t position_or_zero<RankOneMeasurementSet>(const RankOneMeasurementSet&  /*_measurments*/, const size_t  /*_meas*/, const size_t  /*_corePosition*/) {
		return 0;
	}



	template<class MeasurmentSet>
	std::vector<value_t> InternalSolver<MeasurmentSet>::calculate_slicewise_norm_A_projGrad( const size_t _corePosition) {
		std::vector<value_t> normAProjGrad(x.dimensions[_corePosition], 0.0);

		Tensor currentValue({});

		// Look which side of the stack needs less calculations
		if(forwardUpdates[_corePosition].size() < backwardUpdates[_corePosition].size()) {
			update_forward_stack(_corePosition, projectedGradientComponent);

			#pragma omp parallel firstprivate(currentValue)
			{
				std::vector<value_t> partialNormAProjGrad(x.dimensions[_corePosition], 0.0);

				#pragma omp for schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + _corePosition*numMeasurments], false, *backwardStack[i + (_corePosition+1)*numMeasurments], false, 1);
					partialNormAProjGrad[position_or_zero(measurments, i, _corePosition)] += misc::sqr(currentValue[0]/**measurmentNorms[i]*/); // TODO measurmentNorms
				}

				// Accumulate the partical components
				#pragma omp critical
				{
					for(size_t i = 0; i < normAProjGrad.size(); ++i) {
						normAProjGrad[i] += partialNormAProjGrad[i];
					}
				}
			}
		} else {
			update_backward_stack(_corePosition, projectedGradientComponent);

			#pragma omp parallel firstprivate(currentValue)
			{
				std::vector<value_t> partialNormAProjGrad(x.dimensions[_corePosition], 0.0);

				#pragma omp for schedule(static)
				for(size_t i = 0; i < numMeasurments; ++i) {
					contract(currentValue, *forwardStack[i + (_corePosition-1)*numMeasurments], false, *backwardStack[i + _corePosition*numMeasurments], false, 1);
					partialNormAProjGrad[position_or_zero(measurments, i, _corePosition)] += misc::sqr(currentValue[0]/**measurmentNorms[i]*/); // TODO measurmentNorms
				}

				// Accumulate the partical components
				#pragma omp critical
				{
					for(size_t i = 0; i < normAProjGrad.size(); ++i) {
						normAProjGrad[i] += partialNormAProjGrad[i];
					}
				}
			}
		}

		return normAProjGrad;
	}


	template<>
	void InternalSolver<SinglePointMeasurementSet>::update_x(const std::vector<value_t>& _normAProjGrad, const size_t _corePosition) {
		for(size_t j = 0; j < x.dimensions[_corePosition]; ++j) {
			Tensor localDelta;
			localDelta(r1, r2) = projectedGradientComponent(r1, j, r2);
			const value_t PyR = misc::sqr(frob_norm(localDelta));

			// Update
			x.component(_corePosition)(r1, i1, r2) = x.component(_corePosition)(r1, i1, r2) + (PyR/_normAProjGrad[j])*Tensor::dirac({x.dimensions[_corePosition]}, j)(i1)*localDelta(r1, r2);
		}
	}


	template<>
	void InternalSolver<RankOneMeasurementSet>::update_x(const std::vector<value_t>& _normAProjGrad, const size_t _corePosition) {
		const value_t PyR = misc::sqr(frob_norm(projectedGradientComponent));

		// Update
		x.component(_corePosition)(r1, i1, r2) = x.component(_corePosition)(r1, i1, r2) + (PyR/misc::sum(_normAProjGrad))*projectedGradientComponent(r1, i1, r2);
	}

	template<class MeasurmentSet>
	void InternalSolver<MeasurmentSet>::solve_with_current_ranks() {
		reset_convergence_buffer();
		while(true) {
			// Move core back to position zero
			x.move_core(0, true);

			// Rebuild backwardStack
			for(size_t corePosition = x.order()-1; corePosition > 0; --corePosition) {
				update_backward_stack(corePosition, x.get_component(corePosition));
			}

			calculate_residual(0);

			double residualNormSqr = 0;

			#pragma omp parallel for schedule(static) reduction(+:residualNormSqr)
			for(size_t i = 0; i < numMeasurments; ++i) {
				residualNormSqr += misc::sqr(residual[i]);
			}
			const double residualNorm = std::sqrt(residualNormSqr)/normMeasuredValues;
			
			make_step(residualNorm);
			perfData.add(current_iteration(), residualNorm, x, 0);

			if(reached_stopping_criteria() || reached_convergence_criteria()) { break; }

			// Sweep from the first to the last component
			for(size_t corePosition = 0; corePosition < order; ++corePosition) {
				if(corePosition > 0) { // For corePosition 0 this calculation is allready done in the calculation of the residual.
					calculate_residual(corePosition);
				}

				calculate_projected_gradient(corePosition);

				const std::vector<value_t> normAProjGrad = calculate_slicewise_norm_A_projGrad(corePosition);

				update_x(normAProjGrad, corePosition);

				// If we have not yet reached the end of the sweep we need to take care of the core and update our stacks
				if(corePosition+1 < order) {
					x.move_core(corePosition+1, true);
					update_forward_stack(corePosition, x.get_component(corePosition));
				}
			}
		}
	}


	template<class MeasurmentSet>
	inline void calc_measurment_norm(double* _norms, const MeasurmentSet& _measurments);

	template<>
	inline void calc_measurment_norm<SinglePointMeasurementSet>(double* _norms, const SinglePointMeasurementSet& _measurments) {
		for(size_t i = 0; i < _measurments.size(); ++i) {
			_norms[i] = 1.0;
		}
	}

	template<>
	inline void calc_measurment_norm<RankOneMeasurementSet>(double* _norms, const RankOneMeasurementSet& _measurments) {
		for(size_t i = 0; i < _measurments.size(); ++i) {
			_norms[i] = 1.0;
			for(size_t j = 0; j < _measurments.order(); ++j) {
				_norms[i] *= _measurments.positions[i][j].frob_norm();
			}
		}
	}


	template<class MeasurmentSet>
	double InternalSolver<MeasurmentSet>::solve() {
		perfData.start();

		#pragma omp parallel sections
		{
			#pragma omp section
				construct_stacks(forwardStackSaveSlots, forwardUpdates, forwardStackMem, true);

			#pragma omp section
				construct_stacks(backwardStackSaveSlots, backwardUpdates, backwardStackMem, false);
		}

		calc_measurment_norm(measurmentNorms.get(), measurments);

		// We need x to be canonicalized in the sense that there is no edge with more than maximal rank (prior to stack resize).
		x.canonicalize_left();

		resize_stack_tensors();

		// One inital run
		solve_with_current_ranks();

		// If we follow a rank increasing strategie, increase the ransk until we reach the targetResidual, the maxRanks or the maxIterations.
		while( !reached_stopping_criteria() && x.ranks() != maxRanks ) {
			
			// Increase the ranks
			x.move_core(0, true);
			const auto rndTensor = TTTensor::random(x.dimensions, std::vector<size_t>(x.order()-1, 1));
			const auto diff = (1e-5*frob_norm(x))*rndTensor/frob_norm(rndTensor);
			x = x+diff;

			x.round(maxRanks);

			resize_stack_tensors();

			solve_with_current_ranks();
		}
		return current_residual();
	}
	
	// Explicit instantiation of the two template parameters that will be implemented in the xerus library
	template class InternalSolver<SinglePointMeasurementSet>;
	template class InternalSolver<RankOneMeasurementSet>;

	
	
	ADFVariant::ADFVariant(const size_t _maxIteration, const double _targetRelativeResidual, const double _minimalResidualDecrease)
		: OptimizationAlgorithm(0, _maxIteration, _targetRelativeResidual, _minimalResidualDecrease) { }

		
	template<class MeasurmentSet>
	double ADFVariant::operator()(TTTensor& _x, const MeasurmentSet& _measurments, PerformanceData& _perfData) const {
		InternalSolver<MeasurmentSet> solver(*this, _x, _x.ranks(), _measurments, _perfData);
		return solver.solve();
	}
	
	template double ADFVariant::operator()(TTTensor& _x, const SinglePointMeasurementSet& _measurments, PerformanceData& _perfData) const;
	template double ADFVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, PerformanceData& _perfData) const;
	

	template<class MeasurmentSet>
	double ADFVariant::operator()(TTTensor& _x, const MeasurmentSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const {
		InternalSolver<MeasurmentSet> solver(*this, _x, _maxRanks, _measurments, _perfData);
		return solver.solve();
	}
	
	template double ADFVariant::operator()(TTTensor& _x, const SinglePointMeasurementSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const;
	template double ADFVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const;

	
	const ADFVariant ADF(0, 1e-8, 0.9995);
} // namespace xerus
