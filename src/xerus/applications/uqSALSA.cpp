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

#include <xerus/applications/uqSALSA.h>
#include <xerus/misc/check.h>

#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <numeric>


#define REVERSE_ACCESS(vector, index) vector[vector.size()-1-(index)]

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

namespace xerus { namespace uq {

	template<typename ... Args>
	std::string string_format( const std::string& format, Args ... args )
	{
		size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
		REQUIRE(size > 0, "Error during formatting.");
		std::unique_ptr<char[]> buf( new char[ size ] );
		snprintf( buf.get(), size, format.c_str(), args ... );
		return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
	}

	template<typename T>
	std::string print_list(const std::vector<T>& _list) {
		std::ostringstream stream;
		for (size_t i=0; i<_list.size(); ++i) {
			stream << std::to_string(_list[i]) << " ";
		}
		std::string output = "[" + stream.str();
		output.replace(output.length()-1, 1, "]");
		return output;
	}

	template<typename T>
	std::string print_list(const std::vector<T>& _list, const std::function<std::string(const T)>& _formatter) {
		std::ostringstream stream;
		for (size_t i=0; i<_list.size(); ++i) {
			stream << _formatter(_list[i]) << " ";
		}
		std::string output = "[" + stream.str();
		output.replace(output.length()-1, 1, "]");
		return output;
	}

	std::string print_list(const size_t _size, const std::function<std::string(const size_t)>& _formatter) {
		std::ostringstream stream;
		for (size_t i=0; i<_size; ++i) {
			stream << _formatter(i) << " ";
		}
		std::string output = "[" + stream.str();
		output.replace(output.length()-1, 1, "]");
		return output;
	}

	std::vector<size_t> compute_max_theoretic_ranks(const std::vector<size_t>& _dimensions) {
		// np.minimum(np.cumprod(d[:-1]), np.cumprod(d[:0:-1])[::-1])
		const size_t M = _dimensions.size();
		std::vector<size_t> cumprod_left(M-1);
		/* std::inclusive_scan(_dimensions.begin(), _dimensions.end()-1, */
		/*                     cumprod_left.begin(), std::multiplies<>{}); */
		cumprod_left[0] = _dimensions[0];
		for (size_t i=1; i<M-1; ++i) {
			cumprod_left[i] = cumprod_left[i-1] * _dimensions[i];
		}

		std::vector<size_t> cumprod_right(M-1);
		/* std::inclusive_scan(_dimensions.rbegin(), _dimensions.rend()-1, */
		/*                     cumprod_right.rbegin(), std::multiplies<>{}); */
		REVERSE_ACCESS(cumprod_right, 0) = REVERSE_ACCESS(_dimensions, 0);
		for (size_t i=1; i<_dimensions.size()-1; ++i) {
			REVERSE_ACCESS(cumprod_right, i) = REVERSE_ACCESS(cumprod_right, i-1) * REVERSE_ACCESS(_dimensions, i);
		}

		std::vector<size_t> ranks;
		std::transform(cumprod_left.begin(), cumprod_left.end(),
						cumprod_right.begin(), std::back_inserter(ranks),
						[](size_t a, size_t b) {return std::min(a,b);});

		return ranks;
	}

	std::vector<Tensor> ones(std::vector<size_t>& _dimensions) {
		std::vector<Tensor> ret;
		std::transform(_dimensions.begin(), _dimensions.end(), std::back_inserter(ret), [](size_t d) { return Tensor::identity({d,d}); });
		return ret;
	}

	template<typename T>
	void reorder(std::vector<T>& _sequence, const std::vector<size_t>& _order) {
		REQUIRE(_sequence.size() == _order.size(), "...");
		std::vector<T> tmp(_sequence.size());
		for (size_t i=0; i<_sequence.size(); ++i) {
			tmp[i] = _sequence[_order[i]];
		}
		_sequence = tmp;
	}

	std::vector<Tensor> slice_tensor(const Tensor& _tensor) {
		REQUIRE(_tensor.order() == 2, "...");
		Tensor::DimensionTuple dim(1);
		dim[0] = _tensor.dimensions[1];
		std::vector<Tensor> ret(_tensor.dimensions[0], Tensor(dim));

		for (size_t i=0; i<_tensor.dimensions[0]; ++i) {
			for (size_t j=0; j<_tensor.dimensions[1]; ++j) {
				ret[i][j] = _tensor[{i,j}];
			}
		}

		return ret;
	}

	Tensor reinterpret_dimensions(const Tensor& _tensor, const Tensor::DimensionTuple& _dimensions) {
		Tensor ret(_tensor);
		ret.reinterpret_dimensions(_dimensions);
		return ret;
	}

	Tensor diag(const Tensor& _entries, const std::function<value_t(value_t)>& _modifier) {
		Tensor::DimensionTuple dimensions(_entries.dimensions);
		dimensions.insert(dimensions.end(), _entries.dimensions.begin(), _entries.dimensions.end());
		Tensor ret({_entries.size, _entries.size});
		for (size_t i=0; i<_entries.size; ++i) {
			ret[{i,i}] = _modifier(_entries[i]);
		}
		ret.reinterpret_dimensions(dimensions);
		return ret;
	}

	double max_norm(const Tensor& _tensor) {
		double ret = 0.0;
		for (size_t i=0; i<_tensor.size; ++i) {
			ret = std::max(ret, std::abs(_tensor[i]));
		}
		return ret;
	}

	SALSA::SALSA(const TTTensor& _x, const std::vector<Tensor>& _measures, const Tensor& _values) :
		x(_x),
		measures(_measures.size()+1),

		M(x.dimensions.size()),
		N(_values.dimensions.at(0)),
		P(_values.dimensions.at(1)),

		leftLHSStack(M, std::vector<Tensor>(N)),
		leftRHSStack(M, std::vector<Tensor>(N)),
		rightStack(M, std::vector<Tensor>(N)),
		leftRegularizationStack(M),
		rightRegularizationStack(M),

		singularValues(M-1),
		weightedNorms(M),

		maxIRstepsReached(M),

		maxRanks(M-1, std::numeric_limits<size_t>::max()),
		basisWeights(ones(x.dimensions))
		/* weights(N, 1.0), */
	{
		LOG(debug, "Entering SALSA()");
		REQUIRE(M > 1, "Problem needs at least one parameter");
		REQUIRE(N > 0, "Problem needs at least one measurement");
		REQUIRE(P > 0, "Problem needs at least one output dimension");
		REQUIRE(_values.dimensions.size() == 2, "Values must be of order 2 (N,P)");

		// ensure measures are consistent with x and values
		REQUIRE(_measures.size() == M-1, "...");
		for (size_t m=1; m<M; ++m) {
			REQUIRE(_measures[m-1].dimensions == Tensor::DimensionTuple({N, x.dimensions[m]}), "...");
		}
		// ensure x is consistent with values
		REQUIRE(x.dimensions[0] == P, "...");

		LOG(debug, "Split measurement tensors");
		values = slice_tensor(_values);
		for (size_t m=1; m<M; ++m) {
			measures[m] = slice_tensor(_measures[m-1]);
		}

		LOG(debug, "Reshuffle measurements");
		std::vector<size_t> indices(N);
		std::iota(indices.begin(), indices.end(), 0);
		std::random_shuffle(indices.begin(), indices.end());
		reorder(values, indices);
		for (size_t m=1; m<M; ++m) { reorder(measures[m], indices); }

		LOG(debug, "Leaving SALSA()");
	}

	void SALSA::move_core_left(const bool adapt) {
		const size_t pos = x.corePosition;
		LOG(debug, "Entering move_core_left(adapt=" << std::string(adapt?"true":"false") << ")    [" << pos-1 << " <-- " << pos << "]");
		REQUIRE(0 < pos, "core at position " << pos << " can not move in direction 'left' in tensor of order " << x.order());

		Tensor& old_core = x.component(pos);
		Tensor& new_core = x.component(pos-1);
		Tensor U,S,Vt;

		calculate_svd(U, S, Vt, old_core, 1, 0, 0);
		// splitPos == 1 --> U(left,r1) * S(r1,r2) * Vt(r2,ext,right) == old_core(left,ext,right)   (The left part has 1 external index.)
		// maxRank == 0  --> do not perform hard thresholding
		// eps == 0      --> do not round to eps% of norm.
		REQUIRE(Tensor::DimensionTuple(U.dimensions.begin(), U.dimensions.end()-1) == Tensor::DimensionTuple(old_core.dimensions.begin(), old_core.dimensions.begin()+1), "IE");
		REQUIRE(Tensor::DimensionTuple(Vt.dimensions.begin()+1, Vt.dimensions.end()) == Tensor::DimensionTuple(old_core.dimensions.begin()+1, old_core.dimensions.end()), "IE");

		old_core = Vt;
		contract(new_core, new_core, U, 1);  // new_core(i^2,j) << new_core(i^2,l) * U(l,j)

		if (adapt) {
			// adapt the rank (pos-1)--(pos) i.e. x.rank(pos-1)
			size_t maxRank = std::min(maxRanks[pos-1], std::numeric_limits<size_t>::max()-kmin) + kmin;
			REQUIRE(maxRank >= maxRanks[pos-1], "IE");
			double threshold = 0.1*smin;  //TODO: in the unchecked (i.e. commented out) version of vresalsa threshold = 0.1*self.residual(self.trainingSet)
			adapt_rank(new_core, S, old_core, maxRank, threshold);
			x.nodes[pos].neighbors[2].dimension = new_core.dimensions[2];
			x.nodes[pos+1].neighbors[0].dimension = old_core.dimensions[0];
		}

		contract(new_core, new_core, S, 1);  // new_core(i^2,j) << new_core(i^2,l) * S(l,j)
		REQUIRE(new_core.all_entries_valid() && old_core.all_entries_valid(), "IE");
		x.assume_core_position(pos-1);

		calc_right_stack(pos);

		singularValues[pos-1].resize(S.dimensions[0]);
		for (size_t i=0; i<S.dimensions[0]; ++i) {
			singularValues[pos-1][i] = S[{i,i}];
		}

		if (initialized && 1 < pos) {
			Tensor& next_core = x.component(pos-2);
			calculate_svd(U, S, Vt, new_core, 1, 0, 0);  // (U(left,r1), S(r1,r2), Vt(r2,ext,right)) = new_core(left,ext,right)
			REQUIRE(U.order() == 2 && U.dimensions[0] == U.dimensions[1], "IE");
			contract(next_core, next_core, U, 1);
			contract(new_core, S, Vt, 1);
			REQUIRE(new_core.all_entries_valid() && next_core.all_entries_valid(), "IE");

			calc_left_stack(pos-2);
			//TODO: Das muss nicht sein.
			//      Du kannst Vt auch einfach als `rightOrthogonalTransfrom` speichern.
			//      Nach der Berechnung der RHS und des Op werden `leftOrthogonalTransform` und `rightOrthogonalTransform` ranmultipliziert.
			//      Tensor coreTransform;
			//      contract(coreTransform, leftOrthogonalTransform, Tensor::identity({e,e}), 0);
			//      contract(coreTransform, coreTransform, rightOrthogonalTransform, 0);
			//      reshuffle(coreTransform, {0,3,1,4,2,5});  // lleerr -> lerler
			//      contract(op, coreTransform, op, 3); contract(op, op, coreTransform, 3);
			//      contract(rhs, coreTransform, rhs, 3);
			//      Da coreTransform regulär ist kann man die Multiplikation von links mit `coreTransfrom` auch weglassen.
			//      Da außerdem entweder nur `leftOrthogonalTransfrom` oder `rightOrthogonalTransform` aktiv sind
			//      (je nach sweep direction ist die andere eine Identität) kann man auch direkt `coreTransform` speichern.

			singularValues[pos-2].resize(S.dimensions[0]);
			for (size_t i=0; i<S.dimensions[0]; ++i) {
				singularValues[pos-2][i] = S[{i,i}];
			}
		}
		LOG(debug, "Leaving move_core_left()");
	}

	void SALSA::move_core_right(const bool adapt) {
		//TODO: You can use misc::product.
		const size_t pos = x.corePosition;
		LOG(debug, "Entering move_core_right(adapt=" << std::string(adapt?"true":"false") << ")    [" << pos << " --> " << pos+1 << "]");
		REQUIRE(pos+1 < x.order(), "core at position " << pos << " can not move in direction 'right' in tensor of order " << x.order());

		Tensor& old_core = x.component(pos);
		Tensor& new_core = x.component(pos+1);
		Tensor U,S,Vt;

		calculate_svd(U, S, Vt, old_core, 2, 0, 0);  // (U(left,ext,r1), S(r1,r2), Vt(r2,right)) = old_core(left,ext,right)
		REQUIRE(Tensor::DimensionTuple(U.dimensions.begin(), U.dimensions.end()-1) == Tensor::DimensionTuple(old_core.dimensions.begin(), old_core.dimensions.begin()+2), "IE");
		REQUIRE(Tensor::DimensionTuple(Vt.dimensions.begin()+1, Vt.dimensions.end()) == Tensor::DimensionTuple(old_core.dimensions.begin()+2, old_core.dimensions.end()), "IE");

		old_core = U;
		contract(new_core, Vt, new_core, 1);  // new_core(i,j^2) << Vt(i,r) * new_core(r,j^2)

		if (adapt) {
			// adapt the rank (pos)--(pos+1) i.e. x.rank(pos)
			size_t maxRank = std::min(maxRanks[pos], std::numeric_limits<size_t>::max()-kmin) + kmin;
			REQUIRE(maxRank >= maxRanks[pos], "IE");
			double threshold = 0.1*smin;  //TODO: in the unchecked (i.e. commented out) version of vresalsa threshold = 0.1*self.residual(self.trainingSet)
			adapt_rank(old_core, S, new_core, maxRank, threshold);
			x.nodes[pos+1].neighbors[2].dimension = old_core.dimensions[2];
			x.nodes[pos+2].neighbors[0].dimension = new_core.dimensions[0];
		}

		contract(new_core, S, new_core, 1);  // new_core(i,j^2) << S(i,l) * new_core(l,j^2)
		REQUIRE(new_core.all_entries_valid() && old_core.all_entries_valid(), "IE");
		x.assume_core_position(pos+1);

		calc_left_stack(pos);

		singularValues[pos].resize(S.dimensions[0]);
		for (size_t i=0; i<S.dimensions[0]; ++i) {
			singularValues[pos][i] = S[{i,i}];
		}

		if (pos+2 < x.order()) {
			Tensor& next_core = x.component(pos+2);
			calculate_svd(U, S, Vt, new_core, 2, 0, 0);  // (U(left,ext,r1), S(r1,r2), Vt(r2,right)) = new_core(left,ext,right)
			REQUIRE(Vt.order() == 2 && Vt.dimensions[0] == Vt.dimensions[1], "IE");
			contract(next_core, Vt, next_core, 1);
			contract(new_core, U, S, 1);
			REQUIRE(new_core.all_entries_valid() && next_core.all_entries_valid(), "IE");

			calc_right_stack(pos+2);  //TODO: see move_core_left

			singularValues[pos+1].resize(S.dimensions[0]);
			for (size_t i=0; i<S.dimensions[0]; ++i) {
				singularValues[pos+1][i] = S[{i,i}];
			}
		}
		LOG(debug, "Leaving move_core_right()");
	}


	void SALSA::calc_left_stack(const size_t _position) {
		//TODO: do we need to compute the stacks for the validation set?
		LOG(debug, "Entering calc_left_stack(" << _position << ")");
		REQUIRE(_position+1 == x.corePosition, "IE"); //TODO: remove argument
		// compute measCmp.T @ leftLHSStack[_position-1] @ measCmp
		// and     leftRHSStack[_position-1] @ measCmp
		// where   measCmp = measures[position] @ x.get_component(_position)
		REQUIRE(_position+1 < M, "Invalid corePosition");

		if(_position == 0) {
			const Tensor shuffledX = reinterpret_dimensions(x.get_component(_position), {x.dimensions[0], x.rank(0)});  // Remove dangling 1-mode
			#pragma omp parallel for default(none) shared(leftRHSStack, values) firstprivate(N, _position, shuffledX)
			for(size_t i = 0; i < N; ++i) {
				//NOTE: The first component is contracted directly (leftLHSStack[0] = shuffledX.T @ shuffledX).
				//NOTE: Since shuffeldX is left-orthogonal leftLHSStack[0] is the identity.
				contract(leftRHSStack[_position][i], values[i], shuffledX, 1);  // e,er -> r
			}
		} else if(_position == 1) {
			Tensor measCmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			#pragma omp parallel for default(none) shared(measures, leftLHSStack, leftRHSStack) firstprivate(N, _position, shuffledX) private(measCmp)
			for(size_t j = 0; j < N; ++j) {
				contract(measCmp, measures[_position][j], shuffledX, 1);                         // ler,e -> lr
				//NOTE: leftLHSStack[0] is the identity
				contract(leftLHSStack[_position][j], measCmp, true, measCmp, false, 1);          // lr,ls -> rs
				contract(leftRHSStack[_position][j], leftRHSStack[_position-1][j], measCmp, 1);  // r,rs  -> s
			}
		} else {
			Tensor measCmp, tmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			#pragma omp parallel for default(none) shared(measures, leftLHSStack, leftRHSStack) firstprivate(N, _position, shuffledX) private(measCmp, tmp)
			for(size_t j = 0; j < N; ++j) {
				contract(measCmp, measures[_position][j], shuffledX, 1);                         // ler,e -> lr
				contract(tmp, leftLHSStack[_position-1][j], true, measCmp, false, 1);            // tmp(r2,u1) = stack[_pos-1](r1,r2) * measCmp(r1,u1)
				contract(leftLHSStack[_position][j], tmp, true, measCmp, false, 1);              // stack[_pos](u1,u2) = tmp(r2,u1) * measCmp(r2,u2)
				contract(leftRHSStack[_position][j], leftRHSStack[_position-1][j], measCmp, 1);  // r,rs  -> s
			}
		}

		if(_position == 0) {
			Tensor tmp;
			const Tensor shuffledX = reinterpret_dimensions(x.get_component(_position), {x.dimensions[0], x.rank(0)});  // Remove dangling 1-mode
			contract(tmp, basisWeights[_position], shuffledX, 1);
			contract(leftRegularizationStack[_position], shuffledX, true, tmp, false, 1);  // leftRegularizationStack[_position] = np.einsum("er,e,es -> rs", xpos, mpos, xpos)

		} else {
			Tensor tmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			contract(tmp, basisWeights[_position], shuffledX, 1);                                        // tmp(e2,l2,r2) = W(e2,x) * shuffledX(x,l2,r2)
			contract(tmp, shuffledX, true, tmp, false, 1);                                               // tmp(l1,r1,l2,r2) = shuffledX(x,l1,r1) * tmp(x,l2,r2)
			reshuffle(tmp, tmp, {0,2,1,3});
			contract(leftRegularizationStack[_position], leftRegularizationStack[_position-1], tmp, 2);  // stack[_pos](r1,r2) = stack[pos-1](l1,l2) * tmp(l1,l2,r1,r2)
		}
		LOG(debug, "Leaving calc_left_stack()");
	}

	void SALSA::calc_right_stack(const size_t _position) {
		LOG(debug, "Entering calc_right_stack(" << _position << ")");
		REQUIRE(_position == x.corePosition+1, "IE"); //TODO: remove argument
		// compute measCmp.T @ rightStack[_position+1]
		// where   measCmp = measures[position] @ x.get_component(_position)
		REQUIRE(0 < _position && _position < M, "Invalid corePosition");

		if(_position < M-1) {
			Tensor measCmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			#pragma omp parallel for default(none) shared(measures, rightStack) firstprivate(N, _position, shuffledX) private(measCmp)
			for(size_t j = 0; j < N; ++j) {
				contract(measCmp, measures[_position][j], shuffledX, 1);
				contract(rightStack[_position][j], measCmp, rightStack[_position+1][j], 1);
			}
		} else {  // _position == M-1
			const Tensor shuffledX = reinterpret_dimensions(x.get_component(_position), {x.rank(M-2), x.dimensions[M-1]});  // Remove dangling 1-mode
			#pragma omp parallel for default(none) shared(measures, rightStack) firstprivate(N, _position, shuffledX)
			for(size_t j = 0; j < N; ++j) {
				contract(rightStack[_position][j], shuffledX, measures[_position][j], 1);
			}
		}

		if(_position < M-1) {
			Tensor tmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			contract(tmp, basisWeights[_position], shuffledX, 1);
			contract(tmp, shuffledX, true, tmp, false, 1);
			reshuffle(tmp, tmp, {0,2,1,3});
			contract(rightRegularizationStack[_position], tmp, rightRegularizationStack[_position+1], 2);
		} else {
			Tensor tmp;
			Tensor shuffledX = reinterpret_dimensions(x.get_component(_position), {x.rank(M-2), x.dimensions[M-1]});  // Remove dangling 1-mode
			contract(tmp, shuffledX, basisWeights[_position], 1);
			contract(rightRegularizationStack[_position], tmp, false, shuffledX, true, 1);  // stack[M-1](r1,r2) = shuffledX(r1,e1) * W(e1,e2) * shuffledX(e2,r2)
		}
		LOG(debug, "Leaving calc_right_stack()");
	}

	void SALSA::adapt_rank(Tensor& _U, Tensor& _S, Tensor& _Vt, const size_t _maxRank, const double _threshold) const {
		LOG(debug, "Entering adapt_rank(maxRank=" << _maxRank << ", threshold=" << string_format("%.2e", _threshold) << ")");
		const size_t eU = _U.order()-1; //TODO: rename
		const size_t eV = _Vt.order()-1;
		REQUIRE(_U.dimensions[eU] == _S.dimensions[0] &&  _S.dimensions[1] == _Vt.dimensions[0], "Inconsistent dimensions: " << _U.dimensions << " vs " << _S.dimensions << " vs " << _Vt.dimensions);

		size_t rank;
		for (rank=0; rank<_S.dimensions[0]; ++rank) {
			if (_S[{rank,rank}] <= smin) break;
		}
		size_t maxRank = std::min(_maxRank, std::min(_U.size/_U.dimensions[eU], _Vt.size/_Vt.dimensions[0]));
		size_t full_rank = std::min(rank+kmin, maxRank);
		LOG(debug, "rank=" << rank << "    full_rank=" << full_rank);

		while (_S.dimensions[0] < full_rank) {
			LOG(debug, "Increase rank: " << _S.dimensions[0] << " --> " << _S.dimensions[0]+1);
			// Um,Un = U.size/U.dimensions[eU], U.dimensions[eU]
			// Vtm,Vtn = Vt.dimensions[0], Vt.size/Vt.dimensions[0]
			// The rank can only be increased when the dimensions of U and Vt allow it (Um > Un and Vtm < Vtn).
			REQUIRE(_U.size > misc::sqr(_U.dimensions[eU]) && _Vt.size > misc::sqr(_Vt.dimensions[0]), "IE");
			//NOTE: This condition is guaranteed by maxRank <= std::min(_U.size/_U.dimensions[eU], _Vt.size/_Vt.dimensions[0]).

			// Add a new diagonal entry with a value at 1% of the singular value threshold smin.
			_S.resize_mode(0, _S.dimensions[0]+1);
			_S.resize_mode(1, _S.dimensions[1]+1);
			_S += 0.01 * smin * Tensor::dirac(_S.dimensions, {_S.dimensions[0]-1, _S.dimensions[1]-1});
			/* REQUIRE(_S.sparsity() == _S.dimensions[0], "IE: " << _S.sparsity() << " !=  " << _S.dimensions[0] << "Tensor:\n" << _S);  //TODO: Diese Bedingung muss nicht erfüllt sein, denn für 2x2-Matrizen wird immer das dense Format verwendet! */

			Tensor slate, tmp;
			std::vector<size_t> slate_dimensions, slate_index;

			// Increase the size of the last mode of _U by adding a random orthogonal slate.
			_U.resize_mode(eU, _U.dimensions[eU]+1);
			slate_dimensions = std::vector<size_t>(_U.dimensions.cbegin(), _U.dimensions.cend()-1);
			slate = Tensor::random(slate_dimensions);
			slate /= slate.frob_norm();

			contract(tmp, slate, _U, eU);
			REQUIRE(tmp.dimensions == Tensor::DimensionTuple({_U.dimensions[eU]}), "IE");
			contract(tmp, _U, tmp, 1);
			slate -= tmp;
			slate /= slate.frob_norm();

			slate_dimensions.push_back(1);
			slate.reinterpret_dimensions(slate_dimensions);
			slate_index = std::vector<size_t>(eU+1, 0); slate_index[eU] = _U.dimensions[eU]-1;
			_U.offset_add(slate, slate_index);

			_Vt.resize_mode(0, _Vt.dimensions[0]+1);
			slate_dimensions = std::vector<size_t>(_Vt.dimensions.cbegin()+1, _Vt.dimensions.cend());
			slate = Tensor::random(slate_dimensions);
			slate /= slate.frob_norm();

			contract(tmp, _Vt, slate, eV);
			REQUIRE(tmp.dimensions == Tensor::DimensionTuple({_Vt.dimensions[0]}), "IE");
			contract(tmp, tmp, _Vt, 1);
			slate -= tmp;
			slate /= slate.frob_norm();

			slate_dimensions.insert(slate_dimensions.begin(), 1);
			slate.reinterpret_dimensions(slate_dimensions);
			slate_index = std::vector<size_t>(eV+1, 0); slate_index[0] = _Vt.dimensions[0]-1;
			_Vt.offset_add(slate, slate_index);
		}
		if (_S.dimensions[0] > full_rank && _S[{_S.dimensions[0]-1, _S.dimensions[1]-1}] < _threshold) {  // remove at most 1 rank per call
			LOG(debug, "Decrease rank: " << _S.dimensions[0] << " --> " << _S.dimensions[0]-1);
			_S.remove_slate(0, _S.dimensions[0]-1);
			_S.remove_slate(1, _S.dimensions[1]-1);
			_U.remove_slate(eU, _U.dimensions[eU]-1);
			_Vt.remove_slate(0, _Vt.dimensions[0]-1);
		}
		REQUIRE(_U.all_entries_valid() && _S.all_entries_valid() && _Vt.all_entries_valid(), "IE");

		for (rank=0; rank<_S.dimensions[0]; ++rank) {
			if (_S[{rank,rank}] <= smin) break;
		}
		LOG(debug, "Leaving adapt_rank()");
	}

	double SALSA::residual(const std::pair<size_t, size_t>& _slice) const {
		const auto [from, to] = _slice;
		LOG(debug, "Entering residual((" << from << ", " << to << "))");
		REQUIRE(x.corePosition == 0, "IE");
		REQUIRE(from <= to && to <= N, "IE");
		const Tensor shuffledX = reinterpret_dimensions(x.get_component(0), {x.dimensions[0], x.rank(0)});  // Remove dangling 1-mode
		Tensor tmp;
		double res = 0.0, valueNorm = 0.0;
		#pragma omp parallel for default(none) shared(rightStack, values) firstprivate(from, to, shuffledX) private(tmp) reduction(+:res,valueNorm)
		for(size_t j = from; j < to; ++j) {
			contract(tmp, shuffledX, rightStack[1][j], 1);
			res += misc::sqr(frob_norm(values[j] - tmp));
			valueNorm += misc::sqr(frob_norm(values[j]));
		}
		LOG(debug, "Leaving residual()");
		return std::sqrt(res/valueNorm);
	}

	Tensor SALSA::omega_operator() const {  // compute SALSA regularization term
		LOG(debug, "Entering omega_operator()");
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		// compute left part
		Tensor Gamma_sq({l,l});
		if (pos == 0) {
			REQUIRE(l == 1, "IE");
			Gamma_sq[0] = x.frob_norm();
		} else {
			for (size_t j=0; j<l; ++j) {
				Gamma_sq[{j,j}] = 1.0 / misc::sqr(std::max(smin, singularValues[pos-1][j]));
			}
		}
		contract(Gamma_sq, Gamma_sq, Tensor::identity({e,e}), 0);
		contract(Gamma_sq, Gamma_sq, Tensor::identity({r,r}), 0);
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Gamma_sq, Gamma_sq, {0,3,1,4,2,5});  // axbycz -> abcxyz
		REQUIRE(Gamma_sq.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");

		// compute right part
		Tensor Theta_sq({r,r});
		if (pos < M-1) {
			for (size_t j=0; j<r; ++j) {
				Theta_sq[{j,j}] = 1.0 / misc::sqr(std::max(smin, singularValues[pos][j]));
			}
		} else {
			REQUIRE(r == 1, "IE");
			Theta_sq[0] = x.frob_norm();
		}
		contract(Theta_sq, Tensor::identity({e,e}), Theta_sq, 0);
		contract(Theta_sq, Tensor::identity({l,l}), Theta_sq, 0);
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Theta_sq, Theta_sq, {0,3,1,4,2,5});  // axbycz -> abcxyz
		REQUIRE(Theta_sq.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");

		LOG(debug, "Leaving omega_operator()");
		return misc::sqr(omega) * (Gamma_sq + Theta_sq);
	}

	Tensor SALSA::alpha_operator() const {  // compute LASSO regularization term
		LOG(debug, "Entering alpha_operator()");
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		Tensor Op;
		if (pos == 0) { Op = Tensor::identity({1,1}); }
		else { Op = leftRegularizationStack[pos-1]; }
		contract(Op, Op, basisWeights[pos], 0);
		if (pos < M-1) { contract(Op, Op, rightRegularizationStack[pos+1], 0); }
		if (pos == x.order()-1) { Op.reinterpret_dimensions({l,l,e,e,1,1}); }
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Op, Op, {0,3,1,4,2,5});  // axbycz -> abcxyz
		REQUIRE(Op.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");

		LOG(debug, "Leaving alpha_operator()");
		return misc::sqr(alpha) * Op;
	}

	std::pair<Tensor, Tensor> SALSA::ls_operator_and_rhs(const std::pair<size_t, size_t>& _slice) const {
		LOG(debug, "Entering ls_operator((" << _slice.first << ", " << _slice.second << "))");
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		Tensor op;
		Tensor rhs;
		if (pos == 0) {
			op = Tensor({e,e,r,r}, Tensor::Representation::Dense);
			rhs = Tensor({e,r}, Tensor::Representation::Dense);
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(rightStack, values) firstprivate(_slice, e, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				contract(tmp, Tensor::identity({e,e}), rightStack[pos+1][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				op += tmp;

				contract(tmp, values[i], rightStack[pos+1][i], 0);
				rhs += tmp;
			}
			/* //TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen. */
			reshuffle(op, op, {0,2,1,3});  // eerr -> erer
			op.reinterpret_dimensions({1,e,r,1,e,r});
			rhs.reinterpret_dimensions({1,e,r});
		} else if (pos == 1) {
			op = Tensor({e,r,e,r}, Tensor::Representation::Dense);
			rhs = Tensor({l,e,r}, Tensor::Representation::Dense);
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftRHSStack, rightStack, measures) firstprivate(_slice, l, e, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				contract(tmp, measures[pos][i], rightStack[pos+1][i], 0);
				contract(tmp, tmp, measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				op += tmp;

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				rhs += tmp;
			}
			contract(op, Tensor::identity({l,l}), op, 0);  // leftLHSStack[pos-1][i] is the identity and the tensor product is a distributive binary operation
			//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
			reshuffle(op, op, {0,3,1,2,4,5});  // llerer -> lerler
		} else if (pos < M-1) {
			op = Tensor({l,l,e,r,e,r}, Tensor::Representation::Dense);
			rhs = Tensor({l,e,r}, Tensor::Representation::Dense);
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, rightStack, measures) firstprivate(_slice, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				contract(tmp, tmp, measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				/* REQUIRE(op.is_dense() && !op.has_factor() && op.get_internal_dense_data().unique(), "IE"); */
				/* REQUIRE(tmp.is_dense() && !tmp.has_factor() && tmp.get_internal_dense_data().unique(), "IE"); */
				/* misc::add(op.get_dense_data(), tmp.get_dense_data(), tmp.size); */
				op += tmp;

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				/* REQUIRE(rhs.is_dense() && !rhs.has_factor() && rhs.get_internal_dense_data().unique(), "IE"); */
				/* REQUIRE(tmp.is_dense() && !tmp.has_factor() && tmp.get_internal_dense_data().unique(), "IE"); */
				/* misc::add(rhs.get_dense_data(), tmp.get_dense_data(), tmp.size); */
				rhs += tmp;
			}
			//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
			/* reshuffle(tmp, tmp, {0,3,1,4,2,5});  // lleerr -> lerler */
			reshuffle(op, op, {0,3,1,2,4,5});  // llerer -> lerler
		} else {
			op = Tensor({l,l,e,e}, Tensor::Representation::Dense);
			rhs = Tensor({l,e}, Tensor::Representation::Dense);
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, measures) firstprivate(_slice, pos, l, e) private(tmp) reduction(+:op,rhs)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, measures[pos][i], 0);
				op += tmp;

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				rhs += tmp;
			}
			//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
			reshuffle(op, op, {0,2,1,3});  // llee -> lele
			op.reinterpret_dimensions({l,e,1,l,e,1});
			rhs.reinterpret_dimensions({l,e,1});
		}
		REQUIRE(op.dimensions == Tensor::DimensionTuple({l,e,r, l,e,r}), "IE");  // In a macro you need parantheses around an initializer list.
		REQUIRE(rhs.dimensions == Tensor::DimensionTuple({l,e,r}), "IE");

		LOG(debug, "Leaving ls_operator()");
		return std::make_pair(op, rhs);
	}

	double SALSA::slow_residual(const std::pair<size_t, size_t>& _slice) const {
		// TODO: Merge with residual?
		// ||Ax - b||^2 = xtAtAx - 2*xtAtb + btb
		LOG(debug, "Entering slow_residual((" << _slice.first << ", " << _slice.second << "))");

		const Tensor& core = x.get_component(x.corePosition);
		const auto[A, b] = ls_operator_and_rhs(_slice);

		const double xtAtAx = contract(contract(core, A, 3), core, 3)[0];
		const double xtAtb  = contract(core, b, 3)[0];
		const double btb    = misc::sqr(valueNorm_trainingSet);

		LOG(debug, "Leaving slow_residual()");
		return std::sqrt(std::max(xtAtAx - 2*xtAtb + btb, 0.0)) / valueNorm_trainingSet;
	}


	void SALSA::solve_local() {
		const size_t pos = x.corePosition;  //TODO: rename: position
		LOG(debug, "Entering solve_local(position=" << pos << ")");

		const auto[op, rhs] = ls_operator_and_rhs(trainingSet);
		const Tensor op_alpha = alpha_operator();
		const Tensor op_omega = omega_operator();

		Tensor& core = x.component(pos);
		solve(core, op+op_alpha+op_omega, rhs);

		// iterative reweighting
		Tensor IR, op_IRalpha, prev_core;
		size_t step;
		for (step=0; step<maxIRsteps; ++step) {
			IR = diag(core, [sparsityThreshold=sparsityThreshold](double _entry) { return 1.0/std::sqrt(std::max(std::abs(_entry), sparsityThreshold)); });
			contract(op_IRalpha, IR, op_alpha, 3);
			contract(op_IRalpha, op_IRalpha, IR, 3);
			prev_core = core;
			solve(core, op+op_IRalpha+op_omega, rhs);
			if (max_norm(prev_core - core) < IRtolerance*frob_norm(prev_core)) break;
		}
		maxIRstepsReached[pos] = (step == maxIRsteps);

		size_t density = 0; // np.count_nonzero(abs(sol) > sparsityThreshold)/sol.size
		#pragma omp parallel for default(none) firstprivate(core, sparsityThreshold) reduction(+:density)
		for (size_t j=0; j<core.size; ++j) {
			density += std::abs(core[j]) > sparsityThreshold;
		}
		weightedNorms[pos] = double(density)/double(core.size);
		REQUIRE(0 <= weightedNorms[pos] && weightedNorms[pos] <= 1, "IE");
		LOG(debug, "Leaving solve_local()");
	}

	void SALSA::print_parameters() const {
		LOG(debug, "Entering print_parameters()");
		const size_t max_param_len = 26;  // "maxNonImprovingAlphaCycles".size()
		const auto print_param = [max_param_len](std::string name, auto value) {
			const std::string pad(max_param_len-name.length(), ' ');
			std::cout << "  " << name << " = " << pad << value << "\n";
		};
		const std::string sep = std::string(125, '-')+"\n";
		std::cout << sep;
		print_param("dimensions", print_list(x.dimensions));
		print_param("initial_ranks", print_list(x.ranks()));
		print_param("num_samples", N);
		std::cout << sep;
		print_param("controlSetFraction", controlSetFraction);
		std::cout << '\n';
		print_param("targetResidual", targetResidual);
		std::cout << '\n';
		print_param("minDecrease", minDecrease);
		print_param("maxIterations", maxIterations);
		print_param("trackingPeriodLength", trackingPeriodLength);
		print_param("maxNonImprovingAlphaCycles", maxNonImprovingAlphaCycles);
		std::cout << '\n';
		print_param("maxIRsteps", maxIRsteps);
		print_param("IRtolerance", IRtolerance);
		print_param("sparsityThreshold", sparsityThreshold);
		std::cout << '\n';
		print_param("kmin", kmin);
		print_param("maxRanks", print_list<size_t>(maxRanks, [](const size_t _rank) -> std::string {
			if (_rank == std::numeric_limits<size_t>::max()) { return u8"\u221e"; }
			return std::to_string(_rank);
		}));
		std::cout << '\n';
		print_param("fomega", fomega);
		print_param("omega_factor", omega_factor);
		std::cout << '\n';
		print_param("falpha", falpha);
		print_param("alpha_factor", alpha_factor);
		std::cout << sep;
		LOG(debug, "Leaving print_parameters()");
	}

	void SALSA::initialize() {
		LOG(debug, "Entering initialize()");

		// assert np.shape(weights) == (N,)
		//TODO: assert symmetry
		REQUIRE(basisWeights.size() == M, "...");
		for (size_t m=0; m<M; ++m) {
			REQUIRE(basisWeights[m].dimensions == Tensor::DimensionTuple({x.dimensions[m], x.dimensions[m]}), "...");
		}

		// build stacks and compute left and right singular value arrays
		x.move_core(x.order()-1);
		while (x.corePosition > 0) { move_core_left(false); }

		// The sets have already been shuffled in the constructor. Now they need to be split.
		size_t Nt = size_t((1.0-controlSetFraction)*double(N));  // Nv = N - Nt;
		trainingSet = std::make_pair(0, Nt);
		validationSet = std::make_pair(Nt, N);

		valueNorm_trainingSet = 0.0;
		#pragma omp parallel for default(none) shared(values) firstprivate(trainingSet) reduction(+:valueNorm_trainingSet)
		for(size_t j=trainingSet.first; j<trainingSet.second; ++j) {
			valueNorm_trainingSet += misc::sqr(frob_norm(values[j]));
		}
		valueNorm_trainingSet = std::sqrt(valueNorm_trainingSet);

		valueNorm_validationSet = 0.0;
		#pragma omp parallel for default(none) shared(values) firstprivate(validationSet) reduction(+:valueNorm_validationSet)
		for(size_t j=validationSet.first; j<validationSet.second; ++j) {
			valueNorm_validationSet += misc::sqr(frob_norm(values[j]));
		}
		valueNorm_validationSet = std::sqrt(valueNorm_validationSet);

		// compute SALSA parameters
		double res = residual(std::make_pair(0, N));
		double maxResSqrtRes = std::max(res, std::sqrt(res));
		alpha = alpha_factor * std::min(valueNorm_trainingSet, maxResSqrtRes);
		omega = maxResSqrtRes;
		smin = 0.2*std::min(omega, res);
		omega *= omega_factor;

		initialized = true;

		LOG(debug, "Leaving initialize()");
	}

	std::string SALSA::print_fractional_ranks() const {
		LOG(debug, "Entering print_fractional_ranks()");
		/* assert len(self.singularValues) == self.x.order()-1 */
		/* for i in range(self.x.order()-1): */
		/*     assert len(self.singularValues[i]) == self.x.rank(i) */
		const std::string dark_grey = u8"\033[38;5;242m";
		const std::string light_grey = u8"\033[38;5;247m";
		const std::string reset = u8"\033[0m";
		std::string output = print_list<std::vector<double>>(singularValues, [&](const std::vector<double>& _sVs) {
			/* assert np.all(_sVs > 0) */
			size_t rank;
			for (rank=0; rank<_sVs.size(); ++rank) {
				if (_sVs[rank] <= smin) break;
			}
			double inactive = 0.0;
			if (rank < _sVs.size()) {
				inactive = _sVs[rank]/smin;
				REQUIRE(0 < inactive && inactive < 1, "IE");
			}
			return std::to_string(rank) + "." + light_grey + std::to_string(size_t(10*inactive)) + reset + u8"\u002F" + dark_grey + std::to_string(_sVs.size()) + reset;
		});
		LOG(debug, "Leaving print_fractional_ranks()");
		return output;
	}

	std::string SALSA::print_densities() const {
		LOG(debug, "Entering print_densities()");
		const std::string yellow = u8"\033[38;5;230m";
		const std::string reset = u8"\033[0m";
		std::string output = print_list(M, [&](const size_t _index) {
			if (maxIRstepsReached[_index]) {
				return string_format(yellow+"%2u"+reset, std::min(size_t(100*weightedNorms[_index]+0.5), size_t{99}));
			}
			return string_format("%2u", std::min(size_t(100*weightedNorms[_index]+0.5), size_t{99}));
		});
		LOG(debug, "Leaving print_densities()");
		return output + "%";
	}

	void SALSA::run() {
		LOG(debug, "Entering run()");
		std::cout << std::string(125, '=') << '\n'
				  << std::string(55, ' ') << "Running uqSALSA" << std::endl;
		print_parameters();
		REQUIRE(omega_factor > 0.0, "omega_factor must be positive");
		REQUIRE(alpha_factor >= 0.0, "alpha_factor must be positive");
		if (alpha_factor == 0.0) {
			std::cout << "WARNING: Optimizing without l1 regularization" << std::endl;
		}
		initialize();

		REQUIRE(x.corePosition == 0, "IE");
		LOG(debug, "Sweep: left --> right");
		for (size_t m=0; m<M-1; ++m) {
			REQUIRE(x.corePosition == m, "IE");
			LOG(debug, "(ini)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
			solve_local();
			move_core_right(false);
			LOG(debug, "(fin)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
		}
		LOG(debug, "Sweep: right --> left");
		for (size_t m=M-1; m>0; --m) {
			REQUIRE(x.corePosition == m, "IE");
			LOG(debug, "(ini)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
			solve_local();
			move_core_left(false);
			LOG(debug, "(fin)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
		}
		REQUIRE(x.corePosition == 0, "IE");

		/* boost::circular_buffer<double> trainingResiduals{trackingPeriodLength, std::numeric_limits<double>::max()};  // Creates a full circular buffer with every element equal to the max. double. */
		boost::circular_buffer<double> trainingResiduals{trackingPeriodLength};
		std::vector<double> validationResiduals;
		trainingResiduals.push_back(residual(trainingSet));
		validationResiduals.push_back(residual(validationSet));

		initialResidual = trainingResiduals.back();  //TODO: rename
		bestIteration = 0;
		bestX = x;
		bestTrainingResidual = trainingResiduals.back();
		bestValidationResidual = validationResiduals.back();
		double prev_bestValidationResidual = validationResiduals.back();
		/* double bestValidationResidual_cycle = bestValidationResidual; */

		size_t iteration = 0;
		size_t nonImprovementCounter = 0;
		bool omegaMinimal = false;

		auto alpha_residual = [&](){
			const Tensor op_alpha = alpha_operator();
			const Tensor& core = x.get_component(x.corePosition);
			const Tensor IR = diag(core, [sparsityThreshold=sparsityThreshold](double _entry) { return 1.0/std::sqrt(std::max(std::abs(_entry), sparsityThreshold)); });
			Tensor ret;
			contract(ret, core, IR, 3);
			contract(ret, ret, op_alpha, 3);
			contract(ret, ret, IR, 3);
			contract(ret, ret, core, 3);
			REQUIRE(ret.dimensions == Tensor::DimensionTuple({}), "IE");
			return ret[0];
		};
		auto omega_residual = [&](){
			const Tensor op_omega = omega_operator();
			const Tensor& core = x.get_component(x.corePosition);
			Tensor ret;
			contract(ret, core, op_omega, 3);
			contract(ret, ret, core, 3);
			REQUIRE(ret.dimensions == Tensor::DimensionTuple({}), "IE");
			return ret[0];
		};
		double costs = trainingResiduals.back() + alpha_residual() + omega_residual();
		double bestCosts = costs;

		auto print_update = [&](){
			auto update_str = [](double prev, double cur)  {
				std::ostringstream ret;
				if (cur <= prev+1e-8) ret << "\033[38;5;151m" << string_format("%.2e", cur) << "\033[0m";
				else ret << "\033[38;5;181m" << string_format("%.2e", cur) << "\033[0m";
				return ret.str();
			};
			std::cout << "[" << iteration << "]"
					  << " Costs:" << update_str(bestCosts , costs)
					  << "  |  Residuals: trn=" << update_str(bestTrainingResidual , trainingResiduals.back())
					  << ", val=" << update_str(bestValidationResidual , validationResiduals.back())
					  << "  |  Omega: " << string_format("%.2e", omega)
					  << "  |  Densities: " << print_densities();
			std::cout << "  |  Ranks: " << print_fractional_ranks() << std::endl;
		};
		print_update();

		for (iteration=1; iteration<maxIterations; ++iteration) {
			REQUIRE(x.corePosition == 0, "IE");
			LOG(debug, "Sweep: left --> right");
			for (size_t m=0; m<M-1; ++m) {
				REQUIRE(x.corePosition == m, "IE");
				LOG(debug, "(ini)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
				solve_local();
				move_core_right(true);
				LOG(debug, "(fin)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
			}
			LOG(debug, "Sweep: right --> left");
			for (size_t m=M-1; m>0; --m) {
				REQUIRE(x.corePosition == m, "IE");
				LOG(debug, "(ini)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
				solve_local();
				move_core_left(true);
				LOG(debug, "(fin)[corePosition=" << m << "] Residual: " << string_format("%.8e", slow_residual(trainingSet)));
			}
			REQUIRE(x.corePosition == 0, "IE");

			trainingResiduals.push_back(residual(trainingSet));
			validationResiduals.push_back(residual(validationSet));
			bestTrainingResidual = std::min(trainingResiduals.back(), bestTrainingResidual);
			costs = trainingResiduals.back() + alpha_residual() + omega_residual();
			bestCosts = std::min(costs, bestCosts);
			print_update();

			if (validationResiduals.back() < (1-minDecrease)*bestValidationResidual) {
				bestIteration = iteration;
				bestX = x;
				prev_bestValidationResidual = bestValidationResidual;
				bestValidationResidual = validationResiduals.back();
				bestTrainingResidual = trainingResiduals.back();
			}

			if (validationResiduals.back() < targetResidual) {
				std::cout << "Terminating: Minimum residual reached." << std::endl;
				break;
			}

			double res = trainingResiduals.back();
			omega /= omega_factor;
			omega = std::max(std::min(omega/fomega, std::sqrt(res)), res);
			omegaMinimal = misc::approx_equal(omega, res, 1e-6);
			smin = 0.2*std::min(omega, res);
			omega *= omega_factor;

			if (trainingResiduals.size() >= trackingPeriodLength && trainingResiduals.back() > (1-minDecrease)*trainingResiduals.front() && omegaMinimal) {
				REQUIRE(trainingResiduals.size() == trackingPeriodLength, "IE"); // circular buffer... TODO: rewrite...

				if (bestIteration > iteration-validationResiduals.size()) {
					nonImprovementCounter = 0;
					std::cout << string_format(u8"NonImpCnt: %d (val=%.2e \u2198 val=%.2e)", nonImprovementCounter, prev_bestValidationResidual, bestValidationResidual) << std::endl;
				} else {
					nonImprovementCounter += 1;
					std::cout << string_format(u8"NonImpCnt: %d (val=%.2e \u2192)", nonImprovementCounter, bestValidationResidual) << std::endl;
				}

				if (nonImprovementCounter >= maxNonImprovingAlphaCycles) {
					std::cout << "Terminating: Minimum residual decrease deceeded " << nonImprovementCounter << " iterations in a row." << std::endl;
					break;
				}

				res = validationResiduals.back() * valueNorm_validationSet;
				double prev_alpha = alpha;
				if (alpha_factor > 0.0) {
					alpha = alpha/alpha_factor;
					alpha = std::min(alpha/falpha, std::sqrt(res));
					alpha *= alpha_factor;
					std::cout << "Reduce alpha: " << string_format("%.3f", prev_alpha);
					std::cout << std::string(u8" \u2192 ");
					std::cout << string_format("%.3f", alpha) << std::endl;
				} else { REQUIRE(alpha == 0.0, "IE"); }
				//TODO: use disp_shortest_unequal(self.alpha, alpha)

				trainingResiduals.clear();
				validationResiduals.clear();  // This reset is part of the nonImprovementCounter strategy.
			}
		}

		if (iteration == maxIterations) {
			std::cout << "Terminating: Maximum iterations reached." << std::endl;
		}

		std::cout << "Best validation residual in iteration " << bestIteration << ".\n"
					<< std::string(125, '-') << '\n'
					<< "Truncating inactive singular values." << std::endl;

		size_t rank;
		for (size_t m=0; m<M-1; ++m) {
			for (rank=0; rank<singularValues[m].size(); ++rank) {
				if (singularValues[m][rank] <= smin) break;
			}
			if (rank > maxRanks[m]) {
				std::cout << "WARNING: maxRanks[" << m << "] = " << maxRanks[m] << " was reached during optimization.\n";
			}
		}

		//TODO: the user should do this stuff by herself
		/* x = bestX; */
		/* x.round(maxRanks); */
		/* /1* assert np.all(self.x.ranks() <= np.asarray(self.maxRanks)) *1/ */

		std::cout << string_format("Residual decreased from %.2e to %.2e in %u iterations.\n", initialResidual, bestTrainingResidual, iteration)
					<< std::string(125, '=') << std::endl;
		//TODO: den ganzen kram hier in eine `finish`-Routine auslagern.
		LOG(debug, "Leaving run()");
	}

}} // namespace uq | xerus
