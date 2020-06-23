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
		if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
		std::unique_ptr<char[]> buf( new char[ size ] );
		snprintf( buf.get(), size, format.c_str(), args ... );
		return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
	}

	std::vector<size_t> compute_max_theoretic_ranks(const std::vector<size_t>& _dimensions) {
		// np.minimum(np.cumprod(d[:-1]), np.cumprod(d[:0:-1])[::-1])
		std::vector<size_t> cumprod_left(_dimensions.size()-1);
		std::inclusive_scan(_dimensions.begin(), _dimensions.end()-1,
							cumprod_left.begin(), std::multiplies<>{});

		std::vector<size_t> cumprod_right(_dimensions.size()-1);
		std::inclusive_scan(_dimensions.rbegin(), _dimensions.rend()-1,
							cumprod_right.rbegin(), std::multiplies<>{});

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

	Tensor reinterpret_dimensions(const Tensor& _tensor, const Tensor::DimensionTuple _dimensions) {
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

	SALSA::SALSA(const TTTensor& _x, const std::vector<Tensor>& _measures, const Tensor& _values) :
		x(_x),
		measures(_measures.size()+1),

		M(x.dimensions.size()),
		N(_values.dimensions.at(0)),
		P(_values.dimensions.at(1)),

		maxTheoreticalRanks(compute_max_theoretic_ranks(x.dimensions)),

		leftLHSStack(M, std::vector<Tensor>(N)),
		leftRHSStack(M, std::vector<Tensor>(N)),
		rightStack(M, std::vector<Tensor>(N)),
		leftRegularizationStack(M),
		rightRegularizationStack(M),

		singularValues(M-1),
		weightedNorms(M),

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
			REQUIRE(_measures[m-1].dimensions == std::vector<size_t>({N, x.dimensions[m]}), "...");
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
		//TODO: You can use misc::product.
		const size_t pos = x.corePosition;
		LOG(debug, "Entering move_core_left(" << adapt << ")    [" << pos-1 << " <-- " << pos << "]");
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
			size_t maxRank = std::min(maxRanks[pos-1]+kmin, maxTheoreticalRanks[pos-1]);
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

		if (1 < pos) {
			Tensor& next_core = x.component(pos-2);
			calculate_svd(U, S, Vt, new_core, 1, 0, 0);  // (U(left,r1), S(r1,r2), Vt(r2,ext,right)) = new_core(left,ext,right)
			REQUIRE(U.order() == 2 && U.dimensions[0] == U.dimensions[1], "IE");
			contract(next_core, next_core, U, 1);
			contract(new_core, S, Vt, 1);
			REQUIRE(new_core.all_entries_valid() && next_core.all_entries_valid(), "IE");
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
		LOG(debug, "Entering move_core_right(" << adapt << ")    [" << pos << " --> " << pos+1 << "]");
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
			size_t maxRank = std::min(maxRanks[pos]+kmin, maxTheoreticalRanks[pos]);
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
				contract(leftLHSStack[_position][j], measCmp, true, measCmp, false, 1);          // rl,ls -> rs
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
		//TODO: review again!
		LOG(debug, "Entering adapt_rank()");
		size_t eU = _U.order()-1; //TODO: rename
		size_t eV = _Vt.order()-1;
		REQUIRE(_U.dimensions[eU] == _S.dimensions[0] &&  _S.dimensions[1] == _Vt.dimensions[0], "Inconsistent dimensions: " << _U.dimensions << " vs " << _S.dimensions << " vs " << _Vt.dimensions);

		size_t rank, full_rank;
		for (rank=0; rank<_S.dimensions[0]; ++rank) {
			if (_S[{rank,rank}] <= smin) break;
		}
		full_rank = std::min(rank+kmin, _maxRank);

		while (_S.dimensions[0] < full_rank) {
			// Um,Un = U.size/U.dimensions[eU], U.dimensions[eU]
			// Vtm,Vtn = Vt.dimensions[0], Vt.size/Vt.dimensions[0]
			// The rank can only be increased when the dimensions of U and Vt allow it (Um > Un and Vtm < Vtn).
			REQUIRE(_U.size > misc::sqr(_U.dimensions[eU]) && _Vt.size > misc::sqr(_Vt.dimensions[0]), "IE");
			//NOTE: When called for a core move pos <--> pos+1 this condition is guaranteed by _maxRank <= maxTheoreticalRanks[pos].

			// Add a new diagonal entry with a value at 1% of the singular value threshold smin.
			_S.resize_mode(0, _S.dimensions[0]+1);
			_S.resize_mode(1, _S.dimensions[1]+1);
			_S += 0.01 * smin * Tensor::dirac(_S.dimensions, {_S.dimensions[0]-1, _S.dimensions[1]-1});
			REQUIRE(_S.sparsity() == _S.dimensions[0], "IE");

			Tensor slate, tmp;
			std::vector<size_t> slate_dimensions, slate_index;

			// Increase the size of the last mode of _U by adding a random orthogonal slate.
			_U.resize_mode(eU, _U.dimensions[eU]+1);
			slate_dimensions = std::vector<size_t>(_U.dimensions.cbegin(), _U.dimensions.cend()-1);
			slate = Tensor::random(slate_dimensions);
			slate /= slate.frob_norm();
			contract(tmp, slate, _U, eU);
			REQUIRE(tmp.order() == 1 && tmp.dimensions[0] == _U.dimensions[eU], "IE");
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
			REQUIRE(tmp.order() == 1 && tmp.dimensions[0] == _Vt.dimensions[0], "IE");
			contract(tmp, tmp, _Vt, 1);
			slate -= tmp;
			slate /= slate.frob_norm();
			slate_dimensions.insert(slate_dimensions.begin(), 1);
			slate.reinterpret_dimensions(slate_dimensions);
			slate_index = std::vector<size_t>(eV+1, 0); slate_index[0] = _Vt.dimensions[0]-1;
			_Vt.offset_add(slate, slate_index);
		}
		if (_S.dimensions[0] > full_rank && _S[{_S.dimensions[0]-1, _S.dimensions[1]-1}] < _threshold) {  // remove at most 1 rank per call
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

	double SALSA::residual(const std::pair<size_t, size_t> _slice) const {
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
		/* reshuffle(Gamma_sq, Gamma_sq, {0,2,4,1,3,5});  // axbycz -> abcxyz */
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Gamma_sq, Gamma_sq, {0,3,1,4,2,5});  // axbycz -> abcxyz
		/* REQUIRE(Gamma_sq.dimensions == Tensor::DimensionTuple{l,e,r,l,e,r}, "IE"); */

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
		/* reshuffle(Theta_sq, Theta_sq, {0,2,4,1,3,5});  // axbycz -> abcxyz */
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Theta_sq, Theta_sq, {0,3,1,4,2,5});  // axbycz -> abcxyz
		/* REQUIRE(Theta_sq.dimensions == Tensor::DimensionTuple{l,e,r,l,e,r}, "IE"); */

		LOG(debug, "Leaving omega_operator()");
		return misc::sqr(omega) * (Gamma_sq + Theta_sq);
	}

	Tensor SALSA::alpha_operator() const {  // compute LASSO regularization term
		LOG(debug, "Entering alpha_operator()");
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1];

		Tensor Op;
		if (pos == 0) { Op = Tensor::identity({1,1}); }
		else { Op = leftRegularizationStack[pos-1]; }
		contract(Op, Op, basisWeights[pos], 0);
		if (pos < M-1) { contract(Op, Op, rightRegularizationStack[pos+1], 0); }
		if (pos == x.order()-1) { Op.reinterpret_dimensions({l,l,e,e,1,1}); }
		/* reshuffle(Op, Op, {0,2,4,1,3,5});  // axbycz -> abcxyz */
		//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
		reshuffle(Op, Op, {0,3,1,4,2,5});  // axbycz -> abcxyz
		/* REQUIRE(Op.dimensions == Tensor::DimensionTuple{l,e,r,l,e,r}, "IE"); */

		LOG(debug, "Leaving alpha_operator()");
		return misc::sqr(alpha) * Op;
	}

	void SALSA::solve_local() {
		LOG(debug, "Entering solve_local()");
		//TODO: use only the training set for optimization
		const size_t pos = x.corePosition;  //TODO: rename: position
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		//TODO: split N = Nt + Nv (train and validation)
		Tensor op({l,e,r, l,e,r});
		Tensor rhs({l,e,r});
		if (pos == 0) {
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(rightStack, values) firstprivate(trainingSet, e, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=trainingSet.first; i<trainingSet.second; ++i) {
				tmp = reinterpret_dimensions(Tensor::identity({e,e}), {1,1,e,e});
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
				reshuffle(tmp, tmp, {0,3,1,4,2,5});  // lleerr -> lerler
				op += tmp;

				tmp = reinterpret_dimensions(values[i], {1,x.dimensions[0]});
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				rhs += tmp;
			}
		} else if (pos == 1) {
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftRHSStack, rightStack, measures) firstprivate(trainingSet, l, e, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=trainingSet.first; i<trainingSet.second; ++i) {
				contract(tmp, Tensor::identity({l,l}), measures[pos][i], 0);  // leftLHSStack[pos-1][i] is the identity
				contract(tmp, tmp, measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
				reshuffle(tmp, tmp, {0,3,1,4,2,5});  // lleerr -> lerler
				op += tmp;

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				rhs += tmp;
			}
		} else if (pos < M-1) {
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, rightStack, measures) firstprivate(trainingSet, pos) private(tmp) reduction(+:op,rhs)
			for (size_t i=trainingSet.first; i<trainingSet.second; ++i) {
				contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				//TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen.
				reshuffle(tmp, tmp, {0,3,1,4,2,5});  // lleerr -> lerler
				op += tmp;

				/* contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0); */
				/* contract(tmp, tmp, rightStack[pos+1][i], 0); */
				/* reshuffle(tmp, tmp, {0,2,3,1}); */
				/* contract(tmp, tmp, measures[pos][i], 0); */
				/* contract(tmp, tmp, rightStack[pos+1][i], 0); */
				/* op += tmp; */

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, rightStack[pos+1][i], 0);
				rhs += tmp;
			}
		} else {
			Tensor tmp;
			#pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions))
			#pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, measures) firstprivate(trainingSet, pos, l, e) private(tmp) reduction(+:op,rhs)
			for (size_t i=trainingSet.first; i<trainingSet.second; ++i) {
				contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0);
				contract(tmp, tmp, measures[pos][i], 0);
				reshuffle(tmp, tmp, {0,2,1,3});  // llee -> lele
				tmp.reinterpret_dimensions({l,e,1,l,e,1});
				op += tmp;

				/* tmp.reinterpret_dimensions({l,l,e,1}); */
				/* reshuffle(tmp, tmp, {0,2,3,1}); */
				/* contract(tmp, tmp, measures[pos][i], 0); */
				/* tmp.reinterpret_dimensions({l,e,1,l,e,1}); */
				/* op += tmp; */

				contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0);
				tmp.reinterpret_dimensions({l,e,1});
				rhs += tmp;
			}
		}

		/* if (pos <= 1) { */
		/*     #pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions)) */
		/*     #pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, rightStack, measures) firstprivate(N, pos, e, l) private(tmp) reduction(+:op,rhs) */
		/*     for (size_t i=0; i<N; ++i) { */
		/*         if (pos == 0) { */
		/*             tmp = Tensor::identity({e,e}); */
		/*             tmp.reinterpret_dimensions({1,1,e,e}); */
		/*         } else if (pos == 1) { */
		/*             contract(tmp, Tensor::identity({l,l}), Tensor::identity({e,e}), 0); */
		/*         } else { */
		/*             contract(tmp, leftLHSStack[pos-1][i], Tensor::identity({e,e}), 0); */
		/*         } */

		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         /1* reshuffle(tmp, tmp, {0,2,4,1,3}); *1/ */
		/*         //TODO: reshuffle funktioniert anders als in numpy! In numpy gibt die Liste an, wo die Indizes herkommen, hier gibt die Liste an, wo sie hingehen sollen. */
		/*         reshuffle(tmp, tmp, {0,3,1,4,2}); */
		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         op += tmp; */

		/*         if (pos == 0) { */
		/*             tmp = values[i]; */
		/*             tmp.reinterpret_dimensions({1,x.dimensions[0]}); */
		/*         } else { */
		/*             contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0); */
		/*         } */
		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         rhs += tmp; */
		/*     } */
		/* /1* } else if (pos == 1) { *1/ */
		/* } else if (pos < M-1) { */
		/*     #pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions)) */
		/*     #pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, rightStack, measures) firstprivate(N, pos) private(tmp) reduction(+:op,rhs) */
		/*     for (size_t i=0; i<N; ++i) { */
		/*         // firstprivate(_corePosition, _setId, dyadComp, tmp, shuffledX) default(none) */
		/*         contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0); */
		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         reshuffle(tmp, tmp, {0,2,3,1}); */
		/*         contract(tmp, tmp, measures[pos][i], 0); */
		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         op += tmp; */

		/*         contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0); */
		/*         contract(tmp, tmp, rightStack[pos+1][i], 0); */
		/*         rhs += tmp; */
		/*     } */
		/* } else { */
		/*     #pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions)) */
		/*     #pragma omp parallel for default(none) shared(leftLHSStack, leftRHSStack, measures) firstprivate(N, pos, l, e, r) private(tmp) reduction(+:op,rhs) */
		/*     for (size_t i=0; i<N; ++i) { */
		/*         // firstprivate(_corePosition, _setId, dyadComp, tmp, shuffledX) default(none) */
		/*         contract(tmp, leftLHSStack[pos-1][i], measures[pos][i], 0); */
		/*         tmp.reinterpret_dimensions({l,l,e,1}); */
		/*         reshuffle(tmp, tmp, {0,2,3,1}); */
		/*         contract(tmp, tmp, measures[pos][i], 0); */
		/*         tmp.reinterpret_dimensions({l,e,1,l,e,1}); */
		/*         op += tmp; */

		/*         contract(tmp, leftRHSStack[pos-1][i], measures[pos][i], 0); */
		/*         tmp.reinterpret_dimensions({l,e,r}); */
		/*         rhs += tmp; */
		/*     } */
		/* } */

		const Tensor op_alpha = alpha_operator();
		const Tensor op_omega = omega_operator();

		Tensor& core = x.component(pos);

		if (maxIRsteps == 0) { solve(core, op+op_alpha+op_omega, rhs); }
		else {
			Tensor IR, op_IRalpha;  // iterative reweighting
			Tensor prev_core;
			for (size_t step=0; step<maxIRsteps; ++step) {
				IR = diag(core, [sparsityThreshold=sparsityThreshold](double _entry) { return 1.0/std::sqrt(std::max(std::abs(_entry), sparsityThreshold)); });
				contract(op_IRalpha, IR, op_alpha, 3);
				contract(op_IRalpha, op_IRalpha, IR, 3);
				prev_core = core;
				solve(core, op+op_IRalpha+op_omega, rhs);
				if (frob_norm(prev_core - core) < IRtolerance*frob_norm(prev_core)) break;
			}
		}

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
		auto print_param = [max_param_len](std::string name, auto value) {
			const std::string pad(max_param_len-name.length(), ' ');
			std::cout << "  " << name << " = " << pad << value << "\n";
		};
		const std::string sep = std::string(125, '-')+"\n";
		std::cout << sep;
		print_param("dimensions", x.dimensions);
		print_param("initial_ranks", x.ranks());
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
		print_param("maxRanks", maxRanks);
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
			REQUIRE(basisWeights[m].order() == 2 && basisWeights[m].dimensions[0] == x.dimensions[m] && basisWeights[m].dimensions[1] == x.dimensions[m], "...");
		}

		// The sets have already been shuffled in the constructor. Now they need to be split.
		size_t Nt = size_t((1.0-controlSetFraction)*double(N));
		/* Nv = N - Nt; */
		trainingSet = std::make_pair(0, Nt);
		validationSet = std::make_pair(Nt, N);

		// build stacks and compute left and right singular value arrays
		x.move_core(x.order()-1);
		while (x.corePosition > 0) { move_core_left(false); }

		valueNorm_trainingSet = 0.0;
		#pragma omp parallel for default(none) shared(values) firstprivate(trainingSet) reduction(+:valueNorm_trainingSet)
		for(size_t j=trainingSet.first; j<trainingSet.second; ++j) {
			valueNorm_trainingSet += misc::sqr(frob_norm(values[j]));
		}
		valueNorm_trainingSet = std::sqrt(valueNorm_trainingSet);

		// compute SALSA parameters
		double res = residual(std::make_pair(0, N));
		double maxResSqrtRes = std::max(res, std::sqrt(res));

		alpha = alpha_factor * std::min(valueNorm_trainingSet, maxResSqrtRes);
		omega = maxResSqrtRes;
		smin = 0.2*std::min(omega, res);
		omega *= omega_factor;
		LOG(debug, "Leaving initialize()");
	}

	std::string SALSA::print_fractional_ranks() const {
		LOG(debug, "Entering print_fractional_ranks()");
		/* assert len(self.singularValues) == self.x.order()-1 */
		/* for i in range(self.x.order()-1): */
		/*     assert len(self.singularValues[i]) == self.x.rank(i) */
		const std::string grey_30 = u8"\033[38;5;239m";
		const std::string grey_50 = u8"\033[38;5;244m";
		const std::string reset = u8"\033[0m";
		std::ostringstream ranks;
		for (size_t m=0; m<M-1; ++m) {
			/* assert np.all(singularValues[m] > 0) */
			size_t rank;
			for (rank=0; rank<singularValues[m].size(); ++rank) {
				if (singularValues[m][rank] <= smin) break;
			}
			ranks << rank;
			double inactive = 0.0;
			if (rank < singularValues[m].size()) {
				inactive = singularValues[m][rank]/smin;
				REQUIRE(0 < inactive && inactive < 1, "IE");
			}
			/* ranks << wchar_t(2070+size_t(10*inactive)) << L'\u002F' << singularValues[m].size() << L' '; */
			ranks << grey_50 << "." << size_t(10*inactive) << reset << u8"\u002F" << grey_30 << singularValues[m].size() << reset << " ";
		}
		std::string output = "[" + ranks.str();
		output.replace(output.length()-1, 1, "]");
		LOG(debug, "Leaving print_fractional_ranks()");
		return output;
	}

	std::string SALSA::print_densities() const {
		LOG(debug, "Entering print_densities()");
		std::ostringstream densities;
		for (size_t m=0; m<M; ++m) {
			size_t d = std::min(size_t(100*weightedNorms[m]+0.5), size_t{99});
			densities << string_format("%2u ", d);
			/* if (d < 10) densities << " " << d << " "; */
			/* else densities << d << " "; */
		}
		std::string output = "[" + densities.str();
		output.replace(output.length()-1, 1, "]%");
		LOG(debug, "Leaving print_densities()");
		return output;
	}

	void SALSA::run() {
		LOG(debug, "Entering run()");
		std::cout << std::string(125, '=') << '\n'
					<< std::string(55, ' ') << "Running uqSALSA" << std::endl;
		print_parameters();
		initialize();

		REQUIRE(x.corePosition == 0, "IE");
		// sweep left -> right
		for (size_t m=0; m<M-1; ++m) {
			REQUIRE(x.corePosition == m, "IE");
			solve_local();
			move_core_right(false);
		}
		// sweep right -> left
		for (size_t m=M-1; m>0; --m) {
			REQUIRE(x.corePosition == m, "IE");
			solve_local();
			move_core_left(false);
		}
		REQUIRE(x.corePosition == 0, "IE");

		/* boost::circular_buffer<double> trainingResiduals{trackingPeriodLength, std::numeric_limits<double>::max()};  // Creates a full circular buffer with every element equal to the max. double. */
		boost::circular_buffer<double> trainingResiduals{trackingPeriodLength};
		std::vector<double> validationResiduals;
		trainingResiduals.push_back(residual(trainingSet));
		validationResiduals.push_back(residual(validationSet));

		double initialResidual = trainingResiduals.back();  //TODO: rename
		size_t bestIteration = 0;
		TTTensor bestX = x;
		/* double bestAlpha = alpha; */
		double bestTrainingResidual = trainingResiduals.back();
		double bestValidationResidual = validationResiduals.back();
		double prev_bestValidationResidual = validationResiduals.back();
		/* double bestValidationResidual_cycle = bestValidationResidual; */
		//TODO: der ganze best-kram (außer prev_bestValidationResidual) und initialResidual sollte vllt class attributes werden

		size_t iteration = 0;
		size_t nonImprovementCounter = 0;
		bool omegaMinimal = false;

		auto print_update = [&](){
			auto update_str = [](double prev, double cur)  {
				std::ostringstream ret;
				if (cur <= prev+1e-8) ret << "\033[38;5;151m" << string_format("%.2e", cur) << "\033[0m";
				else ret << "\033[38;5;181m" << string_format("%.2e", cur) << "\033[0m";
				return ret.str();
			};
			std::cout << "[" << iteration << "]"
						<< "Residuals: trn=" << update_str(bestTrainingResidual , trainingResiduals.back())
						<< ", val=" << update_str(bestValidationResidual , validationResiduals.back())
						<< "  |  Omega: " << string_format("%.2e", omega)
						<< "  |  Densities: " << print_densities();
			std::cout << "  |  Ranks: " << print_fractional_ranks() << std::endl;
		};
		print_update();

		for (; iteration<maxIterations; ++iteration) {
			REQUIRE(x.corePosition == 0, "IE");
			// sweep left -> right
			for (size_t m=0; m<M-1; ++m) {
				REQUIRE(x.corePosition == m, "IE");
				solve_local();
				move_core_right(true);
			}
			// sweep right -> left
			for (size_t m=M-1; m>0; --m) {
				REQUIRE(x.corePosition == m, "IE");
				solve_local();
				move_core_left(true);
			}
			REQUIRE(x.corePosition == 0, "IE");

			trainingResiduals.push_back(residual(trainingSet));
			validationResiduals.push_back(residual(validationSet));
			print_update();
			bestTrainingResidual = std::min(trainingResiduals.back(), bestTrainingResidual);

			if (validationResiduals.back() < (1-minDecrease)*bestValidationResidual) {
				bestIteration = iteration;
				bestX = x;
				/* bestAlpha = alpha; */
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
			REQUIRE(omega >= res, "IE");
			omegaMinimal = misc::approx_equal(omega, res, 1e-6);
			smin = 0.2*std::min(omega, res);
			omega *= omega_factor;

			if (trainingResiduals.size() >= trackingPeriodLength && trainingResiduals.back() > (1-minDecrease)*trainingResiduals.front() && omegaMinimal) {
				REQUIRE(trainingResiduals.size() == trackingPeriodLength, "IE"); // circular buffer... TODO: rewrite...

				if (bestIteration > iteration-validationResiduals.size()) {
					nonImprovementCounter = 0;
					std::cout << "NonImpCnt: " << nonImprovementCounter << string_format(" (val=%.2e ", prev_bestValidationResidual);
					std::cout << std::string(u8"\u2198");
					std::cout << string_format(" val=%.2e)", bestValidationResidual) << std::endl;
				} else {
					nonImprovementCounter += 1;
					/* nonImpCnt_str = f"NonImpCnt: {{0:{len(str(self.maxNonImprovingAlphaCycles))}d}}".format */
					/* print(nonImpCnt_str(nonImprovementCounter), f"(val={bestValidationResidual:.2e} \u2197 val={min(validationResiduals):.2e})")  #  Best validation residual: {...}. */
					std::cout << "NonImpCnt: " << nonImprovementCounter << string_format(" (val=%.2e ", prev_bestValidationResidual);
					std::cout << std::string(u8"\u2197");
					std::cout << string_format(" val=%.2e)", bestValidationResidual) << std::endl;
				}

				if (nonImprovementCounter >= maxNonImprovingAlphaCycles) {
					std::cout << "Terminating: Minimum residual decrease deceeded " << nonImprovementCounter << " iterations in a row." << std::endl;
					break;
				}

				res = validationResiduals.back() * valueNorm_trainingSet;
				double prev_alpha = alpha;
				alpha = alpha/alpha_factor;
				alpha = std::min(alpha/falpha, std::sqrt(res));
				alpha *= alpha_factor;
				std::cout << "Reduce alpha: " << string_format("%.3f", prev_alpha);
				std::cout << std::string(u8" \u2192 ");
				std::cout << string_format("%.3f", alpha) << std::endl;
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

		x = bestX;
		x.round(maxRanks);
		/* assert np.all(self.x.ranks() <= np.asarray(self.maxRanks)) */

		std::cout << string_format("Residual decreased from %.2e to %.2e in %u iterations.\n", initialResidual, bestTrainingResidual, iteration)
					<< std::string(125, '=') << std::endl;
		//TODO: den ganzen kram hier in eine `finish`-Routine auslagern.
		LOG(debug, "Leaving run()");
	}

}} // namespace uq | xerus
