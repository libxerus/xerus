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
/* #include <unistd.h>  // gethostname */
#include <limits.h>     // HOST_NAME_MAX

#include <xerus/applications/uqSALSA.h>
#include <xerus/misc/check.h>

#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <numeric>
#include <chrono>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

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

#if defined(_OPENMP)
	#include <omp.h>
#endif

#if defined(PROFILING)
	#pragma message("WARNING: Compiling in profiling mode.")
	#if defined(_OPENMP)
		#pragma message("WARNING: OMP enabled.")
	#endif
#endif


namespace xerus { namespace uq {

	// constexpr std::size_t operator "" z(unsigned long long n) { return n; }  // size_t literal

	class Timer {
		using Clock    = std::chrono::high_resolution_clock;
		using Millisecond = std::chrono::duration<double, std::ratio<1,1000>>;
		std::chrono::time_point<Clock> start;

	public:
		Timer() : start(Clock::now()) {}
		void reset() { start = Clock::now(); }
		double elapsed() const {
			return std::chrono::duration_cast<Millisecond>(Clock::now() - start).count();
		}
	};

	struct Foreground {
		static const std::string Good;
		static const std::string Bad;
		static const std::string Alert;
		static const std::string Separate;
		static const std::string Reset;

		static const int GrayscaleBegin;
		static const int GrayscaleEnd;

		static auto Gray(const double _value) {
			REQUIRE(0 <= _value && _value <= 1, "_value must lie between 0 and 1");
			const auto gsb = static_cast<int>(GrayscaleBegin);
			const auto gse = static_cast<int>(GrayscaleEnd);
			return "\033[38;5;" + std::to_string(gsb + int(_value*(gse-gsb))) + "m";
		}
	};
	const std::string Foreground::Good     = "\033[38;5;151m";
	const std::string Foreground::Bad      = "\033[38;5;181m";
	const std::string Foreground::Alert    = "\033[38;5;230m";
	const std::string Foreground::Separate = "\033[38;5;153m";
	const std::string Foreground::Reset    = "\033[39m";
	const int Foreground::GrayscaleBegin = 236;
	const int Foreground::GrayscaleEnd = 256;

	template<typename ... Args>
	std::string string_format( const std::string& format, Args ... args )
	{
		size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
		REQUIRE(size > 0, "Error during formatting.");
		std::unique_ptr<char[]> buf( new char[ size ] );
		snprintf( buf.get(), size, format.c_str(), args ... );
		return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
	}

	template<typename T, typename Function>
	std::string print_list(const std::vector<T>& _list, const Function& _formatter) {
		std::ostringstream stream;
		for (size_t i=0; i<_list.size(); ++i) {
			stream << _formatter(_list[i]) << " ";
		}
		std::string output = "[" + stream.str();
		output.replace(output.length()-1, 1, "]");
		return output;
	}

	template<typename T>
	std::string print_list(const std::vector<T>& _list) {
		return print_list<T, std::string(&)(T)>(_list, &std::to_string);
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

	std::vector<Tensor> identities(const std::vector<size_t>& _dimensions) {
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

	//TODO: A reshuffing version of modify_rank_one that returns a summand with permuted dimensions.
	/* template<size_t Order, typename Function> */
	/* void modify_rank_one(value_t* _summand, const std::array<const value_t*, Order>&& _factors, const std::array<size_t, Order>&& _dimensions, const std::array<size_t, Order>&& _permutation, Function&& _modifier) { */

	template<size_t Order, typename Function>
	void modify_rank_one(value_t* _summand, const std::array<const value_t*, Order>&& _factors, const std::array<size_t, Order>&& _dimensions, Function&& _modifier) { //void (*_modifier)(value_t&, const value_t)) {
		// Der cache sorgt dafür, dass auf die äußeren Arrays nur selten zugegriffen werden muss. -> cache locality für die inneren!
		// Das cumprod sorgt dafür, dass die Produkte immer nur so weit neu berechnet werden müssen, wie nötig.
		/* Timer timer; */
		std::array<size_t, Order> index;
		std::array<value_t, Order+1> cumprod_cache;
		cumprod_cache[0] = value_t(1.0);
		size_t mode;
		for (mode=0; mode<Order; ++mode) {
			index[mode] = 0;
			cumprod_cache[mode+1] = cumprod_cache[mode] * _factors[mode][0];
		}
		size_t position = 0;
		while (true) {
			_modifier(_summand[position++], cumprod_cache[Order]);
			mode = Order-1;
			while(unlikely(++index[mode] == _dimensions[mode])) {
				index[mode] = 0;
				if(unlikely(--mode >= Order)) {
					/* const double elapsed = timer.elapsed(); */
					/* std::cout << "Time for modify_rank_one<" << Order << ">(dimensions=" << std::vector<size_t>(_dimensions.begin(), _dimensions.end()) << "): " << string_format("%.2fms\n", Order, elapsed); */
					return;
				}  // Return on overflow
			}
			for (; mode<Order; ++mode) {
				cumprod_cache[mode+1] = cumprod_cache[mode] * _factors[mode][index[mode]];
			}
		}
	}

	template<size_t Order, typename Function>
	void modify_rank_one(Tensor& _summand, const std::array<const Tensor, Order>&& _factors, Function&& _modifier) {  // void (*_modifier)(value_t&, const value_t)) {  //const std::function<void(value_t&, const value_t)>&& _modifier) {
		// _summand must be an already initialized Tensor of suitable size
		std::array<const value_t*, Order> factors;
		std::array<size_t, Order> dimensions;
		for (size_t m=0; m<Order; ++m) {
			REQUIRE(!_factors[m].has_factor(), "IE");
			factors[m] = _factors[m].get_unsanitized_dense_data();
			dimensions[m] = _factors[m].size;
		}
		modify_rank_one(_summand.get_dense_data(), std::move(factors), std::move(dimensions), std::move(_modifier));
	}

	template<size_t Order>
	void add_rank_one(value_t* _summand, const std::array<const value_t*, Order>&& _factors, const std::array<size_t, Order>&& _dimensions) {
		modify_rank_one<Order>(_summand, std::move(_factors), std::move(_dimensions), [](value_t& _summandEntry, const value_t _productEntry) {
			_summandEntry += _productEntry;
		});
	}

	template<size_t Order>
	void add_rank_one(Tensor& _summand, const std::array<const Tensor, Order>&& _factors) {
		modify_rank_one<Order>(_summand, std::move(_factors), [](value_t& _summandEntry, const value_t _productEntry) {
			_summandEntry += _productEntry;
		});
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
		basisWeights(identities(x.dimensions))
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

		// move the core (pos-1)<--(pos) i.e. adapt x.rank(pos-1)
		calculate_svd(U, S, Vt, old_core, 1, 0, 0);
		// splitPos == 1 --> U(left,r1) * S(r1,r2) * Vt(r2,ext,right) == old_core(left,ext,right)   (The left part has 1 external index.)
		// maxRank == 0  --> do not perform hard thresholding
		// eps == 0      --> do not round to eps% of norm.
		REQUIRE(Tensor::DimensionTuple(U.dimensions.begin(), U.dimensions.end()-1) == Tensor::DimensionTuple(old_core.dimensions.begin(), old_core.dimensions.begin()+1), "IE");
		REQUIRE(Tensor::DimensionTuple(Vt.dimensions.begin()+1, Vt.dimensions.end()) == Tensor::DimensionTuple(old_core.dimensions.begin()+1, old_core.dimensions.end()), "IE");

		old_core = Vt;
		contract(new_core, new_core, U, 1);  // new_core(i^2,j) << new_core(i^2,l) * U(l,j)

		if (adapt) {
			size_t maxRank = std::min(maxRanks[pos-1], std::numeric_limits<size_t>::max()-kmin) + kmin;
			REQUIRE(maxRank >= maxRanks[pos-1], "IE");
			double threshold = 0.1*smin;  //TODO: in the unchecked (i.e. commented out) version of vresalsa threshold = 0.1*self.residual(self.trainingSet)
			adapt_rank(new_core, S, old_core, maxRank, threshold);
		}

		contract(new_core, new_core, S, 1);  // new_core(i^2,j) << new_core(i^2,l) * S(l,j)
		REQUIRE(new_core.all_entries_valid() && old_core.all_entries_valid(), "IE");
		x.nodes[pos].neighbors[2].dimension = new_core.dimensions[2];
		x.nodes[pos+1].neighbors[0].dimension = old_core.dimensions[0];
		x.assume_core_position(pos-1);

		calc_right_stack(pos);

		singularValues[pos-1].resize(S.dimensions[0]);
		for (size_t i=0; i<S.dimensions[0]; ++i) {
			singularValues[pos-1][i] = S[{i,i}];
		}

		if (initialized && 1 < pos) {
			Tensor& next_core = x.component(pos-2);
			// move the core (pos-2)<--(pos-1) i.e. adapt x.rank(pos-1)
			calculate_svd(U, S, Vt, new_core, 1, 0, 0);  // (U(left,r1), S(r1,r2), Vt(r2,ext,right)) = new_core(left,ext,right)
			// REQUIRE(U.order() == 2 && U.dimensions[0] == U.dimensions[1], "IE(" << pos << ") " << new_core.dimensions << " / " << U.dimensions << " / " << S.dimensions << " / " << Vt.dimensions);
			//TODO: This is not the case when new_core has a rank that is lower than new_core.dimenions[0].
			REQUIRE(U.order() == 2, "IE");
			if (U.dimensions[0] != U.dimensions[1]) {
				REQUIRE(U.dimensions[0] > U.dimensions[1], "IE");
				const auto attr = [](const unsigned _code) -> std::string { return u8"\033["+std::to_string(_code)+"m"; };
				const auto alert = Foreground::Alert + attr(1);
				const auto reset = attr(0);
				std::cout << alert + "WARNING: Real rank(" << pos-2 << ") is " << U.dimensions[1] << " and not " << U.dimensions[0] << reset + "\n";
			}

			contract(next_core, next_core, U, 1);
			contract(new_core, S, Vt, 1);
			REQUIRE(new_core.all_entries_valid() && next_core.all_entries_valid(), "IE");
			x.nodes[pos-1].neighbors[2].dimension = next_core.dimensions[2];
			x.nodes[pos].neighbors[0].dimension = new_core.dimensions[0];

			calc_left_stack(pos-2);
			// You could also contract U directly to the operator in solve_local:
			// Then you would minimize `(U@core).T @ op @ (U@core) - (U@core).T @ rhs` for `core`.
			// The problem with that is that the operator is large and multiplication of U requires a reshuffling of its modes (in this case: lerler --> lererl).
			// Another strategy (which is currently employed) is to perform the multiplication with U during the assembly of the operator (during the summation).
			// In this state the individual parts are still small and reshuffling is cheap. The cheapest way is to contract U directly to `U.T @ leftLHSStack @ U` and `leftRHSStack @ U`.
			// This is essentially what calc_left_stack does.
			//
			//TODO: A new (AND POTENTIALLY BETTER) idea is to solve the minimization problem not for `core` but for `mcore = U@core` and obtain core via `core = U.T@mcore`.
			//TODO: A similar strategy: Do not reshuffle the operator but just the RHS: (op : reller, rhs : ler) and let R be the reshuffling operator: (rel -> ler) the you would solve R@op@x == rhs.
			//      But you can also solve op@x == inv(R)@rhs!
			//TODO: Beide Ideen funktionieren deswegen nicht so gut, weil wir dann auch alpha_op und omega_op anpassen müssten. (Eigentlich nur omega_op und der ist sparse...)
			//
			//TODO: Die folgende Beschreibeung (mit Vt) ist für move_core_right.
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

		// REQUIRE(resOnExit <= (1.0+1e-12)*resOnEnter, string_format("Residual increased in core move (%.8e --> %.8e)", res_enter, res_exit));
		//TODO: Wenn adapt=true kann sich das Residuum in einem core-move durchaus ändern.
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

		// move the core (pos)-->(pos+1) i.e. adapt x.rank(pos)
		calculate_svd(U, S, Vt, old_core, 2, 0, 0);  // (U(left,ext,r1), S(r1,r2), Vt(r2,right)) = old_core(left,ext,right)
		REQUIRE(Tensor::DimensionTuple(U.dimensions.begin(), U.dimensions.end()-1) == Tensor::DimensionTuple(old_core.dimensions.begin(), old_core.dimensions.begin()+2), "IE");
		REQUIRE(Tensor::DimensionTuple(Vt.dimensions.begin()+1, Vt.dimensions.end()) == Tensor::DimensionTuple(old_core.dimensions.begin()+2, old_core.dimensions.end()), "IE");

		old_core = U;
		contract(new_core, Vt, new_core, 1);  // new_core(i,j^2) << Vt(i,r) * new_core(r,j^2)

		if (adapt) {
			size_t maxRank = std::min(maxRanks[pos], std::numeric_limits<size_t>::max()-kmin) + kmin;
			REQUIRE(maxRank >= maxRanks[pos], "IE");
			double threshold = 0.1*smin;  //TODO: in the unchecked (i.e. commented out) version of vresalsa threshold = 0.1*self.residual(self.trainingSet)
			adapt_rank(old_core, S, new_core, maxRank, threshold);
		}

		contract(new_core, S, new_core, 1);  // new_core(i,j^2) << S(i,l) * new_core(l,j^2)
		REQUIRE(new_core.all_entries_valid() && old_core.all_entries_valid(), "IE");
		x.nodes[pos+1].neighbors[2].dimension = old_core.dimensions[2];
		x.nodes[pos+2].neighbors[0].dimension = new_core.dimensions[0];
		x.assume_core_position(pos+1);

		calc_left_stack(pos);

		singularValues[pos].resize(S.dimensions[0]);
		for (size_t i=0; i<S.dimensions[0]; ++i) {
			singularValues[pos][i] = S[{i,i}];
		}

		if (pos+2 < x.order()) {
			Tensor& next_core = x.component(pos+2);
			// move the core (pos+1)-->(pos+2) i.e. adapt x.rank(pos+1)
			calculate_svd(U, S, Vt, new_core, 2, 0, 0);  // (U(left,ext,r1), S(r1,r2), Vt(r2,right)) = new_core(left,ext,right)
			// REQUIRE(Vt.order() == 2 && Vt.dimensions[0] == Vt.dimensions[1], "IE");
			//TODO: This is not the case when new_core has a rank that is lower than new_core.dimenions[2].
			REQUIRE(Vt.order() == 2, "IE");
			if (Vt.dimensions[0] != Vt.dimensions[1]) {
				REQUIRE(Vt.dimensions[0] < Vt.dimensions[1], "IE");
				const auto attr = [](const unsigned _code) -> std::string { return u8"\033["+std::to_string(_code)+"m"; };
				const auto alert = Foreground::Alert + attr(1);
				const auto reset = attr(0);
				std::cout << alert + "WARNING: Real rank(" << pos+1 << ") is " << Vt.dimensions[0] << " and not " << Vt.dimensions[1] << reset + "\n";
			}

			contract(next_core, Vt, next_core, 1);
			contract(new_core, U, S, 1);
			REQUIRE(new_core.all_entries_valid() && next_core.all_entries_valid(), "IE");
			x.nodes[pos+2].neighbors[2].dimension = new_core.dimensions[2];
			x.nodes[pos+3].neighbors[0].dimension = next_core.dimensions[0];

			calc_right_stack(pos+2);  //TODO: see move_core_left

			singularValues[pos+1].resize(S.dimensions[0]);
			for (size_t i=0; i<S.dimensions[0]; ++i) {
				singularValues[pos+1][i] = S[{i,i}];
			}
		}

		// REQUIRE(resOnExit <= (1.0+1e-12)*resOnEnter, string_format("Residual increased in core move (%.8e --> %.8e)", res_enter, res_exit));
		//TODO: Wenn adapt=true kann sich das Residuum in einem core-move durchaus ändern.
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
			#pragma omp parallel for default(none) firstprivate(N, _position, shuffledX) schedule(static)
			for(size_t i = 0; i < N; ++i) {
				//NOTE: The first component is contracted directly (leftLHSStack[0] = shuffledX.T @ shuffledX).
				//NOTE: Since shuffeldX is left-orthogonal leftLHSStack[0] is the identity.
				contract(leftRHSStack[_position][i], values[i], shuffledX, 1);  // e,er -> r
			}
		} else if(_position == 1) {
			Tensor measCmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			#pragma omp parallel for default(none) firstprivate(N, _position, shuffledX) private(measCmp) schedule(static)
			for(size_t j = 0; j < N; ++j) {
				contract(measCmp, measures[_position][j], shuffledX, 1);                         // ler,e -> lr
				//NOTE: leftLHSStack[0] is the identity
				contract(leftLHSStack[_position][j], measCmp, true, measCmp, false, 1);          // lr,ls -> rs
				contract(leftRHSStack[_position][j], leftRHSStack[_position-1][j], measCmp, 1);  // r,rs  -> s
			}
		} else {
			Tensor measCmp, tmp;
			const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			#pragma omp parallel for default(none) firstprivate(N, _position, shuffledX) private(measCmp, tmp) schedule(static)
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
			#pragma omp parallel for default(none) firstprivate(N, _position, shuffledX) private(measCmp) schedule(static)
			for(size_t j = 0; j < N; ++j) {
				contract(measCmp, measures[_position][j], shuffledX, 1);
				contract(rightStack[_position][j], measCmp, rightStack[_position+1][j], 1);
			}
		} else {  // _position == M-1
			const Tensor shuffledX = reinterpret_dimensions(x.get_component(_position), {x.rank(M-2), x.dimensions[M-1]});  // Remove dangling 1-mode
			#pragma omp parallel for default(none) firstprivate(N, _position, shuffledX) schedule(static)
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
		const size_t from = _slice.first, to = _slice.second;
		LOG(debug, "Entering residual((" << from << ", " << to << "))");
		REQUIRE(x.corePosition == 0, "IE");
		REQUIRE(from <= to && to <= N, "IE");
		const Tensor shuffledX = reinterpret_dimensions(x.get_component(0), {x.dimensions[0], x.rank(0)});  // Remove dangling 1-mode
		Tensor tmp;
		double res = 0.0, valueNorm = 0.0;
		#pragma omp parallel for default(none) firstprivate(from, to, shuffledX) private(tmp) reduction(+:res,valueNorm) schedule(static)
		for(size_t j = from; j < to; ++j) {
			contract(tmp, shuffledX, rightStack[1][j], 1);
			res += misc::sqr(frob_norm(values[j] - tmp));
			valueNorm += misc::sqr(frob_norm(values[j]));
		}
		LOG(debug, "Leaving residual()");
		return std::sqrt(res/valueNorm);
	}

	template<typename T>
	void apply_permutation(std::vector<T>& _out, const std::vector<T>& _in, const std::vector<size_t>& _permutation) {
		// Prevent corruption of _in in the case it coincides with _out.
		std::vector<T> store;
		const std::vector<T>& base = [&]() {
			if(&_out == &_in) { store = _in; return store; }
			return _in;
		}();

		_out.resize(base.size());
		for (size_t i=0; i<base.size(); i++) {
			_out[i] = base[_permutation[i]];
		}
	}

	template<typename T>
	void apply_inverse_permutation(std::vector<T>& _out, const std::vector<T>& _in, const std::vector<size_t>& _permutation) {
		// Prevent corruption of _in in the case it coincides with _out.
		std::vector<T> store;
		const std::vector<T>& base = [&]() {
			if(&_out == &_in) { store = _in; return store; }
			return _in;
		}();

		_out.resize(base.size());
		for (size_t i=0; i<base.size(); i++) {
			_out[_permutation[i]] = base[i];
		}
	}

	void reshuffle_modes(Tensor& _out, const Tensor& _base, const std::vector<size_t>& _permutation) {
		//TODO: reverse the order of index, ... (cache lines...)
		//TODO: When the first index does not change then the shuffling can be parallelized: arr = emtpy({5,7,11}); reshuffle(arr, arr, {0,2,1}) == for i in range(5): reshuffle(arr[i], arr[i], {1,0})
		/* Timer timer; */
		REQUIRE(_base.is_dense(), "can only reshuffle dense tensors");

		// Prevent corruption of _base in the case it coincides with _out.
		Tensor store;
		const Tensor& base = [&]() {
			if(&_out == &_base) { store = _base; return store; }
			return _base;
		}();

		const std::vector<size_t>& dimensions = base.dimensions;

		size_t blockSize = 1;
		size_t ROrder = dimensions.size();  // Reshuffling Order (order of the first modes that actually get reshuffled)
		while (ROrder>0 && _permutation[ROrder-1] == ROrder-1) {
			ROrder--;
			blockSize *= dimensions[ROrder];
		}

		if(unlikely(!ROrder)) { // No actual reshuffling
			_out = _base;
			return;
		}

		std::vector<size_t> permuted_dimensions;
		apply_permutation(permuted_dimensions, dimensions, _permutation);
		std::vector<size_t> IPOutStrides(ROrder);  // Inversely Permuted Output Strides
		IPOutStrides[ROrder-1] = blockSize;
		for (size_t m=ROrder-1; m-->0;) {
			IPOutStrides[m] = permuted_dimensions[m+1] * IPOutStrides[m+1];
		}
		apply_inverse_permutation(IPOutStrides, IPOutStrides, _permutation);
		std::vector<size_t> index(ROrder);

		_out.reset(std::move(permuted_dimensions), Tensor::Representation::Dense, Tensor::Initialisation::None);
		_out.factor = base.factor;

		value_t* outPosition = _out.get_dense_data();
		const value_t* basePosition = base.get_unsanitized_dense_data();
		while (true) {
			misc::copy(outPosition, basePosition, blockSize);

			size_t mode = ROrder-1;
			basePosition += blockSize;
			outPosition  += IPOutStrides[mode];
			while(unlikely(++index[mode] == dimensions[mode])) {
				outPosition -= dimensions[mode]*IPOutStrides[mode];
				index[mode] = 0;
				if(unlikely(--mode >= ROrder)) {
					/* const double elapsed = timer.elapsed(); */
					/* std::cout << "Time for reshuffle_modes(dimensions=" << base.dimensions << ", permutation=" << _permutation << string_format("): %.2fms\n", elapsed); */
					return;
				}  // Return on overflow
				outPosition += IPOutStrides[mode];
			}
		}
	}

	Tensor SALSA::omega_operator() const {  // compute SALSA regularization term
		/* LOG(profiling, "Entering omega_operator()"); */
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		// compute left part
		Tensor Gamma_sq;
		if (pos == 0) {
			Gamma_sq = x.frob_norm() * Tensor::identity({l,e,r,l,e,r});
		} else {
			Gamma_sq = Tensor({l,e,r,l,e,r}, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);
			std::map<size_t, value_t>& Gamma_sq_data = Gamma_sq.get_sparse_data();
			auto hint = Gamma_sq_data.begin();
			size_t position = 0;
			for (size_t i=0; i<l; ++i) {
				const value_t value = 1.0 / misc::sqr(std::max(smin, singularValues[pos-1][i]));
				for (size_t j=0; j<e; ++j) {
					for (size_t k=0; k<r; ++k) {
						/* Gamma_sq[position] = value; */
						Gamma_sq_data.emplace_hint(hint, position, value);
						hint = Gamma_sq_data.end();
						position += (l*e*r+1);
					}
				}
			}
		}

		// compute right part
		Tensor Theta_sq;
		if (pos < M-1) {
			Theta_sq = Tensor({l,e,r,l,e,r}, Tensor::Representation::Sparse, Tensor::Initialisation::Zero);
			std::map<size_t, value_t>& Theta_sq_data = Theta_sq.get_sparse_data();
			auto hint = Theta_sq_data.begin();
			size_t position = 0;
			for (size_t i=0; i<l; ++i) {
				for (size_t j=0; j<e; ++j) {
					for (size_t k=0; k<r; ++k) {
						const value_t value = 1.0 / misc::sqr(std::max(smin, singularValues[pos][k]));
						/* Theta_sq[position] = value; */
						Theta_sq_data.emplace_hint(hint, position, value);
						hint = Theta_sq_data.end();
						position += (l*e*r+1);
					}
				}
			}
		} else {
			Theta_sq = x.frob_norm() * Tensor::identity({l,e,r,l,e,r});
		}

		REQUIRE(Gamma_sq.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");
		REQUIRE(Theta_sq.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");
		/* LOG(profiling, "Leaving omega_operator()"); */
		return misc::sqr(omega) * (Gamma_sq + Theta_sq);
	}

	Tensor SALSA::alpha_operator() const {  // compute LASSO regularization term
		/* Timer timer; */
		/* LOG(profiling, "Entering alpha_operator(position=" << x.corePosition << ")"); */
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		Tensor op;
		if (pos == 0) {
			Tensor tmp({e*e,r*r}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			// The ordering (r << e) is important for cache locality!
			modify_rank_one<2>(tmp, { basisWeights[pos], rightRegularizationStack[1] }, [](value_t& _opEntry, const value_t _productEntry){
				_opEntry = _productEntry;
			});
			tmp.reinterpret_dimensions({e,e,r,r});

			reshuffle_modes(op, tmp, {0,2,1,3});
			op.reinterpret_dimensions({1,e,r,1,e,r});
		} else if (pos < M-1) {
			Tensor tmp({e*e,r*r}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			// The ordering (r << e) is important for cache locality!
			modify_rank_one<2>(tmp, { basisWeights[pos], rightRegularizationStack[pos+1] }, [](value_t& _opEntry, const value_t _productEntry){
				_opEntry = _productEntry;
			});
			tmp.reinterpret_dimensions({e,e,r,r});
			reshuffle_modes(tmp, tmp, {0,2,1,3});
			op = Tensor({l*l,e*e*r*r}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			modify_rank_one<2>(op, { leftRegularizationStack[pos-1], tmp }, [](value_t& _opEntry, const value_t _productEntry){
				_opEntry = _productEntry;
			});
			op.reinterpret_dimensions({l,l,e,r,e,r});
			reshuffle_modes(op, op, {0,2,3,1,4,5});
		} else {
			Tensor tmp({l*l,e*e}, Tensor::Representation::Dense, Tensor::Initialisation::None);
			// The ordering (l << e) would be better for cache locality but this ordering allows for faster reshuffling.
			modify_rank_one<2>(tmp, { leftRegularizationStack[pos-1], basisWeights[pos] }, [](value_t& _opEntry, const value_t _productEntry){
				_opEntry = _productEntry;
			});
			tmp.reinterpret_dimensions({l,l,e,e});
			reshuffle_modes(op, tmp, {0,2,1,3});
			op.reinterpret_dimensions({l,e,1,l,e,1});
		}
		REQUIRE(op.dimensions == Tensor::DimensionTuple({l,e,r,l,e,r}), "IE");

		/* LOG(profiling, "Leaving alpha_operator()"); */
		/* const double elapsed = timer.elapsed(); */
		/* std::cout << string_format("Time for alpha_operator(position=%d): %.2fms\n", pos, elapsed); */
		return misc::sqr(alpha) * op;
	}

	std::pair<Tensor, Tensor> SALSA::ls_operator_and_rhs(const std::pair<size_t, size_t>& _slice) const {
		/* Timer timer; */
		/* LOG(profiling, "Entering ls_operator((" << _slice.first << ", " << _slice.second << "))"); */
		const size_t pos = x.corePosition;
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		#pragma omp declare reduction(+ : Tensor : omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions, Tensor::Representation::Dense, Tensor::Initialisation::Zero))

		Tensor op;
		Tensor rhs;
		if (pos == 0) {
			Tensor tmp({r,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			rhs = Tensor({e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (r << e) is important for cache locality!

			#pragma omp parallel for default(none) firstprivate(_slice, pos) reduction(+:tmp,rhs) schedule(static)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				add_rank_one<2>(tmp, {rightStack[pos+1][i], rightStack[pos+1][i]});
				add_rank_one<2>(rhs, {values[i], rightStack[pos+1][i]});
			}
			contract(op, Tensor::identity({e,e}), tmp, 0);  // The values contracted together.
			reshuffle(op, op, {0,2,1,3});  // eerr -> erer
			//NOTE: sparse arrays have to be reshuffled with xerus::reshuffle which works with a different permutation as reshuffle_modes.
			op.reinterpret_dimensions({1,e,r,1,e,r});
			rhs.reinterpret_dimensions({1,e,r});
		} else if (pos == 1) {
			Tensor tmp({e,r,e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (r << e) is important for cache locality!
			rhs = Tensor({l,e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (l,r << e) is important for cache locality!

			#pragma omp parallel for default(none) firstprivate(_slice, pos) reduction(+:tmp,rhs) schedule(static)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				add_rank_one<4>(tmp, {measures[pos][i], rightStack[pos+1][i], measures[pos][i], rightStack[pos+1][i]});
				add_rank_one<3>(rhs, {leftRHSStack[pos-1][i], measures[pos][i], rightStack[pos+1][i]});
			}

			op = Tensor({l,l,e,r,e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			Tensor E = Tensor::identity({l,l}); E.use_dense_representation();
			modify_rank_one<2>(op, { E, tmp }, [](value_t& _opEntry, const value_t _productEntry) {
				_opEntry = _productEntry;
			});
			reshuffle_modes(op, op, {0,2,3,1,4,5});  // llerer -> lerler
		} else if (pos < M-1) {
			op = Tensor({e,r,l,l,e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (r << e < l*l) is important for cache locality!
			rhs = Tensor({l,e,r}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (l,r << e) is important for cache locality!

			#pragma omp parallel for default(none) firstprivate(_slice, pos) reduction(+:op,rhs) schedule(static)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				add_rank_one<5>(op, {measures[pos][i], rightStack[pos+1][i], leftLHSStack[pos-1][i], measures[pos][i], rightStack[pos+1][i]});
				add_rank_one<3>(rhs, {leftRHSStack[pos-1][i], measures[pos][i], rightStack[pos+1][i]});
			}
			reshuffle_modes(op, op, {2,0,1,3,4,5});
		} else {
			op = Tensor({l,l,e,e}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (e < l*l) is important for cache locality!
			rhs = Tensor({e,l}, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
			// The ordering (l << e) is important for cache locality!

			#pragma omp parallel for default(none) firstprivate(_slice, pos) reduction(+:op,rhs) schedule(static)
			for (size_t i=_slice.first; i<_slice.second; ++i) {
				add_rank_one<3>(op, {leftLHSStack[pos-1][i], measures[pos][i], measures[pos][i]});
				add_rank_one<2>(rhs, {measures[pos][i], leftRHSStack[pos-1][i]});
			}
			reshuffle_modes(op, op, {0,2,1,3});
			op.reinterpret_dimensions({l,e,1,l,e,1});
			reshuffle_modes(rhs, rhs, {1,0});
			rhs.reinterpret_dimensions({l,e,1});
		}
		REQUIRE(op.dimensions == Tensor::DimensionTuple({l,e,r, l,e,r}), "IE");  // In a macro you need parantheses around an initializer list.
		REQUIRE(rhs.dimensions == Tensor::DimensionTuple({l,e,r}), "IE");

		/* LOG(profiling, "Leaving ls_operator()"); */
		/* const double elapsed = timer.elapsed(); */
		/* std::cout << string_format("Time for ls_operator_and_rhs(position=%d): %.2fms\n", pos, elapsed); */
		op /= misc::sqr(valueNorm_trainingSet);   //TODO: is this a good idea?
		rhs /= misc::sqr(valueNorm_trainingSet);  //TODO: is this a good idea?
		return std::make_pair(op, rhs);
	}

	/* double SALSA::slow_residual(const std::pair<size_t, size_t>& _slice) const { */
	/*     // TODO: Merge with residual? */
	/*     // ||Ax - b||^2 = xtAtAx - 2*xtAtb + btb */
	/*     LOG(debug, "Entering slow_residual((" << _slice.first << ", " << _slice.second << "))"); */

	/*     const Tensor& core = x.get_component(x.corePosition); */
	/*     #if __cplusplus >= 201402L */
	/*         const auto[A, b] = ls_operator_and_rhs(_slice); */
	/*     #else */
	/*         Tensor A,b; */
	/*         std::tie(A,b) = ls_operator_and_rhs(_slice); */
	/*     #endif */

	/*     const double xtAtAx = contract(contract(core, A, 3), core, 3)[0]; */
	/*     const double xtAtb  = contract(core, b, 3)[0]; */
	/*     const double btb    = misc::sqr(valueNorm_trainingSet); */

	/*     LOG(debug, "Leaving slow_residual()"); */
	/*     return std::sqrt(std::max(xtAtAx - 2*xtAtb + btb, 0.0)) / valueNorm_trainingSet; */
	/* } */


	void SALSA::solve_local() {
		const size_t pos = x.corePosition;  //TODO: rename: position
		/* LOG(profiling, "Entering solve_local(position=" << pos << ")"); */
		const size_t l = x.get_component(pos).dimensions[0],
					 e = x.get_component(pos).dimensions[1],
					 r = x.get_component(pos).dimensions[2];

		#if __cplusplus >= 201402L
			const auto[op, rhs] = ls_operator_and_rhs(trainingSet);
		#else
			Tensor op,rhs;
			std::tie(op,rhs) = ls_operator_and_rhs(trainingSet);
		#endif
		Tensor op_alpha = alpha_operator();
		const Tensor op_omega = omega_operator();

		Tensor& core = x.component(pos);
		solve(core, op+op_alpha+op_omega, rhs);

		// iterative reweighting
		size_t step;
		op_alpha.reinterpret_dimensions({l*e*r,l*e*r});
		Tensor op_IRalpha(op_alpha.dimensions, Tensor::Representation::Dense, Tensor::Initialisation::Zero);
		for (step=0; step<maxIRsteps; ++step) {
			const Tensor prev_core = core;
			core.reinterpret_dimensions({l*e*r});
			core.modify_entries([this](value_t& _entry, const size_t _position) {
				_entry = 1.0/std::sqrt(std::max(std::abs(_entry), sparsityThreshold));
			});

			op_IRalpha = op_alpha;
			modify_rank_one<2>(op_IRalpha, { core, core }, [](value_t& _opEntry, const value_t _productEntry) {
				_opEntry *= _productEntry;
			});
			core.reinterpret_dimensions({l,e,r});
			op_IRalpha.reinterpret_dimensions({l,e,r,l,e,r});
			solve(core, op+op_IRalpha+op_omega, rhs);
			if (max_norm(prev_core - core) < IRtolerance*frob_norm(prev_core)) break;
		}
		maxIRstepsReached[pos] = (step == maxIRsteps);

		size_t density = 0; // np.count_nonzero(abs(sol) > sparsityThreshold)/sol.size
		#pragma omp parallel for default(none) firstprivate(core, sparsityThreshold) reduction(+:density) schedule(static)
		for (size_t j=0; j<core.size; ++j) {
			density += std::abs(core[j]) > sparsityThreshold;
		}
		weightedNorms[pos] = double(density)/double(core.size);
		REQUIRE(0 <= weightedNorms[pos] && weightedNorms[pos] <= 1, "IE");
		/* LOG(profiling, "Leaving solve_local()"); */
	}

	void SALSA::print_parameters() const {
		LOG(debug, "Entering print_parameters()");
		const std::string sep = std::string(125, '-')+"\n";
		std::cout << std::string(125, '=') << '\n'
			  << std::string(55, ' ') << "Running uqSALSA" << std::endl;
		std::cout << sep;
		#if __cplusplus >= 201402L
			const size_t max_param_len = 20;  // "trackingPeriodLength".size()
			const auto print_param = [max_param_len](std::string name, auto value) {
				const std::string pad(max_param_len-name.length(), ' ');
				std::cout << "  " << name << " = " << pad << value << "\n";
			};
			{
				char hostname[HOST_NAME_MAX];
				gethostname(hostname, HOST_NAME_MAX);
				print_param("hostname", hostname);
			}
			#if defined(_OPENMP)
				print_param("threads", omp_get_max_threads());
			#else
				print_param("threads", 1);
			#endif
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
			print_param("maxSweeps", maxSweeps);
			print_param("trackingPeriodLength", trackingPeriodLength);
			print_param("maxStagnatingEpochs", maxStagnatingEpochs);
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
			print_param("omegaFactor", omegaFactor);
			std::cout << '\n';
			print_param("falpha", falpha);
			print_param("alphaFactor", alphaFactor);
			std::cout << sep;
		#endif
		std::cout << std::flush;  // Make sure the user does not have to wait for the first sweep to complete.
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

		for (size_t m=0; m<M; ++m) { basisWeights[m].use_dense_representation(); }
		for (size_t m=1; m<M; ++m) { for (size_t i=0; i<N; ++i) { measures[m][i].use_dense_representation(); } }
		for (size_t i=0; i<N; ++i) { values[i].use_dense_representation(); }

		// build stacks and compute left and right singular value arrays
		x.move_core(x.order()-1);
		while (x.corePosition > 0) { move_core_left(false); }

		// The sets have already been shuffled in the constructor. Now they need to be split.
		size_t Nt = size_t((1.0-controlSetFraction)*double(N));  // Nv = N - Nt;
		trainingSet = std::make_pair(0, Nt);
		validationSet = std::make_pair(Nt, N);

		valueNorm_trainingSet = 0.0;
		#pragma omp parallel for default(none) firstprivate(trainingSet) reduction(+:valueNorm_trainingSet) schedule(static)
		for(size_t j=trainingSet.first; j<trainingSet.second; ++j) {
			valueNorm_trainingSet += misc::sqr(frob_norm(values[j]));
		}
		valueNorm_trainingSet = std::sqrt(valueNorm_trainingSet);

		valueNorm_validationSet = 0.0;
		#pragma omp parallel for default(none) firstprivate(validationSet) reduction(+:valueNorm_validationSet) schedule(static)
		for(size_t j=validationSet.first; j<validationSet.second; ++j) {
			valueNorm_validationSet += misc::sqr(frob_norm(values[j]));
		}
		valueNorm_validationSet = std::sqrt(valueNorm_validationSet);

		// compute SALSA parameters
		const auto res = residual(std::make_pair(0, N));
		const auto maxResSqrtRes = std::max(res, std::sqrt(res));
		// alpha and omega should be of the same order as the residual.
		// This ensures that the regularization terms initially dominate the cost and can promote generalization.
		// Since alpha enters the cost functional squared we want to choose alpha = omega = std::sqrt(res).
		//TODO: But when res is larger than 1 this is smaller than res and therefore we take the std::max. (Why is this important?)
		// alpha = alphaFactor * std::min(valueNorm_trainingSet, maxResSqrtRes);
		alpha = alphaFactor * maxResSqrtRes;
		omega = maxResSqrtRes;
		smin = 0.2*std::min(omega, res);
		omega *= omegaFactor;

		initialized = true;

		LOG(debug, "Leaving initialize()");
	}

	std::string SALSA::print_fractional_ranks() const {
		LOG(debug, "Entering print_fractional_ranks()");
		/* assert len(self.singularValues) == self.x.order()-1 */
		/* for i in range(self.x.order()-1): */
		/*     assert len(self.singularValues[i]) == self.x.rank(i) */
		std::string output = print_list<std::vector<double>>(singularValues, [&](const std::vector<double>& _sVs) {
			/* assert np.all(_sVs > 0) */
			size_t rank;
			for (rank=0; rank<_sVs.size(); ++rank) {
				if (_sVs[rank] <= smin) break;
			}
			double inactive = 0.0f;
			if (rank < _sVs.size()) {
				inactive = _sVs[rank]/smin;
				REQUIRE(0.0f <= inactive && inactive < 1.0f, "IE: " << inactive);
			}
			return std::to_string(rank) + "." + Foreground::Gray(inactive) + std::to_string(unsigned(10.0f*inactive)) + Foreground::Reset + u8"\u002F" + Foreground::Separate + std::to_string(_sVs.size()) + Foreground::Reset;
		});
		LOG(debug, "Leaving print_fractional_ranks()");
		return output;
	}

	std::string SALSA::print_densities() const {
		LOG(debug, "Entering print_densities()");
		const std::string output = print_list(M, [&](const size_t _index) {
			if (maxIRstepsReached[_index]) {
				return string_format(Foreground::Alert + "%2u" + Foreground::Reset, std::min(size_t(100*weightedNorms[_index]+0.5), size_t{99}));
			}
			return string_format("%2u", std::min(size_t(100*weightedNorms[_index]+0.5), size_t{99}));
		});
		LOG(debug, "Leaving print_densities()");
		return output + "%";
	}

	void SALSA::run() {
		LOG(debug, "Entering run()");
		REQUIRE(omegaFactor > 0.0, "omegaFactor must be positive");
		REQUIRE(alphaFactor >= 0.0, "alphaFactor must be non-negative");

		print_parameters();
		if (!misc::hard_equal(alphaFactor, 0.0)) {
			const auto attr = [](const unsigned _code) -> std::string { return u8"\033["+std::to_string(_code)+"m"; };
			const auto alert = Foreground::Alert + attr(1);
			const auto reset = attr(0);
			std::cout << alert + "WARNING: Optimizing with l1 regularization is an experimental feature" + reset + "\n";
		}
		initialize();

		const auto perform_sweep = [&](const bool _adapt) {
			REQUIRE(x.corePosition == 0, "IE");
			LOG(debug, "Sweep: left --> right");
			for (size_t m=0; m<M-1; ++m) {
				REQUIRE(x.corePosition == m, "IE");
				solve_local();
				move_core_right(_adapt);
			}
			LOG(debug, "Sweep: right --> left");
			for (size_t m=M-1; m>0; --m) {
				REQUIRE(x.corePosition == m, "IE");
				solve_local();
				move_core_left(_adapt);
			}
			REQUIRE(x.corePosition == 0, "IE");
		};

		#if !defined(PROFILING)
			perform_sweep(false);
		#endif

		boost::circular_buffer<double> trainingResiduals{trackingPeriodLength};
		trainingResiduals.push_back(residual(trainingSet));
		auto bestTrainingResidual = trainingResiduals.back();
		std::vector<double> validationResiduals;
		validationResiduals.push_back(residual(validationSet));

		initialResidual = trainingResiduals.back();  //TODO: rename
		bestIteration = 0;
		double prev_bestValidationResidual = validationResiduals.back();
		bestState = State{alpha, omega, x, trainingResiduals.back(), validationResiduals.back()};

		size_t iteration = 0;
		size_t nonImprovementCounter = 0;
		bool omegaMinimal = false;

		auto alpha_residual = [&](){
			const Tensor op_alpha = alpha_operator();
			const Tensor& core = x.get_component(x.corePosition);
			const Tensor IR = diag(core, [this](double _entry) { return 1.0/std::sqrt(std::max(std::abs(_entry), sparsityThreshold)); });
			Tensor ret;
			contract(ret, core, IR, 3);
			contract(ret, ret, op_alpha, 3);
			contract(ret, ret, IR, 3);
			contract(ret, ret, core, 3);
			REQUIRE(ret.dimensions == Tensor::DimensionTuple({}), "IE");
			return misc::sqr(alpha)*ret[0];
		};
		auto omega_residual = [&](){
			const Tensor op_omega = omega_operator();
			const Tensor& core = x.get_component(x.corePosition);
			Tensor ret;
			contract(ret, core, op_omega, 3);
			contract(ret, ret, core, 3);
			REQUIRE(ret.dimensions == Tensor::DimensionTuple({}), "IE");
			return misc::sqr(omega)*ret[0];
		};
		auto alphaCosts = alpha_residual();
		auto bestAlphaCosts = alphaCosts;
		auto omegaCosts = omega_residual();
		auto bestOmegaCosts = omegaCosts;
		auto totalCosts = trainingResiduals.back() + alphaCosts + omegaCosts;
		auto bestTotalCosts = totalCosts;

		auto print_update = [&](const bool improvement){
			const auto update_str = [](const double prev, const double cur)  {
				std::ostringstream ret;
				if (cur <= prev+1e-8) { ret << Foreground::Good; }
				else { ret << Foreground::Bad; }
				ret << string_format("%.2e", cur) << Foreground::Reset;
				return ret.str();
			};
			const auto attr = [](const unsigned _code) -> std::string { return u8"\033["+std::to_string(_code)+"m"; };
			const auto reset = attr(0);
			const auto bold = attr(1);
			if (improvement) { std::cout << bold; }
			std::cout << "[" << iteration << "] Costs:"
								  << " LS="              << update_str(bestTrainingResidual , trainingResiduals.back())
								  << u8", R\u03B1="      << update_str(bestAlphaCosts, alphaCosts)
								  << u8", R\u03C9="      << update_str(bestOmegaCosts, omegaCosts)
								  << "  |  Validation: " << update_str(bestState.validationResidual , validationResiduals.back())
								  << "  |  \u03C9: "     << string_format("%.2e", omega)
								  << "  |  Densities: "  << print_densities()
								  << "  |  Ranks: "      << print_fractional_ranks() << reset << std::endl;  // Flush to ensure that the user does not have to wait for other sweeps to complete
		};
		print_update(true);

		for (iteration=1; iteration<maxSweeps; ++iteration) {
			perform_sweep(true);

			trainingResiduals.push_back(residual(trainingSet));
			alphaCosts = alpha_residual();
			omegaCosts = omega_residual();
			totalCosts = trainingResiduals.back() + alphaCosts + omegaCosts;
			validationResiduals.push_back(residual(validationSet));

			// check if the validation residual decreased in a meaningful way during the last sweep
			const bool sweepImprovement = validationResiduals.back() < (1-minDecrease)*bestState.validationResidual;
			print_update(sweepImprovement);

			bestTrainingResidual = std::min(trainingResiduals.back(), bestTrainingResidual);
			bestAlphaCosts = std::min(alphaCosts, bestAlphaCosts);
			bestOmegaCosts = std::min(omegaCosts, bestOmegaCosts);
			bestTotalCosts = std::min(totalCosts, bestTotalCosts);
			if (sweepImprovement) {
				bestIteration = iteration;
				prev_bestValidationResidual = bestState.validationResidual;
				bestState = State{alpha, omega, x, trainingResiduals.back(), validationResiduals.back()};
			}

			if (validationResiduals.back() < targetResidual) {
				std::cout << "Terminating: Minimum residual reached." << std::endl;
				break;
			}

			auto res = trainingResiduals.back();
			omega /= omegaFactor;
			// omega = std::max(std::min(omega/fomega, std::sqrt(res)), res);
			omega = std::min(omega/fomega, std::max(res, std::sqrt(res)));
			omegaMinimal = omega <= res+1e-6; // misc::approx_equal(omega, res, 1e-6);
			smin = 0.2*std::min(omega, res);
			omega *= omegaFactor;

			// check for stagnation in the training residual (trainingResiduals is a circular buffer)
			const bool stagnation = trainingResiduals.size() == trackingPeriodLength && trainingResiduals.back() > (1-minDecrease)*trainingResiduals.front() && omegaMinimal;
			if (stagnation) {
				// check if the best validation residual decreased while minimizing with this alpha (validationResiduals contains all validation residuals since this alpha was chosen)
				const bool epochImprovement = bestIteration+validationResiduals.size() > iteration;
				if (epochImprovement) {
					nonImprovementCounter = 0;
					std::cout << string_format(u8"Epochs with stagnating validation residual: %d (val=%.2e \u2198 val=%.2e)", nonImprovementCounter, prev_bestValidationResidual, bestState.validationResidual) << std::endl;
				} else {
					nonImprovementCounter += 1;
					std::cout << string_format(u8"Epochs with stagnating validation residual: %d (val=%.2e \u2192)", nonImprovementCounter, bestState.validationResidual) << std::endl;
				}
				// check termination criterion
				if (nonImprovementCounter >= maxStagnatingEpochs) {
					std::cout << "Terminating: Minimum residual decrease deceeded " << nonImprovementCounter << " iterations in a row." << std::endl;
					break;
				}

				// adapt alpha
				if (alphaFactor > 0.0) {
					const auto prev_alpha = alpha;
					// Select omega and x from the most recent optimal state.
					//TODO: come up with a better name than alpha-cycle
					// Note that alpha may be lower than bestState.alpha if one (alpha-)cycle of optimization did not result in a sufficient reduction of the validation resiudal.
					omega = bestState.omega;
					smin = 0.2*std::min(omega/omegaFactor, res);
					// x = TTTensor(bestState.x);  //TODO: Why is this explicit copy necessary?
					x = bestState.x;
					REQUIRE(x.corePosition == 0, "IE");
					// reinitialize right stack
					initialized = false;
					x.move_core(x.order()-1);
					while (x.corePosition > 0) { move_core_left(false); }
					initialized = true;
					res = std::sqrt(misc::sqr(bestState.trainingResidual) + misc::sqr(bestState.validationResidual));
					// res = validationResiduals.back() * valueNorm_validationSet;
					alpha /= alphaFactor;
					alpha = std::min(alpha/falpha, std::max(res, std::sqrt(res)));
					alpha *= alphaFactor;

					//TODO: use disp_shortest_unequal(self.alpha, alpha)
					std::cout << "Reduce \u03B1: " << string_format("%.3f", prev_alpha)
							  << std::string(u8" \u2192 ")
							  << string_format("%.3f", alpha) << std::endl;
				} else { REQUIRE(misc::hard_equal(alpha, 0.0), "IE"); }

				// clear buffers to ensure that they contain only values for the current choice of alpha
				trainingResiduals.clear();
				validationResiduals.clear();
			}
		}

		if (iteration == maxSweeps) {
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
				const auto attr = [](const unsigned _code) -> std::string { return u8"\033["+std::to_string(_code)+"m"; };
				const auto alert = Foreground::Alert + attr(1);
				const auto reset = attr(0);
				std::cout << alert + "WARNING: maxRanks[" << m << "] = " << maxRanks[m] << " was reached during optimization." + reset + "\n";
			}
		}

		std::cout << string_format("Residual decreased from %.2e to %.2e in %u iterations.\n", initialResidual, bestTrainingResidual, iteration)
					<< std::string(125, '=') << std::endl;
		LOG(debug, "Leaving run()");
	}

}} // namespace uq | xerus
