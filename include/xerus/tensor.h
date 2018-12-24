// // Xerus - A General Purpose Tensor Library
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
 * @brief Header file for the Tensor class.
 */

#pragma once

#include <map>
#include <limits>
#include <memory>
#include <random>
#include <functional>

#include "basic.h"
#include "misc/containerSupport.h"
#include "misc/fileIO.h"
#include "misc/random.h"

#include "indexedTensor.h"

namespace xerus {
	//Forwad declarations
	class TensorNetwork;
	
	// NOTE these two functions are used in the template functions of Tensor so they have to be declared before the class..
	class Tensor;
	
    /** 
     * @brief Low-level contraction between Tensors.
     * @param _result Output for the result of the contraction.
     * @param _lhs left hand side of the contraction.
     * @param _lhsTrans Flags whether the LHS should be transposed (in the matrifications sense).
     * @param _rhs right hand side of the contraction.
     * @param _rhsTrans Flags whether the RHS should be transposed (in the matrifications sense).
     * @param _numModes number of modes that shall be contracted.
     */
    void contract(Tensor& _result, const Tensor& _lhs, const bool _lhsTrans, const Tensor& _rhs, const bool _rhsTrans, const size_t _numModes);
    Tensor contract(const Tensor& _lhs, const bool _lhsTrans, const Tensor& _rhs, const bool _rhsTrans, const size_t _numModes);
    
    
	
	/**
	 * @brief: Performs a simple reshuffle. Much less powerfull then a full evaluate, but more efficient.
	 * @details @a _shuffle shall be a vector that gives for every old index, its new position.
	 */
	void reshuffle(Tensor& _out, const Tensor& _base, const std::vector<size_t>& _shuffle);
	Tensor reshuffle(const Tensor& _base, const std::vector<size_t>& _shuffle);
	
	
	/// @brief Class that handles simple (non-decomposed) tensors in a dense or sparse representation.
	class Tensor final {
	public:
		static size_t sparsityFactor; // NOTE not const so that users can modify this value!
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Auxiliary types- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Flags determining the initialisation of the data of Tensor objects. 
		 * @details None means that no initialisation is performed, i.e. the data can be random.
		 * Zero means that the data is zero initialized.
		 */
		enum class Initialisation : bool { None, Zero };
		
		/** 
		 * @brief Flags indicating the internal representation of the data of Tensor objects. 
		 * @details Dense means that an value_t array of 'size' is used to store each entry individually, 
		 * using row-major order. Sparse means that only the non-zero entries are stored explicitly in a set containing
		 * their value and position.
		 */
		enum class Representation : bool { Dense, Sparse };
		
		///@brief: Represention of the dimensions of a Tensor.
		typedef std::vector<size_t> DimensionTuple; // NOTE must not be declared as "using.." (internal segfault in gcc4.8.1)
		
		///@brief: Represention of a MultiIndex, i.e. the tuple of positions for each dimension determining a single position in a Tensor.
		typedef std::vector<size_t> MultiIndex; // NOTE must not be declared as "using.." (internal segfault in gcc4.8.1)
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Member variables - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		/// @brief Vector containing the individual dimensions of the tensor.
		DimensionTuple dimensions;
		
		/// @brief Size of the Tensor -- always equal to the product of the dimensions.
		size_t size = 1;
		
		/// @brief Single value representing a constant scaling factor.
		value_t factor = 1.0;
		
		/// @brief The current representation of the Tensor (i.e Dense or Sparse)
		Representation representation = Representation::Sparse;
		
	private:
		/** 
		 * @brief Shared pointer to the dense data array, if representation is dense. 
		 * @details The data is stored such that indices increase from right to left (row-major order). 
		 * If the tensor is modified and not sole owner a deep copy is performed.
		 */
		std::shared_ptr<value_t> denseData;
		
		/** 
		 * @brief Shared pointer to the a map containing the non-zero entries, if representation is Sparse. 
		 * @details The entries are stored in a map which uses the position of each entry assuming row-major ordering as key value.
		 * If the tensor is modified and not sole owner a deep copy is performed.
		 */
		std::shared_ptr<std::map<size_t, value_t>> sparseData;
		
	public:
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/// @brief Constructs an order zero Tensor with the given inital representation
		explicit Tensor(const Representation _representation = Representation::Sparse);
		
		
		/// @brief Tensors are default copy constructable.
		Tensor( const Tensor&  _other ) = default;
		
		
		/// @brief Tensors are default move constructable.
		Tensor(       Tensor&& _other ) noexcept = default;
		
		/** 
		 * @brief: Creates a new tensor with the given dimensions.
		 * @param _dimensions the dimensions of the new tensor.
		 * @param _representation (optional) the initial representation of the tensor.
		 * @param _init (optional) inital data treatment, i.e. whether the tensor is to be zero Initialized.
		 */
		explicit Tensor(DimensionTuple _dimensions, const Representation _representation = Representation::Sparse, const Initialisation _init = Initialisation::Zero);
		
		
		/** 
		 * @brief: Creates a new (dense) tensor with the given dimensions, using a provided data.
		 * @param _dimensions the dimensions of the new tensor.
		 * @param _data inital dense data in row-major order.
		 */
		template<XERUS_ADD_MOVE(Vec, DimensionTuple), XERUS_ADD_MOVE(SPtr, std::shared_ptr<value_t>)>
		explicit Tensor(Vec&& _dimensions, SPtr&& _data)
		: dimensions(std::forward<Vec>(_dimensions)), size(misc::product(dimensions)), representation(Representation::Dense), denseData(std::forward<SPtr>(_data)) { }
		
		
		/** 
		 * @brief: Creates a new (dense) tensor with the given dimensions, using a provided data.
		 * @param _dimensions the dimensions of the new tensor.
		 * @param _data inital dense data in row-major order.
		 */
		explicit Tensor(DimensionTuple _dimensions, std::unique_ptr<value_t[]>&& _data);
		
		
		/** 
		 * @brief Constructs a Tensor with the given dimensions and uses the given function to assign the values to the entries.
		 * @details In this overload no value is passed to _f, i.e. _f must determine the values of the entries independend of their position,
		 * or keep track of the position itself. _f may assume that it is called for the entries in the order they are stored (i.e. row-major order)
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _f the function to use to set the entries of the Tensor. 
		 */
		explicit Tensor(DimensionTuple _dimensions, const std::function<value_t()>& _f);
		
		
		/** 
		 * @brief Constructs a Tensor with the given dimensions and uses the given function to assign the values to the entries.
		 * @details In this overload the position of each entry assuming row-major order is passed to  _f.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _f the function to use to set the entries of the Tensor. 
		 */
		explicit Tensor(DimensionTuple _dimensions, const std::function<value_t(const size_t)>& _f);
		
		
		/** 
		 * @brief Constructs a Tensor with the given dimensions and uses the given function to assign the values to the entries.
		 * @details In this overload the complete position of each entry is passed to  _f.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _f the function to use to set the entries of the Tensor. 
		 */
		explicit Tensor(DimensionTuple _dimensions, const std::function<value_t(const MultiIndex&)>& _f);
		
		
		/** 
		 * @brief Constructs a Tensor with the given dimensions and uses the given function @a _f to create @a _N non zero entries.
		 * @details @a _f is called with the current number of entries present and the number of possible entries (i.e. size). @a _f shall return a pair containg the position
		 * and value of the next entry. @a _f is required not to return a position twice.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _N the number of non-zero entries to be created.
		 * @param _f the function to be used to create each non zero entry. 
		 */
		Tensor(DimensionTuple _dimensions, const size_t _N, const std::function<std::pair<size_t, value_t>(size_t, size_t)>& _f);
		
		
		/** 
		 * @brief Constructs a dense Tensor with the given dimensions and uses the given random generator and distribution to assign the values to the entries.
		 * @details The entries are assigned in the order they are stored (i.e. row-major order). Each assigned is a seperate call to the random distribution.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _rnd the random generator to be used.
		 * @param _dist the random distribution to be used.
		 */
		template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64>
		static Tensor XERUS_warn_unused random(DimensionTuple _dimensions, distribution& _dist=xerus::misc::defaultNormalDistribution, generator& _rnd=xerus::misc::randomEngine) {
			Tensor result(std::move(_dimensions), Representation::Dense, Initialisation::None);
			value_t* const dataPtr = result.denseData.get();
			for(size_t i = 0; i < result.size; ++i) {
				dataPtr[i] = _dist(_rnd);
			}
			return result;
		}
		
		
		/** 
		 * @brief Constructs a dense Tensor with the given dimensions and uses the given random generator and distribution to assign the values to the entries.
		 * @details See the std::vector variant for details.
		 */
		template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64>
		XERUS_force_inline static Tensor XERUS_warn_unused random(std::initializer_list<size_t>&& _dimensions, distribution& _dist=xerus::misc::defaultNormalDistribution, generator& _rnd=xerus::misc::randomEngine) {
			return Tensor::random(DimensionTuple(std::move(_dimensions)), _dist, _rnd);
		}
		
		
		/** 
		 * @brief Constructs a dense Tensor with the given dimensions and uses the given random generator and distribution to assign the values to the entries.
		 * @details The entries are assigned in the order they are stored (i.e. row-major order). Each assigned is a seperate call to the random distribution.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _rnd the random generator to be used.
		 * @param _dist the random distribution to be used.
		 */
		template<class generator=std::mt19937_64>
		static Tensor XERUS_warn_unused random_orthogonal(DimensionTuple _dimensions1, DimensionTuple _dimensions2, generator& _rnd=xerus::misc::randomEngine) {
			std::vector<size_t> dimensions = _dimensions1;
			dimensions.insert(dimensions.end(), _dimensions2.begin(), _dimensions2.end());
			const size_t m = misc::product(_dimensions1);
			const size_t n = misc::product(_dimensions2);
			const size_t max = std::max(m,n);
			const size_t min = std::min(m,n);
			Tensor result({max,min}, Representation::Sparse, Initialisation::Zero);
			
			typename generator::result_type randomness = 0;
			const size_t restart = size_t(std::log2(generator::max()));
			for (size_t i=0; i<min; ++i) {
				auto idx = i%restart;
				if (idx == 0) {
					randomness = _rnd();
				}
				if (randomness & (1<<idx)) {
					result[{i,i}] = 1;
				} else {
					result[{i,i}] = -1;
				}
			}
			
			for (size_t i=0; i<min-1; ++i) { // do k = n-1 to 1 by -1; 
				Tensor u = Tensor::random({max-i}, misc::defaultNormalDistribution, _rnd);
				u[0] -= u.frob_norm();
				u /= u.frob_norm();
				u.apply_factor();
				contract(u, u, false, u, false, 0);
				u *= -2.0;
				Tensor p = Tensor::identity({max,max});
				p.offset_add(u, {i,i});
				contract(result, p, false, result, false, 1);
			}
			
			if (m != max) {
				reshuffle(result, result, {1,0});
			}
			result.reinterpret_dimensions(std::move(dimensions));
			
			return result;
		}
		
		
		/** 
		 * @brief Constructs a random sparse Tensor with the given dimensions.
		 * @details The given random generator @a _rnd and distribution @a _dist are used to assign the values to @a _n randomly choosen entries.
		 * @param _dimensions the future dimensions of the Tensor.
		 * @param _N the number of non-zero entries to be created.
		 * @param _rnd the random generator to be used.
		 * @param _dist the random distribution to be used.
		 */
		template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64>
		static Tensor XERUS_warn_unused random(DimensionTuple _dimensions, const size_t _N, distribution& _dist=xerus::misc::defaultNormalDistribution, generator& _rnd=xerus::misc::randomEngine) {
			Tensor result(std::move(_dimensions), Representation::Sparse, Initialisation::Zero);
			XERUS_REQUIRE(_N <= result.size, " Cannot create " << _N << " non zero entries in a tensor with only " << result.size << " total entries!");
			
			std::uniform_int_distribution<size_t> entryDist(0, result.size-1);
			while(result.sparseData->size() < _N) {
				result.sparseData->emplace(entryDist(_rnd), _dist(_rnd));
			}
			return result;
		}
		
		
		/** 
		 * @brief Constructs a random sparse Tensor with the given dimensions.
		 * @details See the std::vector variant for details.
		 */
		template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64>
		XERUS_force_inline static Tensor XERUS_warn_unused random(std::initializer_list<size_t>&& _dimensions, const size_t _N, distribution& _dist, generator& _rnd) {
			return Tensor::random(DimensionTuple(_dimensions), _N, _rnd, _dist);
		}
		
		
		/** 
		 * @brief: Returns a Tensor with all entries equal to one.
		 * @param _dimensions the dimensions of the new tensor.
		 */
		static Tensor XERUS_warn_unused ones(DimensionTuple _dimensions);
		
		
		/** 
		 * @brief: Returns a Tensor representation of the identity operator with the given dimensions.
		 * @details That is combining the first half of the dimensions and the second half of the dimensions results in an identity matrix.
		 * @param _dimensions the dimensions of the new tensor. It is required that _dimensions[i] = _dimensions[d/2+i], otherwise this cannot be the identity operator.
		 */
		static Tensor XERUS_warn_unused identity(DimensionTuple _dimensions);
		
		
		/** 
		 * @brief: Returns a Tensor representation of the kronecker delta.
		 * @details That is each entry is one if all indices are equal and zero otherwise. Note iff d=2 this coincides with identity.
		 * @param _dimensions the dimensions of the new tensor.
		 */
		static Tensor XERUS_warn_unused kronecker(DimensionTuple _dimensions);
		
		
		/** 
		 * @brief: Returns a Tensor with a single entry equals one and all other zero.
		 * @param _dimensions the dimensions of the new tensor.
		 * @param _position The position of the one
		 */
		static Tensor XERUS_warn_unused dirac(DimensionTuple _dimensions, const MultiIndex& _position);
		
		
		/** 
		 * @brief: Returns a Tensor with a single entry equals one and all other zero.
		 * @param _dimensions the dimensions of the new tensor.
		 * @param _position The position of the one
		 */
		static Tensor XERUS_warn_unused dirac(DimensionTuple _dimensions, const size_t _position);
		

		/// @brief Returns a copy of this Tensor that uses a dense representation.
		Tensor dense_copy() const;
		
		
		/// @brief Returns a copy of this Tensor that uses a sparse representation.
		Tensor sparse_copy() const;
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Standard operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		/** 
		 * @brief Standard assignment operator.
		 * @param _other the Tensor to be assinged to this one.
		 * @return a reference to this Tensor.
		 */
		Tensor& operator=(const Tensor&  _other) = default;
		
		
		/** 
		 * @brief Standard move-assignment operator.
		 * @param _other the Tensor to be move-assinged to this one.
		 * @return a reference to this Tensor.
		 */
		Tensor& operator=(      Tensor&& _other) = default;
		
		
		/** 
		 * @brief Assigns the given TensorNetwork to this Tensor by completely contracting the network.
		 * @param _other the TensorNetwork to be to this Tensor.
		 * @return a reference to this Tensor.
		 */
		Tensor& operator=(const TensorNetwork& _network);
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Information - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Returns the degree of the tensor.
		 * @details The degree is always equals to dimensions.size()
		 * @return the degree of the tensor
		 */
		size_t degree() const;
		
		/** 
		 * @brief Checks whether the tensor has a non-trivial global scaling factor.
		 * @return true if there is a non-trivial factor, false if not.
		 */
		bool has_factor() const;
		
		/// @brief Returns whether the current representation is dense.
		bool is_dense() const;
		
		/// @brief Returns whether the current representation is sparse.
		bool is_sparse() const;
		
		/** 
		 * @brief Returns the number currently saved entries. 
		 * @details Note that this is not nessecarily the number of non-zero entries as the saved entries may contain
		 * zeros. Even more if a dense representation is used size is returned. 
		 */
		size_t sparsity() const;
		
		/** 
		 * @brief Determines the number of non-zero entries.
		 * @param _eps (optional) epsilon detrmining the maximal value, that is still assumed to be zero.
		 * @return the number of non-zero entries found.
		 */
		size_t count_non_zero_entries(const value_t _eps = std::numeric_limits<value_t>::epsilon()) const;
		
		/** 
		 * @brief Checks the tensor for illegal entries, e.g. nan, inf,...
		 * @return TRUE there are no invalid entries, FALSE otherwise.
		 */
		bool all_entries_valid() const;
		
		/** 
		 * @brief Approximates the cost to reorder the tensor.
		 * @return the approximated costs.
		 */
		size_t reorder_cost() const;
		
		/** 
		 * @brief Calculates the frobenious norm of the tensor.
		 * @return the frobenious norm.
		 */
		value_t frob_norm() const;
		
		/** 
		 * @brief Calculates the 1-norm of the tensor.
		 * @return the 1-norm.
		 */
		value_t one_norm() const;
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		/** 
		 * @brief Adds the @a _other Tensor entrywise to this one.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide.
		 * @param _other the Tensor to be added to this one.
		 * @return a reference to this Tensor.
		 */
		Tensor& operator+=(const Tensor& _other);
		
		/** 
		 * @brief Subtracts the @a _other Tensor entrywise from this one.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide.
		 * @param _other the Tensor to be subtracted to this one.
		 * @return a reference to this Tensor.
		 */
		Tensor& operator-=(const Tensor& _other);
		
		/** 
		 * @brief Performs the entrywise multiplication with a constant @a _factor.
		 * @details Internally this only results in a change in the global factor.
		 * @param _factor the factor,
		 * @return a reference to this Tensor.
		 */
		Tensor& operator*=(const value_t _factor);
		
		/** 
		 * @brief Performs the entrywise divison by a constant @a _divisor.
		 * @details Internally this only results in a change in the global factor.
		 * @param _divisor the divisor,
		 * @return a reference to this Tensor.
		 */ 
		Tensor& operator/=(const value_t _divisor);
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Access - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Read/Write access a single entry.
		 * @param _position the position of the desired entry, assuming row-major ordering.
		 * @return a reference to the selected entry.
		 */
		value_t& operator[](const size_t _position);
		
		/** 
		 * @brief Read access a single entry.
		 * @param _position the position of the desired entry, assuming row-major ordering.
		 * @return the value of the selected entry.
		 */
		value_t operator[](const size_t _position) const;
		
		/** 
		 * @brief Read/Write access a single entry.
		 * @param _positions the positions of the desired entry.
		 * @return a reference to the selected entry.
		 */
		value_t& operator[](const MultiIndex& _positions);
		
		/** 
		 * @brief Read access a single entry.
		 * @param _positions the positions of the desired entry.
		 * @return the value of the selected entry.
		 */
		value_t operator[](const MultiIndex& _positions) const;
		
		
		/** 
		 * @brief Unsanitized access to a single entry.
		 * @param _position the position of the desired entry, assuming row-major ordering.
		 * @return the value of the selected entry.
		 */
		value_t& at(const size_t _position);
		
		/** 
		 * @brief Unsanitized read access to a single entry.
		 * @param _position the position of the desired entry, assuming row-major ordering.
		 * @return the value of the selected entry.
		 */
		value_t cat(const size_t _position) const;
		
		/** 
		 * @brief Returns a pointer for direct access to the dense data array in row major order. 
		 * @details Also takes care that this direct access is safe, i.e. that this tensor is using a dense representation, is the sole owner of the data and that no non trivial factor exists.
		 * @return pointer to the dense data array.
		 */
		value_t* get_dense_data();
		
		/** 
		 * @brief Gives access to the internal data pointer, without any checks.
		 * @details Note that the dense data array might not exist because a sparse representation is used, may shared with other tensors 
		 * or has to be interpreted considering a gloal factor. Both can be avoid if using get_dense_data(). The tensor data itself is stored in row-major ordering.
		 * @return pointer to the internal dense data array.
		 */
		value_t* get_unsanitized_dense_data();
		
		/** 
		 * @brief Gives access to the internal data pointer, without any checks.
		 * @details Note that the dense data array might not exist because a sparse representation is used, may shared with other tensors 
		 * or has to be interpreted considering a gloal factor. Both can be avoid if using get_dense_data(). The tensor data itself is stored in row-major ordering.
		 * @return pointer to the internal dense data array.
		 */
		const value_t* get_unsanitized_dense_data() const;
		
		/** 
		 * @brief Returns a pointer to the internal dense data array for complete rewrite purpose ONLY.
		 * @details This is equivalent to calling reset() with the current dimensions, dense representation and no initialisation and then
		 * calling get_unsanitized_dense_data().
		 * @return pointer to the internal dense data array.
		 */
		value_t* override_dense_data();
		
		/** 
		 * @brief Gives access to the internal shared data pointer, without any checks.
		 * @details Note that the data array might be shared with other tensors or has to be interpreted considering a global
		 * factor. Both can be avoid if using get_dense_data(). The tensor data itself is stored in row-major ordering.
		 * @return The internal shared pointer to the data array.
		 */
		const std::shared_ptr<value_t>& get_internal_dense_data();
		
		/** 
		 * @brief Returns a reference for direct access to the sparse data map. 
		 * @details Also takes care that this direct access is safe, i.e. that this tensor is using a dense representation, is the sole owner of the data and that no non trivial factor exists.
		 * @return reference to the sparse data map.
		 */
		std::map<size_t, value_t>& get_sparse_data();
		
		/** 
		 * @brief Gives access to the internal sparse map, without any checks.
		 * @details Note that the sparse data map might not exist because no sparse representation is used, 
		 * may shared with other tensors or has to be interpreted considering a gloal factor. Both can be avoid if using get_sparse_data().
		 * @return reference to the internal sparse data map.
		 */
		std::map<size_t, value_t>& get_unsanitized_sparse_data();
		
		/** 
		 * @brief Gives access to the internal sparse map, without any checks.
		 * @details Note that the sparse data map might not exist because no sparse representation is used, 
		 * may shared with other tensors or has to be interpreted considering a gloal factor. Both can be avoid if using get_sparse_data().
		 * @return reference to the internal sparse data map.
		 */
		const std::map<size_t, value_t>& get_unsanitized_sparse_data() const;
		
		/** 
		 * @brief Returns a pointer to the internal sparse data map for complete rewrite purpose ONLY.
		 * @details This is equivalent to calling reset() with the current dimensions, sparse representation and no initialisation and then
		 * calling get_unsanitized_sparse_data().
		 * @return reference to the internal sparse data map.
		 */
		std::map<size_t, value_t>& override_sparse_data();
		
		/** 
		 * @brief Gives access to the internal shared sparse data pointer, without any checks.
		 * @details Note that the sparse data map might not exist because no sparse representation is used, 
		 * may shared with other tensors or has to be interpreted considering a gloal factor. Both can be avoid if using get_sparse_data().
		 * @return The internal shared pointer to the sparse data map.
		 */
		const std::shared_ptr<std::map<size_t, value_t>>& get_internal_sparse_data();
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Indexing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Indexes the Tensor for read/write use.
		 * @param _args several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		template<typename... args>
		internal::IndexedTensor<Tensor> operator()(args... _args) {
			return internal::IndexedTensor<Tensor>(this, std::vector<Index>({_args...}), false);
		}
		
		
		/** 
		 * @brief Indexes the Tensor for read only use.
		 * @param _args several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		template<typename... args>
		internal::IndexedTensorReadOnly<Tensor> operator()(args... _args) const {
			return internal::IndexedTensorReadOnly<Tensor>(this, std::vector<Index>({_args...}));
		}
		
		
		/** 
		 * @brief Indexes the tensor for read/write use.
		 * @param _indices several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		internal::IndexedTensor<Tensor> operator()(const std::vector<Index>&  _indices);
		
		
		/** 
		 * @brief Indexes the tensor for read/write use.
		 * @param _indices several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		internal::IndexedTensor<Tensor> operator()(	  std::vector<Index>&& _indices);

		
		/** 
		 * @brief Indexes the tensor for read only use.
		 * @param _indices several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		internal::IndexedTensorReadOnly<Tensor> operator()(const std::vector<Index>&  _indices) const;
		
		
		/** 
		 * @brief Indexes the tensor for read only use.
		 * @param _indices Several [indices](@ref Index) determining the desired index order.
		 * @return an internal representation of an IndexedTensor.
		 */
		internal::IndexedTensorReadOnly<Tensor> operator()(	  std::vector<Index>&& _indices) const;
		
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Modifiers - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Resets the tensor to the given dimensions and representation.
		 * @details Leaves the Tensor in the same state as if newly constructed with the the same arguments.
		 * @param _newDim the dimensions of the new tensor.
		 * @param _representation the new representation of the tensor.
		 * @param _init (optional) data treatment, i.e. whether the tensor shall be zero initialized.
		 */
		void reset(DimensionTuple _newDim, const Representation _representation, const Initialisation _init = Initialisation::Zero);
		
		
		/** 
		 * @brief Resets the tensor to the given dimensions, preserving the current representation.
		 * @param _newDim the dimensions of the new tensor.
		 * @param _init (optional) data treatment, i.e. whether the tensor shall be zero initialized.
		 */
		void reset(DimensionTuple _newDim, const Initialisation _init = Initialisation::Zero);
		
		/** 
		 * @brief Resets the tensor as if default initialized.
		 */
		void reset();
		
		
		/** 
		 * @brief Resets the tensor to the given dimensions and uses the given data.
		 * @param _newDim the dimensions of the new tensor.
		 * @param _newData new dense data in row-major order.
		 */
		void reset(DimensionTuple _newDim, const std::shared_ptr<value_t>& _newData);
		
		
		/** 
		 * @brief Resets the tensor to the given dimensions with data @a _newData.
		 * @param _newDim the dimensions of the new tensor.
		 * @param _newData new dense data in row-major order.
		 */
		void reset(DimensionTuple _newDim, std::unique_ptr<value_t[]>&& _newData);
		
		
		/**
		 * @brief Resets the tensor to a given dimensionstuple with sparse data @a _newData
		 */
		void reset(DimensionTuple _newDim, std::map<size_t, value_t>&& _newData);
		
		
		/** 
		 * @brief Reinterprets the dimensions of the tensor.
		 * @details For this simple reinterpretation it is nessecary that the size implied by the new dimensions is the same as to old size 
		 * (a vector with 16 entries cannot be interpreted as a 10x10 matrix, but it can be interpreted as a 4x4 matrix). If a real change in size is 
		 * required use resize_mode() instead.
		 * @param _newDimensions the dimensions the tensor shall be interpreted to have. 
		 */
		void reinterpret_dimensions(DimensionTuple _newDimensions);
		
		/** 
		 * @brief Resizes a specific mode of the Tensor.
		 * @details Use this function only if the content of the tensor shall stay, otherwise use reset().
		 * @param _mode the mode to resize.
		 * @param _newDim the new dimension that mode shall have.
		 * @param _cutPos the position within the selected mode in front of which slates are inserted 
		 * or removed. By default the current dimension, i.e new slates are added after the last current one
		 * and removed starting from the last one.
		 */
		void resize_mode(const size_t _mode, const size_t _newDim, size_t _cutPos=~0ul);
		
		
		/** 
		 * @brief Fixes a specific mode to a specific value, effectively reducing the order by one.
		 * @param _mode the mode in which the slate shall be fixed, e.g. 0 to fix the first mode.
		 * @param _slatePosition the position in the corresponding mode that shall be used. 0 <= _slatePosition < dimension[_mode]
		 */
		void fix_mode(const size_t _mode, const size_t _slatePosition);
		
		
		/** 
		 * @brief Removes a single slate from the Tensor, reducing dimension[_mode] by one.
		 * @param _mode the mode that will be reduced.
		 * @param _pos the index within the selected mode for which the slate shall be removed.
		 */
		void remove_slate(const size_t _mode, const size_t _pos);
		
		
		/** 
		 * @brief Performs the trace over the given modes
		 * @param _firstMode the first mode involved in the trace.
		 * @param _secondMode the second mode involved in the trace.
		 */
		void perform_trace(size_t _firstMode, size_t _secondMode);
		
		
		/** 
		 * @brief Modifies the diagonal entries according to the given function.
		 * @details In this overload only the current diagonal entries are passed to @a _f, one at a time. At the moment this is only defined for matricies.
		 * @param _f the function to call to modify each entry.
		 */
		void modify_diagonal_entries(const std::function<void(value_t&)>& _f);
		
		
		/** 
		 * @brief Modifies the diagonal entries according to the given function.
		 * @details In this overload the current diagonal entries are passed to @a _f, one at a time, together with their position on the diagonal. At the moment this is only defined for matricies.
		 * @param _f the function to call to modify each entry.
		 */
		void modify_diagonal_entries(const std::function<void(value_t&, const size_t)>& _f);
		
		
		/** 
		 * @brief Modifies every entry according to the given function.
		 * @details In this overload only the current entry is passed to @a _f.
		 * @param _f the function to call to modify each entry.
		 */
		void modify_entries(const std::function<void(value_t&)>& _f);
		
		
		/** 
		 * @brief Modifies every entry according to the given function.
		 * @details In this overload the current entry together with its position, assuming row-major ordering is passed to @a _f.
		 * @param _f the function to call to modify each entry.
		 */
		void modify_entries(const std::function<void(value_t&, const size_t)>& _f);
		
		
		/** 
		 * @brief Modifies every entry according to the given function.
		 * @details In this overload the current entry together with its complete position is passed to @a _f.
		 * @param _f the function to call to modify each entry.
		 */
		void modify_entries(const std::function<void(value_t&, const MultiIndex&)>& _f);
		
		/** 
		 * @brief Adds the given Tensor with the given offsets to this one.
		 * @param _other Tensor that shall be added to this one, the orders must coincide.
		 * @param _offsets the offsets to be used.
		 */
		void offset_add(const Tensor& _other, const std::vector<size_t>& _offsets);
		
		/** 
		 * @brief Converts the Tensor to a dense representation.
		 */
		void use_dense_representation();
		
		
		/** 
		 * @brief Converts the Tensor to a dense representation if sparsity * sparsityFactor >= size
		 */
		void use_dense_representation_if_desirable();
		
		
		/** 
		 * @brief Converts the Tensor to a sparse representation.
		 */
		void use_sparse_representation(const value_t _eps = std::numeric_limits<value_t>::epsilon());
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		/** 
		 * @brief Creates a string representation of the Tensor.
		 * @note the mapping is not unique and can thus not be used to recreate the original tensor
		 * @return the string representation. 
		 */
		std::string to_string() const;
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Auxiliary functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		static size_t multiIndex_to_position(const MultiIndex& _multiIndex, const DimensionTuple& _dimensions);
		
		static MultiIndex position_to_multiIndex(size_t _position, const DimensionTuple& _dimensions);
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal Helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	protected:
		template<int sign>
		static void plus_minus_equal(Tensor& _me, const Tensor& _other);
		
		/// @brief Adds the given sparse data to the given full data
		static void add_sparse_to_full(const std::shared_ptr<value_t>& _denseData, const value_t _factor, const std::shared_ptr<const std::map<size_t, value_t>>& _sparseData);
		
		/// @brief Adds the given sparse data to the given sparse data
		static void add_sparse_to_sparse(const std::shared_ptr<std::map<size_t, value_t>>& _sum, const value_t _factor, const std::shared_ptr<const std::map<size_t, value_t>>& _summand);
		
	public:
		
		/// @brief Ensures that this tensor is the sole owner of its data. If needed new space is allocated and all entries are copied.
		void ensure_own_data();
		
		/// @brief Ensures that this tensor is the sole owner of its data space. If needed new space is allocated with entries left undefined.
		void ensure_own_data_no_copy();
		
		/// @brief Checks whether there is a non-trivial scaling factor and applies it if nessecary.
		void apply_factor();
		
		/// @brief Checks whether there is a non-trivial factor and applies it. Even if no factor is applied ensure_own_data() is called.
		void ensure_own_data_and_apply_factor();
	};
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - External functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	
    /** 
     * @brief Low-level contraction between Tensors.
     * @param _result Output for the result of the contraction.
     * @param _lhs left hand side of the contraction.
     * @param _rhs right hand side of the contraction.
     * @param _numModes number of indices that shall be contracted.
     */
    XERUS_force_inline void contract(Tensor& _result, const Tensor& _lhs,  const Tensor& _rhs, const size_t _numModes) {
        contract(_result, _lhs, false, _rhs, false, _numModes);
    }
    
    XERUS_force_inline Tensor contract(const Tensor& _lhs, const Tensor& _rhs, const size_t _numModes) {
        return contract(_lhs, false, _rhs, false, _numModes);
    }
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	
	/** 
	 * @brief Calculates the entrywise sum of @a _lhs and @a _rhs.
	 * @details To be well-defined it is required that the dimensions of  @a _lhs and @a _rhs coincide.
	 * @param _lhs the first summand.
	 * @param _rhs the second summand.
	 * @return the sum.
	 */
	Tensor operator+(Tensor _lhs, const Tensor& _rhs);
	
	/** 
	 * @brief Calculates the entrywise difference between @a _lhs and @a _rhs.
	 * @details To be well-defined it is required that the dimensions of  @a _lhs and @a _rhs coincide.
	 * @param _lhs the minuend.
	 * @param _rhs the subtrahend.
	 * @return the difference.
	 */
	Tensor operator-(Tensor _lhs, const Tensor& _rhs);
	
	/** 
	 * @brief Calculates the entrywise multiplication of the Tensor @a _tensor with the constant @a _factor.
	 * @details Internally this only results in a change in the global factor.
	 * @param _factor the factor to be used.
	 * @param _tensor the Tensor that shall be scaled.
	 * @return the resulting scaled Tensor.
	 */
	Tensor operator*(const value_t _factor, Tensor _tensor);
	
	/** 
	 * @brief Calculates the entrywise multiplication of the Tensor @a _tensor with the constant @a _factor.
	 * @details Internally this only results in a change in the global factor.
	 * @param _tensor the Tensor that shall be scaled.
	 * @param _factor the factor to be used.
	 * @return the resulting scaled Tensor.
	 */
	Tensor operator*(Tensor _tensor, const value_t _factor);
	
	/** 
	 * @brief Calculates the entrywise divison of the Tensor @a _tensor with the constant @a _divisor.
	 * @details Internally this only results in a change in the global factor.
	 * @param _tensor the Tensor that shall be scaled.
	 * @param _divisor the factor to be used.
	 * @return the resulting scaled Tensor.
	 */
	Tensor operator/(Tensor _tensor, const value_t _divisor);
	
	
	
	/** 
	* @brief Calculates the frobenius norm of the given tensor
	* @param _tensor the Tensor of which the frobenious norm shall be calculated.
	* @return the frobenius norm .
	*/
	static XERUS_force_inline value_t frob_norm(const Tensor& _tensor) { return _tensor.frob_norm(); }
	
	/** 
	* @brief Calculates the 1-norm of the given tensor
	* @param _tensor the Tensor of which the norm shall be calculated.
	* @return the 1-norm
	*/
	static XERUS_force_inline value_t one_norm(const Tensor& _tensor) { return _tensor.one_norm(); }
	
	/** 
	 * @brief Low-Level SVD calculation of a given Tensor @a _input = @a _U @a _S @a _Vt.
	 * @param _U Output Tensor for the resulting U.
	 * @param _S Output Tensor for the resulting S.
	 * @param _Vt Output Tensor for the resulting Vt.
	 * @param _input input Tensor of which the SVD shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the SVD is calculated.
	 */
	value_t calculate_svd(Tensor& _U, Tensor& _S, Tensor& _Vt, Tensor _input, const size_t _splitPos, const size_t _maxRank, const value_t _eps);
	
	/** 
	 * @brief Low-Level QR calculation of a given Tensor @a _input = @a _Q @a _R.
	 * @param _Q Output Tensor for the resulting Q.
	 * @param _R Output Tensor for the resulting R.
	 * @param _input input Tensor of which the QR shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the QR is calculated.
	 */
	void calculate_qr(Tensor& _Q, Tensor& _R, Tensor _input, const size_t _splitPos);
	
	/** 
	 * @brief Low-Level RQ calculation of a given Tensor @a _input = @a _R @a _Q.
	 * @param _R Output Tensor for the resulting R.
	 * @param _Q Output Tensor for the resulting Q.
	 * @param _input input Tensor of which the RQ shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the RQ is calculated.
	 */
	void calculate_rq(Tensor& _R, Tensor& _Q, Tensor _input, const size_t _splitPos);
	
	/** 
	 * @brief Low-Level QC calculation of a given Tensor @a _input = @a _Q @a _C.
	 * @details This is a rank revealing QR decomposition with coloum pivoting. In contrast to an QR
	 * the C is not nessecarily upper triangular.
	 * @param _Q Output Tensor for the resulting Q.
	 * @param _C Output Tensor for the resulting R.
	 * @param _input input Tensor of which the QC shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the QC is calculated.
	 */
	void calculate_qc(Tensor& _Q, Tensor& _C, Tensor _input, const size_t _splitPos);
	
	/** 
	 * @brief Low-Level CQ calculation of a given Tensor @a _input = @a _C @a _Q.
	 * @details This is a rank revealing RQ decomposition with coloum pivoting. In contrast to an RQ
	 * the C is not nessecarily upper triangular.
	 * @param _C Output Tensor for the resulting R.
	 * @param _Q Output Tensor for the resulting Q.
	 * @param _input input Tensor of which the CQ shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the CQ is calculated.
	 */
	void calculate_cq(Tensor& _C, Tensor& _Q, Tensor _input, const size_t _splitPos);
	
	/** 
	 * @brief Low-Level calculation of the pseudo inverse of a given Tensor.
	 * @details Currently doen by calculation of the SVD.
	 * @param _inverse Output the pseudo inverse.
	 * @param _input input Tensor of which the CQ shall be calculated.
	 * @param _splitPos index position at defining the matrification for which the pseudo inverse is calculated.
	 */
	void pseudo_inverse(Tensor& _inverse, const Tensor& _input, const size_t _splitPos);
	Tensor pseudo_inverse(const Tensor& _input, const size_t _splitPos);
	
	/** 
	 * @brief Solves the least squares problem ||@a _A @a _X - @a _B||_F.
	 * @param _X Output Tensor for the resulting X.
	 * @param _A input Tensor A.
	 * @param _B input Tensor b.
	 * @param _extraDegree number of modes that @a _X and @a _B share and for which the least squares problem is independently solved.
	 */
	void solve_least_squares(Tensor& _X, const Tensor& _A, const Tensor& _B, const size_t _extraDegree = 0);
	
	/**
	 * @brief Solves the equation Ax = b for x. If the solution is not unique, the output need not be the minimal norm solution.
	 * @param _X Output Tensor for the result
	 * @param _A input Operator A
	 * @param _B input right-hand-side b
	 * @param _extraDegree number of modes that @a _x and @a _B sharefor which the solution should be computed independently.
	 */
	void solve(Tensor &_X, const Tensor &_A, const Tensor &_B, size_t _extraDegree = 0);

	/**
	 * @brief Solves the equation A*x = lambda*x for x and lambda and A symmetric. It calls the LAPACK routine DGEEV. It calculates the eigenvalue with smallest real part.
	 * @param _X Output Tensor for the result
	 * @param _A input Operator A symmetric with respect to matrification
	 * @return the smallest eigenvalue
	 */
	double get_smallest_eigenvalue(Tensor &_X, const Tensor &_A);

	
	/**
	 * @brief calculates the entrywise product of two Tensors
	 */
	Tensor entrywise_product(const Tensor &_A, const Tensor &_B);
	
	/** 
	* @brief Checks whether two tensors are approximately equal.
	* @details Check whether ||@a _a - @a _b ||/(||@a a ||/2 + ||@a _b ||/2) < @a _eps, i.e. whether the relative difference in the frobenius norm is sufficently small.
	* @param _a the first test candidate.
	* @param _b the second test candidate
	* @param _eps the maximal relative difference between @a _a and @a _b.
	* @return TRUE if @a _a and @a _b are determined to be approximately equal, FALSE otherwise.
	*/
	bool approx_equal(const xerus::Tensor& _a, const xerus::Tensor& _b, const xerus::value_t _eps = EPSILON);
	
	/** 
	* @brief Checks whether two Tensors are approximately entrywise equal.
	* @details Check whether |@a _a[i] - @a _b[i] |/(|@a _a[i] | + |@a _b[i] | < @a _eps is fullfilled for all i.
	* @param _a the first test candidate.
	* @param _b the second test candidate
	* @param _eps the maximal relative difference between the entries of @a _a and @a _b.
	* @return TRUE if @a _a and @a _b are determined to be approximately equal, FALSE otherwise.
	*/
	bool approx_entrywise_equal(const xerus::Tensor& _a, const xerus::Tensor& _b, const xerus::value_t _eps = EPSILON);
	
	
	/** 
	 * @brief Compares the given Tensor entriewise to the given data.
	 * @param _tensor the tensor.
	 * @param _values the data to compare to, must be of the same size as the Tensor has entries (i.e. size).
	 * @param _eps (optional) epsilon used to determine whether an entry is equal to a data point. 
	 * @return TRUE if approx_equal() is true for all entries/data points, FALSE otherwise.
	 */
	bool approx_entrywise_equal(const xerus::Tensor& _tensor, const std::vector<value_t>& _values, const xerus::value_t _eps = EPSILON);
	
	/** 
	 * @brief Prints the Tensor to the given outStream
	 * @param _out the outstream to be printed to.
	 * @param _tensor the tensor.
	 * @return @a _out.
	 */
	std::ostream& operator<<(std::ostream& _out, const Tensor& _tensor);
	
	namespace misc {
		/**
		* @brief pipes all information necessary to restore the current tensor into @a _stream.
		* @note that this excludes header information
		*/
		void stream_writer(std::ostream &_stream, const Tensor &_obj, const FileFormat _format);

		/**
		* @brief tries to restore the tensor from a stream of data. 
		*/
		void stream_reader(std::istream &_stream, Tensor &_obj, const FileFormat _format);
	}
}
