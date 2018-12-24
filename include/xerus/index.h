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
 * @brief Header file for the Index class.
 */

#pragma once

#include "basic.h"

#include <atomic>
#include <bitset>
#include <vector>

namespace xerus {
	
	/** 
	* @brief Class used to represent indices that can be used to write tensor calculations
	* in index notation.
	* @details The xerus::Index class is used to write indexed tensor expressen, e.g. A(i,j)*B(j,k).
	 * Here i,j,k are of type xerus::Index. The Index class provides numerous information used
	 * internally. As an enduser only the basic constructors and the ^, &, and / operators should
	 * of interest.
	*/
	class Index {
	public:
		#ifndef XERUS_DISABLE_RUNTIME_CHECKS
		/// Enum defining the possible flags an Index my possess.
			enum Flag {FIXED, INVERSE_SPAN, FRACTIONAL_SPAN, ASSINGED, OPEN, NUM_FLAGS};
		#else
		/// Enum defining the possible flags an Index my possess.
			enum Flag {FIXED, INVERSE_SPAN, FRACTIONAL_SPAN, OPEN, NUM_FLAGS};
		#endif
		
	private:
		/// @brief Counter that creates a unique local ID for every thread.
		static std::atomic<uint64> idThreadInitCounter;
		
		/// @brief Unqiue local ID of the thread.
		static thread_local uint64 idCounter;
		
	public:
		/// @brief Unqiue ID of the index. In case the fixed flag is set, this is the fixed position.
		uint64 valueId;
		
		/// @brief The span states how many dimensions are covered by the index.
		size_t span = 1;
		
		/// @brief The product of the external dimensions this index correstponds to. Only set for assinged indices.
		size_t assingedDimension = 0;
		
		/// @brief Bitset of all possible flags the index may possess.
		std::bitset<NUM_FLAGS> flags;
		
		
		
		/// @brief Empty constructor that creates a new Index with new ID. Use this to create indices.
		Index() : valueId(idCounter++), span(1) { };
		
		/// @brief Indices are default copy constructable.
		Index(const Index& _other) noexcept = default;
		
		/// @brief Integers are implicitly allowed to be casted to Index, to allow expression as A(i) = B(3,i), i.e. A is the third row of B.
		template<class I, typename std::enable_if<std::is_integral<I>::value && std::is_unsigned<I>::value, bool>::type = true> 
		Index(const I _i) noexcept : valueId(_i), span(1) {
			flags[Flag::FIXED] = true;
		}
		
		/// @brief Integers are implicitly allowed to be casted to Index, to allow expression as A(i) = B(3,i), i.e. A is the third row of B.
		template<class I, typename std::enable_if<std::is_integral<I>::value && !std::is_unsigned<I>::value, bool>::type = true> 
		Index(const I _i) : Index(static_cast<uint64>(_i)) {
			using ::xerus::misc::operator<<;
			XERUS_REQUIRE(_i >= 0, "Negative valueId= " <<_i<< " given");
		}

		/// @brief Internal constructor, do not use this unless you know what you are doing.
		explicit Index(const uint64 _valueId, const size_t _span) noexcept : valueId(_valueId), span(_span) {};
		
		/// @brief Internal constructor, do not use this unless you know what you are doing.
		explicit Index(const uint64 _valueId, const size_t _span, const Flag _flag1) noexcept : valueId(_valueId), span(_span) {
			flags[_flag1] = true;
		}
		
		
		
		/// @brief Indices are default assignable.
		Index& operator=(const Index&) = default;
		
		/// @brief Returns the span this index actually represents in a tensor of given order.
		void set_span(const size_t _degree);
		
		/// @brief Returns the span this index actually represents in a tensor of given order.
		size_t actual_span(const size_t _degree) const;
		
		/// @brief Checks whether the Index represents a fixed number.
		bool fixed() const;
		
		/// @brief Checks whether the index is open.
		bool open() const;
		
		/** @brief: Sets whether the index is open.
		 * @param _open new openness status of the index.
		 */
		void open(const bool _open);
		
		/** @brief: Returns the fixed position of a fixed index.
		 */
		size_t fixed_position() const;
		
		
		
		/// @brief Returns the (mult)Dimension assinged to this index.
		size_t dimension() const;
		
		/** @brief: Allow the creation of Indices covering more than one dimension using the power operator.
		 * E.g. A(i^2) = B(i^2) + C(i^2), defines A as the entriewise sum of the matrices B and C.
		 * @param _span The number of dimensions the index is supposed to cover.
		 */
		Index operator^(const size_t _span) const;
		
		/** @brief: Allow the creation of Indices covering all but x dimensions using the and operator. 
		 * E.g. A() = B(i&0) * C(i&0), defines A as the full contraction between B and C,
		 * indifferent of the actual degree of B and C.
		 * @param _span Number of dimensions NOT to be covered by this index.
		 */
		Index operator&(const size_t _span) const;
		
		/** @brief: Allow the creation of Indices covering an x-th fraction of the indices. 
		 * E.g. A(i&0) = B(i/2, j/2) * C(j&0), defines A as the contraction between the symmetric matrification
		 * of B and the vectorisation of C, indifferent of the actual degree of B and C.
		 * @param _span the fraction of the dimensions to be covered by this index.
		 */
		Index operator/(const size_t _span) const;
		
		/**
		 * @brief: Checks whether all indices in _indices are open. This is naturally only usefull
		 * for assinged indices, i.e. indices returned by IndexedTensorReadOnly::get_assigned_indices().
		 * @param _indices std::vector of indices to check. Every contained index is required to be assinged.
		 */
		static bool all_open(const std::vector<Index>& _indices);
	};
	
	/// @brief Two Indices are equal if their valueId coincides. Fixed indices are never equal.
	bool operator==(const Index& _a, const Index& _b);
	
	/// @brief Two Indices are equal if their valueId coincides. Fixed indices are never equal.
	bool operator!=(const Index& _a, const Index& _b);
	
	/// @brief The Comparision operator is needed for indices to be orderable in std::set, the valueId is used.
	bool operator<(const Index& _a, const Index& _b);
	
	/// @brief Allows to pretty print indices, giving the valueId and span.
	std::ostream& operator<<(std::ostream& _out, const xerus::Index& _idx); 
}
