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
 * @brief Implementation of the Index class.
 */

#include <xerus/index.h>

#include <xerus/misc/standard.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

namespace xerus {
	
	std::atomic<uint64> Index::idThreadInitCounter(0);
	thread_local uint64 Index::idCounter = (idThreadInitCounter++)<<54;
	
	
	void Index::set_span(const size_t _degree) {
		REQUIRE(!flags[Flag::FIXED] || span == 1, "Fixed indices must have span one.");
		if(flags[Flag::INVERSE_SPAN]) {
			REQUIRE(!flags[Flag::FIXED], "Fixed indices must not have inverse span."); 
			REQUIRE(span <= _degree, "Index with inverse span would have negative actual span. Tensor degree: " << _degree << ", inverse span " << span);
			span = _degree-span;
			flags.reset();
		} else if( flags[Flag::FRACTIONAL_SPAN] ) {
			REQUIRE(!flags[Flag::FIXED], "Fixed indices must not have fractional span.");
			REQUIRE(_degree%span == 0, "Fractional span must divide the tensor degree. Here tensor degree = " << _degree << ", span = " << span);
			span = _degree/span;
			flags.reset();
		}
	}
	
	
	size_t Index::actual_span(const size_t _degree) const {
		if(flags[Flag::INVERSE_SPAN]) {
			REQUIRE(!flags[Flag::FIXED], "Fixed indices must not have inverse span."); 
			REQUIRE(span <= _degree, "Index with inverse span would have negative actual span. Tensor degree: " << _degree << ", inverse span " << span);
			return _degree-span;
		} 
		if( flags[Flag::FRACTIONAL_SPAN] ) {
			REQUIRE(!flags[Flag::FIXED], "Fixed indices must not have fractional span.");
			REQUIRE(_degree%span == 0, "Fractional span must divide the tensor degree. Here tensor degree = " << _degree << ", span = " << span);
			return _degree/span;
		}
		REQUIRE(!flags[Flag::FIXED] || span == 1, "Fixed indices must have span one.");
		return span;
	}
	
	
	bool Index::fixed() const {
		#ifndef XERUS_DISABLE_RUNTIME_CHECKS
			if(flags[Index::Flag::FIXED]) {
				REQUIRE(!flags[Flag::INVERSE_SPAN], "Fixed indices must not have inverse span."); 
				REQUIRE(!flags[Flag::FRACTIONAL_SPAN], "Fixed indices must not have fractional span.");
				REQUIRE(span == 1, "Fixed indices must have span one.");
				return true;
			} 
			return false;
		#else
			return flags[Index::Flag::FIXED];
		#endif
	}
	
	
	bool Index::open() const {
		REQUIRE(flags[Index::Flag::ASSINGED], "Check for index openness only allowed if the index is assinged.");
		return flags[Index::Flag::OPEN];
	}
	
	
	void Index::open(const bool _open) {
		flags[Flag::OPEN] = _open;
		IF_CHECK( flags[Flag::ASSINGED] = true; )
	}
	
	
	size_t Index::dimension() const {
		REQUIRE(flags[Index::Flag::ASSINGED], "Check for index dimension only allowed if the index is assinged.");
		return assingedDimension;
	}
	
	
	size_t Index::fixed_position() const {
		REQUIRE(flags[Index::Flag::FIXED], "fixed_position() must only be called for fixed indices.");
		return size_t(valueId);
	}
	
	
	Index Index::operator^(const size_t _span) const {
		REQUIRE(flags.none(), "Cannot apply ^ operator to an index that has any flag set.");
		return Index(valueId, _span);
	}
	
	
	Index Index::operator&(const size_t _span) const {
		REQUIRE(flags.none(), "Cannot apply & operator to an index that has any flag set.");
		return Index(valueId, _span, Flag::INVERSE_SPAN);
	}
	
	
	Index Index::operator/(const size_t _span) const {
		REQUIRE(flags.none(), "Cannot apply & operator to an index that has any flag set.");
		return Index(valueId, _span, Flag::FRACTIONAL_SPAN);
	}
	
	
	bool Index::all_open(const std::vector<Index>& _indices) {
		for(const Index& idx : _indices) {
			if(!idx.open()) { return false; }
		}
		return true;
	}
	
	
	
	
	bool operator==(const Index& _a, const Index& _b) {
		return _a.valueId == _b.valueId && !_a.fixed() && !_b.fixed();
	}
	
	bool operator!=(const Index& _a, const Index& _b) {
		return _a.valueId != _b.valueId || _a.fixed() || _b.fixed();
	}
	
	bool operator<(const Index& _a, const Index& _b) {
		return _a.valueId < _b.valueId;
	}
	
	std::ostream& operator<<(std::ostream& _out, const xerus::Index& _idx) {
		_out << "index#" << _idx.valueId << (_idx.flags[Index::Flag::INVERSE_SPAN] ? "&" : (_idx.flags[Index::Flag::FRACTIONAL_SPAN] ? "/" : "^")) << _idx.span;
		return _out;
	}
} // namespace xerus
