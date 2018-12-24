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
* @brief Header file for some additional math functions.
*/

#pragma once

#include <limits>
#include <cmath>
#include "standard.h"
#include "sfinae.h"

namespace xerus {
	namespace misc {
		///@brief: Calculates the signum (-1, 0, 1) of a given value.
		template<class T> 
		constexpr int sgn(const T _value) noexcept {
			return (T(0) < _value) - (_value < T(0));
		}
		
		
		///@brief: Calculates _a*_a.
		template<class T>
		constexpr T sqr(const T &_a) noexcept {
			return _a*_a;
		}
		
		
		///@brief: Calculates _base^_exp by binary exponentiation
		template<class T, class I, typename std::enable_if<std::is_integral<I>::value && std::is_unsigned<I>::value, bool>::type = true> 
		constexpr T pow(const T &_base, const I _exp) noexcept {
			return _exp==0 ? T(1) : (_exp%2==0 ? pow(T(_base*_base), _exp/2) : T(_base*pow(_base, _exp-1)));
		}
		
		
		///@brief: Calculates _base^_exp by binary exponentiation
		template<class T, class I, typename std::enable_if<std::is_integral<I>::value && !std::is_unsigned<I>::value, bool>::type = true> 
		constexpr T pow(const T &_base, const I _exp) noexcept {
			return _exp==0 ? 
						T(1) :
						(
							_exp<0 ? 
								T( 1/pow(_base, -_exp)) 
							: 
								( _exp%2==0 ? pow(T(_base*_base), _exp/2) : T(_base*pow(_base, _exp-1)) )
						);
		}
		
		
		///@brief: Checks whether the relative difference between @a _a and @a _b (i.e. |a-b|/(|a|/2+|b|/2)) is smaller than @a _eps.
		template<class T, typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
		bool approx_equal(T _a, T _b, T _eps = 4*std::numeric_limits<T>::epsilon()) noexcept {
			return std::abs(_a-_b) <= _eps*0.5*(std::abs(_a)+std::abs(_b));
		}
		
		
		///@brief: Checks whether @a _a and @a _b are equal (for non floating point types).
		template<class T, typename std::enable_if<!std::is_floating_point<T>::value, bool>::type = true>
		bool approx_equal(T _a, T _b) {
			return _a == _b;
		}
		
		
		///@brief: Checks for hard equality ( == operator) without compiler warnings.
		template<class T>
		constexpr bool hard_equal(const T _a, const T _b) noexcept {
			#pragma GCC diagnostic push
				#pragma GCC diagnostic ignored "-Wfloat-equal"
				return _a == _b;
			#pragma GCC diagnostic pop
		}
		
		
		///@brief: Checks for hard inequality ( != operator) without compiler warnings.
		template<class T>
		constexpr bool hard_not_equal(const T _a, const T _b) noexcept {
			#pragma GCC diagnostic push
				#pragma GCC diagnostic ignored "-Wfloat-equal"
				return _a != _b;
			#pragma GCC diagnostic pop
		}
	}
}
