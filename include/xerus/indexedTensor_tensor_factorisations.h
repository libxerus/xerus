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
* @brief Header file for the classes defining factorisations of Tensors.
*/

#pragma once

#include <vector>
#include <limits>

#include "basic.h"

namespace xerus {
	class Tensor;
	
	namespace internal {
		template<class tensor_type> class IndexedTensorReadOnly;
		template<class tensor_type> class IndexedTensor;
	}
	
	/**
	* @brief Abstract super class for all tensor factorisations.
	*/
	class TensorFactorisation {
	public:
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const = 0;
		
		virtual ~TensorFactorisation() = default;
	};
	
	/**
	* @brief Helper class to allow an intuitive syntax for SVD factorisations.
	* @details The simplest example is (U(i,r1), S(r1,r2), Vt(r2,j)) = SVD(A(i,j)) to calculate the SVD of A. However A, U, S and Vt can
	*  also be a higher order Tensors. In order to calculate the SVD however a matrification imposed by the index order is used.
	*/
	class SVD : public TensorFactorisation{
	public:
		internal::IndexedTensorReadOnly<Tensor>* input;
		const double epsilon;
		const double softThreshold;
		const size_t maxRank;
		const bool preventZero;
		
		SVD(internal::IndexedTensorReadOnly<Tensor>&& _input) :
			input(&_input), epsilon(EPSILON), softThreshold(0.0), maxRank(std::numeric_limits<size_t>::max()), preventZero(false) { }
			
		SVD(internal::IndexedTensorReadOnly<Tensor>&& _input, const double _softTreshold, const bool _preventZero = false) : 
			input(&_input), epsilon(0.0), softThreshold(_softTreshold), maxRank(std::numeric_limits<size_t>::max()), preventZero(_preventZero) { }
			
		SVD(internal::IndexedTensorReadOnly<Tensor>&& _input, const size_t _maxRank, const double _epsilon = EPSILON) :
			input(&_input), epsilon(_epsilon), softThreshold(0.0), maxRank(_maxRank), preventZero(false) { }
			
		SVD(internal::IndexedTensorReadOnly<Tensor>&& _input, const size_t _maxRank, const double _epsilon, const double _softTreshold, const bool _preventZero) : 
			input(&_input), epsilon(_epsilon), softThreshold(_softTreshold), maxRank(_maxRank), preventZero(_preventZero) { }
		
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const override;
	};

	/**
	* @brief Helper class to allow an intuitive syntax for QR factorisations.
	* @details The simplest example is (Q(i,k), R(k,j)) = QR(A(i,j)) to calculate the QR of A. However A, Q and R can
	*  also be a higher order Tensors. In order to calculate the QR however a matrification imposed by the index order is used.
	*/
	class QR : public TensorFactorisation {
	public:
		internal::IndexedTensorReadOnly<Tensor>* input;
		QR(internal::IndexedTensorReadOnly<Tensor>&& _input) : input(&_input) { }
		
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const override;
	};

	/**
	* @brief Helper class to allow an intuitive syntax for RQ factorisations.
	* @details The simplest example is (R(i,k), Q(k,j)) = RQ(A(i,j)) to calculate the RQ of A. However A, Q and R can
	*  also be a higher order Tensors. In order to calculate the RQ however a matrification imposed by the index order is used.
	*/
	class RQ : public TensorFactorisation {
	public:
		internal::IndexedTensorReadOnly<Tensor>* input;
		RQ(internal::IndexedTensorReadOnly<Tensor>&& _input) : input(&_input) { }
		
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const override;
	};
	
	/**
	* @brief Helper class to allow an intuitive syntax for an rank revealing orthogonal factorisation.
	* @details This calculates a factorisation QC=A with orthogonal Q and r x m matrix C where r is typically not much larger than the rank of A.
	* The simplest example is (Q(i,k), C(k,j)) = QC(A(i,j)) to calculate the QC decomposition of A. However A, Q and C can
	*  also be a higher order Tensors.
	*/
	class QC : public TensorFactorisation {
	public:
		internal::IndexedTensorReadOnly<Tensor>* input;
		QC(internal::IndexedTensorReadOnly<Tensor>&& _input) : input(&_input) { }
		
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const override;
	};
	
	/**
	* @brief Helper class to allow an intuitive syntax for an rank revealing orthogonal factorisation.
	* @details This calculates a factorisation CQ=A with orthogonal Q and n x r matrix C where r is typically not much larger than the rank of A.
	* The simplest example is (C(i,k), Q(k,j)) = QC(A(i,j)) to calculate the CQ decomposition of A. However A, Q and C can
	*  also be a higher order Tensors.
	*/
	class CQ : public TensorFactorisation {
	public:
		internal::IndexedTensorReadOnly<Tensor>* input;
		CQ(internal::IndexedTensorReadOnly<Tensor>&& _input) : input(&_input) { }
		
		virtual void operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const override;
	};
}
