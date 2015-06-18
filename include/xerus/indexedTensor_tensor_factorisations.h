// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2015 Benjamin Huber and Sebastian Wolf. 
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
 * @brief Header file for the classes defining factorisations of FullTensors.
 */

#pragma once

#include "indexedTensorWritable.h"

namespace xerus {

    /**
	 * @brief Helper class to allow an intuitive syntax for SVD factorisations.
	 * @details The simplest example is (U(i,r1), S(r1,r2), Vt(r2,j)) = SVD(A(i,j)) to calculate the SVD of A. However A, U, S and Vt can
	 *  also be a higher order Tensors. In order to calculate the SVD however a matrification imposed by the index order is used.
	 */
	class SVD {
    public:
        const IndexedTensorReadOnly<Tensor>& input;
        const double epsilon;
		const double softThreshold;
		const size_t maxRank;
        SVD(const IndexedTensorReadOnly<Tensor>& _input) : 
			input(_input), epsilon(EPSILON), softThreshold(false), maxRank(std::numeric_limits<size_t>::max()) { }
			
        SVD(const IndexedTensorReadOnly<Tensor>& _input, const double _epsilon, const bool _softTreshold = 0.0) : 
			input(_input), epsilon(_epsilon), softThreshold(_softTreshold), maxRank(std::numeric_limits<size_t>::max()) { }
			
        SVD(const IndexedTensorReadOnly<Tensor>& _input, const size_t _maxRank, const double _epsilon = EPSILON) : 
			input(_input), epsilon(_epsilon), softThreshold(false), maxRank(_maxRank) { }
        
        void operator()(const std::vector<const IndexedTensorWritable<Tensor>*>& _output) const ;
    };

	/**
	 * @brief Helper class to allow an intuitive syntax for QR factorisations.
	 * @details The simplest example is (Q(i,k), R(k,j)) = QR(A(i,j)) to calculate the QR of A. However A, Q and R can
	 *  also be a higher order Tensors. In order to calculate the QR however a matrification imposed by the index order is used.
	 */
    class QR {
    public:
        const IndexedTensorReadOnly<Tensor>* input;
        QR(const IndexedTensorReadOnly<Tensor>& _input) : input(&_input) { }
        
        void operator()(const std::vector<const IndexedTensorWritable<Tensor>*>& _output) const;
    };

    /**
	 * @brief Helper class to allow an intuitive syntax for RQ factorisations.
	 * @details The simplest example is (R(i,k), Q(k,j)) = RQ(A(i,j)) to calculate the RQ of A. However A, Q and R can
	 *  also be a higher order Tensors. In order to calculate the RQ however a matrification imposed by the index order is used.
	 */
	class RQ {
    public:
        const IndexedTensorReadOnly<Tensor>* input;
        RQ(const IndexedTensorReadOnly<Tensor>& _input) : input(&_input) { }
        
        void operator()(const std::vector<const IndexedTensorWritable<Tensor>*>& _output) const;
    };
	
	/**
	 * @brief Helper class to allow an intuitive syntax for an rank revealing orthogonal factorisation.
	 * @details This calculates a factorisation QC=A with orthogonal Q and r x m matrix C where r is typically not much larger than the rank of A.
	 * The simplest example is (Q(i,k), C(k,j)) = QC(A(i,j)) to calculate the QC decomposition of A. However A, Q and R can
	 *  also be a higher order Tensors.
	 */
	class QC {
	public:
		const IndexedTensorReadOnly<Tensor>* input;
		QC(const IndexedTensorReadOnly<Tensor>& _input) : input(&_input) { }
		
		void operator()(const std::vector<const IndexedTensorWritable<Tensor>*>& _output) const;
	};
}