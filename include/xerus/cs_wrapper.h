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

#pragma once

#include <suitesparse/cs.h>

#include "sparseTensor.h"

namespace xerus {
    /// Unique_ptr wrapper that should always be used for the CS sparse matrix format
    typedef std::unique_ptr<cs_di, cs_di*(*)(cs_di*)> CsUniquePtr;
    
    bool sparse_result(const size_t _lhsDim, const size_t _midDim, const size_t _rhsDim, const size_t _lhsEntries, const size_t _rhsEntries);
    
    /// Allocates a CS sparse matrix with given dimensions and number of entries
    CsUniquePtr create_cs(const size_t _m, const size_t _n, const size_t _N); 

    // Converts an Indexed SparseTensor and an given matrification to the CSparse sparse matrix format
    CsUniquePtr to_cs_format(const IndexedTensorReadOnly<Tensor>& _tensor, const std::vector<Index>& _lhsIndices, const std::vector<Index>& _rhsIndices);
    
    /// Calculates the Matrix Matrix product between to CS sparse matrices
    CsUniquePtr matrix_matrix_product(const CsUniquePtr& _lhs, const CsUniquePtr& _rhs);
    
    /// Retransforms a CS sparse matrix to sparseTensor format
    SparseTensor from_cs_format(const CsUniquePtr& _cs_format, const std::vector<size_t>& _dimensions);
    
    void print_cs(const CsUniquePtr& _cs_format);
}