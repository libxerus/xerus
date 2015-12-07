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


#include<xerus.h>

#include "../../include/xerus/misc/test.h"
using namespace xerus;

UNIT_TEST(Tensor, SelfContained,
    std::mt19937_64 rnd;
	std::uniform_int_distribution<size_t> spanDist(0, 3);
	std::uniform_int_distribution<size_t> dimDist(1, 4);
    std::uniform_real_distribution<value_t> dist (-1.0, 1.0); 
	
	const size_t numIdx = 10;
	
	std::vector<Index> indices;
	for(size_t i = 0; i < numIdx; ++i) { indices.emplace_back(); }
	
	for(size_t run = 0; run < 50; ++run) {
		std::vector<std::vector<size_t>> indexDim(numIdx);
		std::vector<size_t> indexSpan;
		
		for(size_t i = 0; i < numIdx; ++i) { 
			size_t k = spanDist(rnd);
			indexSpan.emplace_back(k);
			for(size_t j = 0; j < k; ++j) {
				indexDim[i].emplace_back(dimDist(rnd));
			}
		}
		
		FullTensor A1 = FullTensor::random(indexDim[0] | indexDim[1] | indexDim[2] | indexDim[3], rnd, dist);
		SparseTensor A2 = SparseTensor::random(indexDim[0] | indexDim[1] | indexDim[2] | indexDim[3], misc::product(indexDim[0] | indexDim[1] | indexDim[2] | indexDim[3])/13, rnd, dist);
		SparseTensor A3 = SparseTensor::random(indexDim[0] | indexDim[1] | indexDim[2] | indexDim[3], misc::product(indexDim[0] | indexDim[1] | indexDim[2] | indexDim[3])/20, rnd, dist);
		FullTensor B1 = FullTensor::random(indexDim[1] | indexDim[4] | indexDim[5] | indexDim[6], rnd, dist);
		FullTensor C1 = FullTensor::random(indexDim[5] | indexDim[0] | indexDim[7] | indexDim[8], rnd, dist);
		FullTensor C2 = FullTensor::random(indexDim[5] | indexDim[0] | indexDim[7] | indexDim[8], rnd, dist);
		FullTensor D1 = FullTensor::random(indexDim[2] | indexDim[8] | indexDim[3] | indexDim[6], rnd, dist);
		SparseTensor D2 = SparseTensor::random(indexDim[2] | indexDim[8] | indexDim[3] | indexDim[6], misc::product(indexDim[2] | indexDim[8] | indexDim[3] | indexDim[6])/27, rnd, dist);
		FullTensor E1 = FullTensor::random(indexDim[7] | indexDim[9], rnd, dist);
		FullTensor F1 = FullTensor::random(indexDim[4], rnd, dist);
		FullTensor res;

		res(indices[4]^indexSpan[4], indices[9]^indexSpan[9]) = 
			(A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			+A2(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3]))
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			*D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			+
			(A3(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			-A2(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3]))
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			*D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			-
			(A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			+A3(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3]))
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			*D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			;
		
			MTEST(frob_norm(res) <= 1e-10, frob_norm(res) << " >  1e-10");
			
			
			res(indices[4]^indexSpan[4], indices[9]^indexSpan[9]) = 
			A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*(C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			+ C2(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8]))
			*(D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			+ D2(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6]))
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			+
			A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*(C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			- C2(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8]))
			*(D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			- D2(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6]))
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			-
			2.0*A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*C1(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			*D1(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			-
			2.0*A1(indices[0]^indexSpan[0], indices[1]^indexSpan[1], indices[2]^indexSpan[2], indices[3]^indexSpan[3])
			*B1(indices[1]^indexSpan[1], indices[4]^indexSpan[4], indices[5]^indexSpan[5], indices[6]^indexSpan[6])
			*C2(indices[5]^indexSpan[5], indices[0]^indexSpan[0], indices[7]^indexSpan[7], indices[8]^indexSpan[8])
			*D2(indices[2]^indexSpan[2], indices[8]^indexSpan[8], indices[3]^indexSpan[3], indices[6]^indexSpan[6])
			*E1(indices[7]^indexSpan[7], indices[9]^indexSpan[9])
			;
		
			MTEST(frob_norm(res) <= 1e-10, frob_norm(res) << " >  1e-10");
	}
	
)
