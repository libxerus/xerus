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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static misc::UnitTest tt_round("TT", "TTTensor_Rounding", [](){
    Index i,j,k;
    
    Tensor A1 = Tensor::random({2});
    Tensor B1;
    TTTensor TTA1(A1, 1e-14);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    TTA1.round(1e-14);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    TTA1.round(1);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    
    Tensor A2 = Tensor::random({2,2});
    Tensor B2;
    TTTensor TTA2(A2, 1e-14);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    TTA2.round(1e-14);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    TTA2.round(2);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    
    Tensor A3 = Tensor::random({2,7});
    Tensor B3;
    TTTensor TTA3(A3, 1e-14);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    TTA3.round(1e-14);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    TTA3.round(2);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    
    Tensor A4 = Tensor::random({2,2,2,2,2,2,2,2});
	for (double eps = 1e-3; eps < 1.0; eps*=1.2) {
		TTTensor B4(A4);
		B4.round(eps);
		MTEST(approx_equal(A4, B4, eps), eps << " vs " << frob_norm(A4-Tensor(B4))/frob_norm(A4));
	}
    
    
    Tensor A5 = Tensor::random({5,6,3,1,4,2,8,1});
	for (double eps = 1e-3; eps < 1.0; eps*=1.2) {
		TTTensor B5(A5);
		B5.round(eps);
		MTEST(approx_equal(A5, B5, eps), eps << " vs " << frob_norm(A5-Tensor(B5))/frob_norm(A5));
	}
});


static misc::UnitTest tt_noround("TT", "no_rounding", [](){
    Index i,j,k;
	TTTensor a = TTTensor::random({2,2,2,2,2,2,2}, {2,2,2,2,2,2});
	TTTensor b(a);
	a.round(2);
	TEST(approx_equal(Tensor(a),Tensor(b), 1e-14));
	TTTensor c = TTTensor::random({2,2,2,2,2,2,2}, {2,2,2,2,2,2});
	a(i&0) = a(i&0) + 0.0*c(i&0);
	LOG(unit_test, a.ranks());
	a.round(2);
	TEST(approx_equal(Tensor(a), Tensor(b), 1e-14));
});
