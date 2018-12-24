// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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

static misc::UnitTest ht_consitent("HT", "consistent networks", [](){
	auto ones = xerus::HTTensor::ones({3,4,5});
	auto rt = xerus::HTNetwork<false>::random({2,3,4,5,6,7},3);
	auto rtop = xerus::HTNetwork<true>::random({2,3,4,5,6,7},3);
	auto rt2 = xerus::Tensor::random({2,3,4,5,7,8});

	xerus::HTOperator rtop2(rt2,0.5);


	rt.require_valid_network();
	ones.require_valid_network();
	rtop.require_valid_network();
	rtop2.require_valid_network();
});


static misc::UnitTest ht_sum("HT", "sum", [](){
	//Random numbers
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::normal_distribution<value_t> dist (0.0, 1.0);
	std::uniform_int_distribution<size_t> intDist (1, 10);
	
	Index i;
	
	std::vector<size_t> dimensions;
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	Tensor A = Tensor::random(dimensions, dist);
	Tensor B = Tensor::random(dimensions, dist);
	Tensor C;
	HTTensor htA(A);
	HTTensor htB(B);
	HTTensor htC;
	HTOperator hoA(A);
	HTOperator hoB(B);
	HTOperator hoC;
	
	C(i&0) = A(i&0) + B(i&0);
	htC(i&0) = htA(i&0) + htB(i&0);
	hoC(i&0) = hoA(i&0) + hoB(i&0);
	MTEST(frob_norm(Tensor(htC)(i&0) - C(i&0)) < 5.0*1e-13, frob_norm(Tensor(htC)(i&0) - C(i&0)));
	MTEST(frob_norm(Tensor(hoC)(i&0) - C(i&0)) < 5.0*1e-13, frob_norm(Tensor(hoC) - C));
});

static misc::UnitTest ht_diff("HT", "difference", [](){
	Tensor A = Tensor::random({10,10,10,10});
	Tensor B = Tensor::random({10,10,10,10});
	Tensor C;
	HTTensor htA(A);
	HTTensor htB(B);
	HTTensor htC(4);

	Index i;
	C(i&0) = A(i&0) - B(i&0);
	htC(i&0) = htA(i&0) - htB(i&0);

	double fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	MTEST(fnorm < 1e-12, fnorm);
});


static misc::UnitTest ht_real_diff("HT", "real_difference", [](){
	HTTensor htA = HTTensor::random({10,10,10,10,10}, {4,4,4,4,4,4,4,4,4,4,4});
	HTTensor htB = HTTensor::random({10,10,10,10,10}, {4,4,4,4,4,4,4,4,4,4,4});
	HTTensor htC(5);

	Index i;
	htC(i&0) = htA(i&0) - htA(i&0);
	MTEST(frob_norm(htC(i&0)) < 1e-9, "1 " << frob_norm(htC(i&0)));

	htC(i&0) = htB(i&0) - htB(i&0);
	MTEST(frob_norm(htC(i&0)) < 1e-9, "2 " << frob_norm(htC(i&0)));

	htC(i&0) = (htA(i&0) + htB(i&0)) - (htA(i&0) + htB(i&0));
	MTEST(frob_norm(htC(i&0)) < 1e-9, "3 " << frob_norm(htC(i&0)));

	htC(i&0) = (htA(i&0) + htB(i&0));
	TEST(htC.ranks() == std::vector<size_t>({ 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 1, 1 }));
	htC(i&0) = (htB(i&0) + htA(i&0));

	TEST(htC.ranks() == std::vector<size_t>({ 8, 8, 8, 8, 8, 1, 8, 8, 8, 8, 8, 1, 1, 1 }));
	htC(i&0) = (htA(i&0) + htB(i&0)) - (htB(i&0) + htA(i&0));

	MTEST(frob_norm(htC(i&0)) < 1e-9, "4 " << frob_norm(htC(i&0)));

	htC(i&0) = (73*htA(i&0) + htB(i&0)) - (htB(i&0) + 73*htA(i&0));
	MTEST(frob_norm(htC(i&0)) < 1e-7, "5 " << frob_norm(htC(i&0)));

	htA = HTTensor::random({10,10,10,10,10}, {2,5,7,2,4,4,3,6,5,2,4});
	htC(i&0) = htA(i&0) - htA(i&0);
	MTEST(frob_norm(htC(i&0)) < 1e-9, "6 " << frob_norm(htC(i&0)));

	htC(i&0) = htB(i&0) - htB(i&0);
	MTEST(frob_norm(htC(i&0)) < 1e-9, "7 " << frob_norm(htC(i&0)));

	htC(i&0) = (htA(i&0) + htB(i&0)) - (htA(i&0) + htB(i&0));
	MTEST(frob_norm(htC(i&0)) < 1e-9, "8 " << frob_norm(htC(i&0)));

	htC(i&0) = (htA(i&0) + htB(i&0)) - (htB(i&0) + htA(i&0));
	MTEST(frob_norm(htC(i&0)) < 1e-9, "9 " << frob_norm(htC(i&0)));

	htC(i&0) = (73*htA(i&0) + htB(i&0)) - (htB(i&0) + 73*htA(i&0));
	MTEST(frob_norm(htC(i&0)) < 1e-8, "10 " << frob_norm(htC(i&0)));
});

static misc::UnitTest ht_diff_stacks("HT", "difference_of_HTStacks", [](){
	HTOperator htO = HTOperator::random({10,10,10,10,10,10,10,10}, {4,4,4,4,4,4});
	HTTensor htA = HTTensor::random({10,10,10,10}, {4,4,4,4,4,4});
	HTTensor htB = HTTensor::random({10,10,10,10}, {4,4,4,4,4,4});
	HTTensor htC;

	Index i,j,k;
	htC(i&0) = htO(i/2, j/2)*htA(j&0) - htO(i/2, j/2)*htA(j&0);
	LOG(unit_tests, "Frob norm 1 " << frob_norm(htC(i&0)));
	TEST(frob_norm(htC(i&0)) < 1e-7);

	htC(i&0) = htO(i/2, j/2)*htB(j&0) - htO(i/2, j/2)*htB(j&0);
	LOG(unit_tests, "Frob norm 2 " << frob_norm(htC(i&0)));
	TEST(frob_norm(htC(i&0)) < 1e-7);
});


//static misc::UnitTest ht_stack_norm("HT", "htStacks_frob_norm", [](){
//	const Index i, j, k;
//
//	HTOperator htO1 = HTOperator::identity({10,10,10,10,10,10,10,10,10,10});
//	HTOperator htO2 = HTOperator::identity({10,10,10,10,10,10,10,10,10,10});
//
//	MTEST(misc::approx_equal(frob_norm(htO1(i&0)*htO2(i&0)), double(misc::pow(10, 5))), frob_norm(htO1(i&0)*htO2(i&0)) << " vs " << misc::pow(10, 5));
//
//	TEST(misc::approx_equal(frob_norm(htO1(i/2, j/2)*htO2(j/2, k/2)), std::sqrt(misc::pow(10, 5))));
//});

static misc::UnitTest ht_spec_sumdiff("HT", "special_sum_diff", [](){
	Tensor A({10,10,10,10}); // NOTE that this is the 0 tensor
	Tensor B = Tensor::random({10,10,10,10});
	Tensor C;
	HTTensor htA(A);
	HTTensor htB(B);
	HTTensor htC(4);


	Index i;

	C(i&0) = A(i&0) + B(i&0);
	htC(i&0) = htA(i&0) + htB(i&0);
	TEST(frob_norm(Tensor(htC)(i&0) - C(i&0)) < 5*1e-12);
	TEST(frob_norm(Tensor(htC)(i&0) - Tensor(htB)(i&0)) < 3.1*1e-12);

	C(i&0) = B(i&0) + A(i&0);
	htC(i&0) = htB(i&0) + htA(i&0);
	TEST(frob_norm(Tensor(htC)(i&0) - C(i&0)) < 5*1e-12);
	TEST(frob_norm(Tensor(htC)(i&0) - Tensor(htB)(i&0)) < 3.1*1e-12);

	C(i&0) = A(i&0) - B(i&0);
	htC(i&0) = htA(i&0) - htB(i&0);
	MTEST(frob_norm(Tensor(htC)(i&0) - C(i&0)) < 5*1e-12, frob_norm(Tensor(htC)(i&0) - C(i&0)));
	MTEST(frob_norm(Tensor(htC)(i&0) + Tensor(htB)(i&0)) < 3.1*1e-12, frob_norm(Tensor(htC)(i&0) + Tensor(htB)(i&0)));

	C(i&0) = B(i&0) - A(i&0);
	htC(i&0) = htB(i&0) - htA(i&0);
	MTEST(frob_norm(Tensor(htC)(i&0) - C(i&0)) < 5*1e-12, frob_norm(Tensor(htC)(i&0) - C(i&0)));
	MTEST(frob_norm(Tensor(htC)(i&0) - Tensor(htB)(i&0)) < 3.1*1e-12, frob_norm(Tensor(htC)(i&0) - Tensor(htB)(i&0)));

	Tensor X({10});
	Tensor Y = Tensor::random({10});
  Tensor Z;
	HTTensor htX(X);
	HTTensor htY(Y);
	HTTensor htZ(1);

	Z(i&0) = X(i&0) + Y(i&0);
	htZ(i&0) = htX(i&0) + htY(i&0);
	TEST(frob_norm(Tensor(htZ)(i&0) - Z(i&0)) < 3.1*1e-12);
	TEST(frob_norm(Tensor(htZ)(i&0) - Tensor(htY)(i&0)) < 3.1*1e-12);

	Z(i&0) = Y(i&0) + X(i&0);
	htZ(i&0) = htY(i&0) + htX(i&0);
	TEST(frob_norm(Tensor(htZ)(i&0) - Z(i&0)) < 3.1*1e-12);
	TEST(frob_norm(Tensor(htZ)(i&0) - Tensor(htY)(i&0)) < 3.1*1e-12);

	Z(i&0) = X(i&0) - Y(i&0);
	htZ(i&0) = htX(i&0) - htY(i&0);
	TEST(frob_norm(Tensor(htZ)(i&0) - Z(i&0)) < 3.1*1e-12);
	TEST(frob_norm(Tensor(htZ)(i&0) + Tensor(htY)(i&0)) < 3.1*1e-12);

	Z(i&0) = Y(i&0) - X(i&0);
	htZ(i&0) = htY(i&0) - htX(i&0);
	TEST(frob_norm(Tensor(htZ)(i&0) - Z(i&0)) < 3.1*1e-12);
	TEST(frob_norm(Tensor(htZ)(i&0) - Tensor(htY)(i&0)) < 3.1*1e-12);
});

static misc::UnitTest ht_prod("HT", "product", [](){
	Index i,j,k,l;

	HTOperator htA = HTOperator::random({10,10,10,10}, 1);
	HTOperator htB = HTOperator::random({10,10,10,10}, 1);
	HTTensor htD = HTTensor::random({10,10}, 2);
	Tensor A(htA);
	Tensor B(htB);
	Tensor D(htD);

	Tensor C;
	HTTensor htCt;
	HTOperator htC;

	C(i^2) = A(i^2,j^2) * D(j^2);
	htCt(i^2) = htA(i^2,j^2) * htD(j^2);
	double fnorm = frob_norm(Tensor(htCt)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	C(i^2,k^2) = A(i^2,j^2) * B(j^2,k^2);
	htC(i^2,k^2) = htA(i^2,j^2) * htB(j^2,k^2);
	TEST(htC.nodes.size() == 4);
	fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	C(i/2,k/2) = A(j/2,i/2) * B(j/2,k/2);
	htC(i^2,k/2) = htA(j^2,i/2) * htB(j^2,k/2);
	fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	htC(i^2,k/2) = htB(j/2,k/2) * htA(j^2,i^2);
	fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	C(i^2,k^2) = A(i^2,j^2) * B(k^2,j^2);
	htC(i^2,k^2) = htA(i^2,j^2) * htB(k^2,j^2);
	fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	C(i^2,k^2) = A(j^2,i^2) * B(k^2,j^2);
	htC(i^2,k^2) = htA(j^2,i^2) * htB(k^2,j^2);
	fnorm = frob_norm(Tensor(htC)(i&0) - C(i&0));
	LOG(unit_tests, "frob_norm " << fnorm);
	TEST(fnorm < 10*10*10*10*1e-15);

	FAILTEST(htC(i^2,k^2) = htA(j^2,i) * htB(k^2,j^2));
	FAILTEST(htC(i^2,k^2) = htA(j^2,i^2) * htB(k^2,k^2));
});

static misc::UnitTest ht_id("HT", "identities", [](){
	Tensor I = Tensor({2,2,2,2},[](const std::vector<size_t>& _idx)->value_t{
		if ((_idx[0] == _idx[2] && _idx[1] == _idx[3]) || (_idx[0] == 0 && _idx[1]==1 &&_idx[2] ==0 && _idx[3]==0)) {
			return 1;
		} else {
			return 0;
		}
	}
	);
	HTOperator htI(I);
	HTOperator htC(4);
	Tensor C;
	Index i,j,k;

	htC(i^2,k^2) = htI(i^2,j^2) * htI(j^2,k^2);
	C(i^2,k^2) = I(i^2,j^2) * I(j^2,k^2);
	LOG(unit_test, frob_norm(C(i&0) - Tensor(htC)(i&0)));
	TEST(approx_equal(C, Tensor(htC), 1e-15));

	htC(k^2,i^2) = htI(i^2,j^2) * htI(j^2,k^2);
	C(k^2,i^2) = I(i^2,j^2) * I(j^2,k^2);
	LOG(unit_test, frob_norm(C(i&0) - Tensor(htC)(i&0)));
	TEST(approx_equal(C, Tensor(htC), 1e-15));

	htC(i^2,k^2) = htI(i^2,j^2) * htI(k^2,j^2);
	C(i^2,k^2) = I(i^2,j^2) * I(k^2,j^2);
	LOG(unit_test, frob_norm(C(i&0) - Tensor(htC)(i&0)));
	TEST(approx_equal(C, Tensor(htC), 1e-15));

	htC(k^2,i^2) = htI(i^2,j^2) * htI(k^2,j^2);
	C(k^2,i^2) = I(i^2,j^2) * I(k^2,j^2);
	LOG(unit_test, frob_norm(C(i&0) - Tensor(htC)(i&0)));
	TEST(approx_equal(C, Tensor(htC), 1e-15));
});

static misc::UnitTest ht_trans("HT", "transpose", [](){
	Tensor A = Tensor::random({10,10,10,10});
	Tensor B;
	HTOperator htA(A);
	Index i,j;

	B(i^2,j^2) = A(j^2,i^2);
	htA.transpose();
	LOG(unit_test, frob_norm(B(i&0) - Tensor(htA)(i&0)));
	TEST(approx_equal(B, Tensor(htA), 1e-14));
});

static misc::UnitTest ht_axb("HT", "ax_b", [](){
	HTTensor X = HTTensor::random({10,10,10}, {2,2,2,2,2});
	HTTensor B = HTTensor::random({10,10,10}, {2,2,2,2,2});

	Tensor I({10,10,10,10,10,10}, [](const std::vector<size_t> &_idx) {
		if (_idx[0]==_idx[3] && _idx[1] == _idx[4] && _idx[2] == _idx[5]) {
			return 1.0;
		} else {
			return 0.0;
		}
	});

	HTOperator A(I);
	HTTensor T;
	HTTensor S;

	Index i,j,k;


	T(i^3) = A(i^3, j^3) * X(j^3);
	T(i^3) = T(i^3) - B(i^3);
	S(i^3) = A(i^3, j^3) * X(j^3) - B(i^3);
	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);

	Tensor fA(A);
	Tensor fX(X);
	Tensor fT;
	fT(i^3) = fA(i^3, j^3) * fX(j^3);
	TEST(frob_norm(fT(i^3) - fX(i^3))<1e-7);

	T(i^3) = A(i^3, j^3) * X(j^3);
	TEST(frob_norm(Tensor(T) - fT) < 1e-7);
	TEST(frob_norm(Tensor(T)(i^3) - fT(i^3)) < 1e-7);

	T(i^3) = A(i^3, j^3) * X(j^3);
	TEST(frob_norm(A(i^3, j^3) * X(j^3) - T(i^3)) < 1e-7);

	TEST(frob_norm(T(i^3) - X(i^3))<1e-7);
	T(i^3) = T(i^3) - X(i^3);
	S(i^3) = A(i^3, j^3) * X(j^3) - X(i^3);
	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
	TEST(frob_norm(S(i^3)) < 1e-7);

	T(i^3) = A(j^3, i^3) * X(j^3);
	TEST(frob_norm(T(i^3) - X(i^3))<1e-7);
	T(i^3) = T(i^3) - X(i^3);
	S(i^3) = A(j^3, i^3) * X(j^3) - X(i^3);
	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
	TEST(frob_norm(S(i^3)) < 1e-7);

	T(i^3) = A(j^3, i^3) * B(j^3);
	T(i^3) = T(i^3) - B(i^3);
	S(i^3) = A(j^3, i^3) * B(j^3) - B(i^3);
	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
	TEST(frob_norm(S(i^3)) < 1e-7);
});

static misc::UnitTest ht_opt("HT", "operator_times_tensor", [](){
	Tensor A = Tensor::random({6,6,6,6});
	Tensor B = Tensor::random({6,6,6,6});
	Tensor C = Tensor::random({6,6});
	Tensor D;
	Tensor Do;
	HTOperator htA(A);
	htA.round(2); A = Tensor(htA);
	HTOperator htB(B);
	htB.round(2); B = Tensor(htB);
	HTTensor htC(C);
	HTTensor htD(C);
	HTOperator htDo(A);

	Index i,j,k,l,m;
	htD(i^2) = htA(i^2,j^2) * htB(j^2,k^2) * htC(k^2);
	D(i^2) = A(i^2,j^2) * B(j^2,k^2) * C(k^2);
	MTEST(approx_equal(D, Tensor(htD), 3e-13), "1 " << frob_norm(D-Tensor(htD)) << " / " << frob_norm(D));

	htD(i^2) = htA(i^2,j^2) * htB(k^2,j^2) * htC(k^2);
	D(i^2) = A(i^2,j^2) * B(k^2,j^2) * C(k^2);
	MTEST(approx_equal(D, Tensor(htD), 2e-13), "2 " << frob_norm(D-Tensor(htD)) << " / " << frob_norm(D));

	htDo(i^2,k^2) = htA(i^2,j^2) * htB(j^2,k^2);
	Do(i^2,k^2) = A(i^2,j^2) * B(j^2,k^2);
	MTEST(approx_equal(Do, Tensor(htDo), 2e-13), "3 " << frob_norm(Do-Tensor(htDo)) << " / " << frob_norm(Do));

	htDo(i^2,k^2) = htA(i^2,j^2) * htA(j^2,k^2);
	Do(i^2,k^2) = A(i^2,j^2) * A(j^2,k^2);
	MTEST(approx_equal(Do, Tensor(htDo), 2e-13), "4 " << frob_norm(Do-Tensor(htDo)) << " / " << frob_norm(Do));

	htDo(i^2,l^2) = htA(i^2,j^2) * htB(j^2,k^2) * htA(l^2,k^2);
	Do(i^2,l^2) = A(i^2,j^2) * B(j^2,k^2) * A(l^2,k^2);
	MTEST(approx_equal(Do, Tensor(htDo), 2e-13), "5 " << frob_norm(Do-Tensor(htDo)) << " / " << frob_norm(Do));

	htDo(i^2,l^2) = htA(i^2,j^2) * htB(j^2,k^2) * htB(l^2,k^2);
	Do(i^2,l^2) = A(i^2,j^2) * B(j^2,k^2) * B(l^2,k^2);
	MTEST(approx_equal(Do, Tensor(htDo), 2e-15), "6 " << frob_norm(Do-Tensor(htDo)) << " / " << frob_norm(Do));

	htDo(i^2,m^2) = htA(i^2,j^2) * htB(j^2,k^2) * htB(l^2,k^2) * htA(l^2,m^2);
	Do(i^2,m^2) = A(i^2,j^2) * B(j^2,k^2) * B(l^2,k^2) * A(l^2,m^2);
	MTEST(approx_equal(Do, Tensor(htDo), 2e-13), "7 " << frob_norm(Do-Tensor(htDo)) << " / " << frob_norm(Do));
});

static misc::UnitTest ht_fullcontr("HT", "full_contraction", [](){
	Tensor A = Tensor::random({10,10,10,10});
	Tensor B = Tensor::random({10,10,10,10});
	HTTensor htA(A);
	HTTensor htB(B);
	HTOperator hoA(A);
	HTOperator hoB(B);

	Index i;
	TEST(misc::approx_equal(frob_norm(A(i&0)), frob_norm(htA(i&0)), 3e-13));
	TEST(misc::approx_equal(frob_norm(A(i&0)), frob_norm(hoA(i&0)), 2e-13));
	TEST(misc::approx_equal(frob_norm(B(i&0)), frob_norm(htB(i&0)), 2e-13));
	TEST(misc::approx_equal(frob_norm(B(i&0)), frob_norm(hoB(i&0)), 1e-13));
	TEST(misc::approx_equal(frob_norm(A(i&0)-B(i&0)), frob_norm(htA(i&0)-htB(i&0)), 1e-12));
	TEST(misc::approx_equal(frob_norm(A(i&0)-B(i&0)), frob_norm(hoA(i&0)-hoB(i&0)), 1e-12));
	Tensor C;
	C() = A(i/1)*B(i&0);

	HTTensor htC(0);
	htC() = htA(i&0)*htB(i&0);
	TEST(misc::approx_equal(C[0], htC[0], 1e-12));

	HTOperator hoC(0);
	hoC() = htA(i&0)*htB(i&0);
	TEST(misc::approx_equal(C[{}], hoC[{}], 1e-12));
});
//
//static misc::UnitTest tt_disjoint("TT", "disjoint_product", [](){
//	//Random numbers
//	std::mt19937_64 &rnd = xerus::misc::randomEngine;
//	std::uniform_int_distribution<size_t> dimDist(1, 5);
//
//	std::vector<size_t> dimsA;
//	std::vector<size_t> dimsB;
//
//	const size_t D = 5;
//	Index i,j;
//
//	for(size_t d = 0; d <= D; ++d) {
//		Tensor A = Tensor::random(dimsA);
//		Tensor B = Tensor::random(dimsB);
//		Tensor C;
//		TTTensor ttA(A);
//		TTTensor ttB(B);
//		TTTensor ttC;
//
//
//		ttC = dyadic_product(ttA, ttB);
//		C(i/2,j/2) = A(i&0)*B(j&0);
//
//		LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//		TEST(approx_equal(C, Tensor(ttC), 1e-13));
//
//		dimsA.push_back(dimDist(rnd));
//		dimsB.push_back(dimDist(rnd));
//	}
//});
