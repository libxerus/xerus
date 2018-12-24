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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static misc::UnitTest tensor_solve("Tensor", "solve_Ax_equals_b", [](){
    Index i,j,k;
          
    Tensor A1({4,2,2});
    A1[{0,0,0}] = 1;
    A1[{1,0,1}] = 1;
    A1[{2,1,0}] = 1;
    A1[{3,1,1}] = 1;

    Tensor b1({4});
    b1[0] = 73;
    b1[1] = -73;
    b1[2] = 128;
    b1[3] = 93;
    
    Tensor x1({2,2});
    
    
    x1(k,i) = (b1(j)/A1(j,k,i));
    
    TEST((b1[0] - x1[{0,0}]) < 1e-14);
    TEST((b1[1] - x1[{0,1}]) < 1e-14);
    TEST((b1[2] - x1[{1,0}]) < 1e-14);
    TEST((b1[3] - x1[{1,1}]) < 1e-14);
    
    Tensor A2({4,2,2});
    A2[{0,0,0}] = 1;
    A2[{1,0,1}] = 1;
    A2[{2,1,0}] = 0;
    A2[{3,1,1}] = 0;
    
    Tensor b2({4});
    b2[0] = 73;
    b2[1] = -73;
    b2[2] = 0;
    b2[3] = 0;
    
    Tensor x2({2,2});
    
    x2(k,i) = (b2(j)/A2(j,k,i));
    
    TEST((b2[0] - x2[{0,0}]) < 1e-14);
    TEST((b2[1] - x2[{0,1}]) < 1e-14);
    TEST((x2[{1,0}]) < 1e-14);
    TEST((x2[{1,1}]) < 1e-14);
});

static misc::UnitTest tensor_solve_smallest_ev("Tensor", "get smallest eigenvalue of Matrix", [](){
    Index i,j,k,l,m,n,o,p;

    Tensor A1 = Tensor::random({4,3,4,3});
    Tensor x1({4,3});
    double lambda1 = get_smallest_eigenvalue(x1,A1);
    TEST(frob_norm(A1(i,j,k,l)*x1(k,l) - lambda1 * x1(i,j)) < 1e-13);

    Tensor A2 = Tensor::random({4,3,4,5,4,3,4,5});
		Tensor x2({4,3,4,5});
		double lambda2 = get_smallest_eigenvalue(x2,A2);
		TEST(frob_norm(A2(i,j,k,l,m,n,o,p)*x2(m,n,o,p) - lambda2 * x2(i,j,k,l)) < 5e-2);
});


static misc::UnitTest solve_vs_lsqr("Tensor", "solve vs least squares", [](){
	const size_t N = 500;
	Tensor A({N, N});
	for (size_t i=0; i<N; ++i) {
		if (i>0) A[{i, i-1}] = -1;
		if (i<N-1) A[{i, i+1}] = -1;
		A[{i,i}] = 2;
	}
	A[0] = 1;
	
	Tensor B = Tensor::random({N});

	Index i,j,k;
	
	// sparse
// 	LOG(blabla, "sparse start");
	Tensor X;
	solve(X, A, B);
// 	LOG(blabla, "sparse end");
	MTEST(frob_norm(A(i,j)*X(j)-B(i)) < 1e-10, "1 " << frob_norm(A(i,j)*X(j)-B(i)));
	
	A.use_dense_representation();
	// dense (cholesky)
// 	LOG(blabla, "chol start");
	Tensor X2;
	solve(X2, A, B);
// 	LOG(blabla, "chol end");
	MTEST(frob_norm(A(i,j)*X(j)-B(i)) < 1e-10, "2 " << frob_norm(A(i,j)*X(j)-B(i)));
	
	MTEST(approx_equal(X, X2, 1e-10), X.frob_norm() << " " << X2.frob_norm() << " diff: " << frob_norm(X-X2));
	
	// dense (LDL)
	A[0] = -0.9;
// 	LOG(blabla, "LDL start");
	solve(X, A, B);
// 	LOG(blabla, "LDL end");
	MTEST(frob_norm(A(i,j)*X(j)-B(i)) < 1e-10, "3 " << frob_norm(A(i,j)*X(j)-B(i)));
	
	// dense (LU)
	A[1] = -0.9;
// 	LOG(blabla, "LU start");
	solve(X, A, B);
// 	LOG(blabla, "LU end");
	MTEST(frob_norm(A(i,j)*X(j)-B(i)) < 1e-10, "4 " << frob_norm(A(i,j)*X(j)-B(i)));
	
	// dense (SVD)
	A[1] = -0.9;
// 	LOG(blabla, "SVD start");
	solve_least_squares(X, A, B);
// 	LOG(blabla, "SVD end");
	MTEST(frob_norm(A(i,j)*X(j)-B(i)) < 1e-10, "5 " << frob_norm(A(i,j)*X(j)-B(i)));
	
});

static misc::UnitTest tensor_solve_sparse("Tensor", "solve_sparse", [](){
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::normal_distribution<double> dist(0.0, 1.0);
	const size_t N = 100;
	std::uniform_int_distribution<size_t> eDist(1, N*N-1);
	
	Index i,j,k;
	
	Tensor id = Tensor::identity({N,N});
	Tensor r = Tensor({N}, [](size_t _i)->value_t{return double(_i);});
	Tensor x;
	x(i) = r(j) / id(j,i);
	MTEST(frob_norm(x-r) < 1e-14, "d " << frob_norm(x-r));
	
	r.use_sparse_representation();
	x(i) = r(j) / id(j,i);
	MTEST(frob_norm(x-r) < 1e-14, "d " << frob_norm(x-r));
	r.use_dense_representation();
	
	// consistency with dense solve:
	for (size_t n=0; n<N*3; ++n) {
		id[eDist(rnd)] = dist(rnd);
	}
	id.use_sparse_representation();
	
	// test faithful reconstruction
	internal::CholmodSparse idt(id.get_unsanitized_sparse_data(), N, N, false);
	Tensor id2({N,N});
	id2.get_unsanitized_sparse_data() = idt.to_map();
	MTEST(frob_norm(id-id2) < 1e-12, frob_norm(id-id2)); 
	
	Tensor fid(id);
	fid.use_dense_representation();
	Tensor fx;
	TEST(id.is_sparse());
	
	fx(i) = r(j) / fid(j,i);
	x(i) = r(j) / id(j,i);
	MTEST(frob_norm(id(j,i)*x(i) - r(j))/frob_norm(x)<1e-12, frob_norm(id(j,i)*x(i) - r(j))/frob_norm(x));
	MTEST(frob_norm(fid(j,i)*fx(i) - r(j))/frob_norm(x)<1e-12, frob_norm(fid(j,i)*fx(i) - r(j))/frob_norm(x));
	MTEST(frob_norm(fx-x)/frob_norm(x)<1e-12, frob_norm(fx-x)/frob_norm(x));
});

static misc::UnitTest tensor_solve_trans("Tensor", "solve_transposed", [](){
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::normal_distribution<double> dist(0.0, 1.0);
	const size_t N = 100;
	std::uniform_int_distribution<size_t> eDist(1,N*N-1);
	
	Index i,j,k;
	
	Tensor A = Tensor::identity({N,N});
	for (size_t n=0; n<N*3; ++n) {
		A[eDist(rnd)] = dist(rnd);
	}
	A.use_sparse_representation();
	Tensor At;
	At(i,j) = A(j,i);
	At.use_sparse_representation();
	
	Tensor r = Tensor({N}, [](size_t _i)->value_t{return double(_i);});
	Tensor x1, x2;
	x1(i) = r(j) / A(i,j);
	x2(i) = r(j) / At(j,i);
	MTEST(frob_norm(x1-x2) < 1e-12, "s " << frob_norm(x1-x2));
	
	A.use_dense_representation();
	At.use_dense_representation();
	
	Tensor x3, x4, residual;
	x3(i) = r(j) / A(i,j);
	x4(i) = r(j) / At(j,i);
	
	residual(i) = A(i,j) * (x3(j) - x4(j));
	MTEST(frob_norm(x3-x4) < 1e-12, "d " << frob_norm(x3-x4) << " residual: " << frob_norm(residual));
	
	residual(i) = A(i,j) * (x1(j) - x3(j));
	MTEST(frob_norm(x1-x3)/frob_norm(x1) < 1e-12, "sd " << frob_norm(x1-x3)/frob_norm(x1) << " residual: " << frob_norm(residual));
});


static misc::UnitTest tensor_solve_matrix("Tensor", "solve_matrix", [](){
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> nDist(1, 100);
	std::uniform_int_distribution<size_t> n2Dist(1, 10);
	std::uniform_int_distribution<size_t> dDist(1, 3);
	std::uniform_int_distribution<size_t> d2Dist(0, 3);
	std::normal_distribution<double> realDist(0, 1);
	
	Index i,j,k;
	
	for(size_t run = 0; run < 10; ++run) {
		std::vector<size_t> mDims, nDims, pDims;
		const size_t degM = dDist(rnd);
		const size_t degN = dDist(rnd);
		const size_t degP = d2Dist(rnd);
		for(size_t xi = 0; xi < degM; ++xi) { mDims.push_back(n2Dist(rnd)); }
		for(size_t xi = 0; xi < degN; ++xi) { nDims.push_back(n2Dist(rnd)); }
		for(size_t xi = 0; xi < degP; ++xi) { pDims.push_back(n2Dist(rnd)); }
		
		
		auto A = Tensor::random(mDims | nDims);
		A *= realDist(rnd);
		Tensor B;
		Tensor realX = Tensor::random(nDims | pDims);
		
		B(i^degM, k^degP) = A(i^degM, j^degN)*realX(j^degN, k^degP);
		
		auto factor = realDist(rnd);
		B *= factor;
		realX *= factor;
		
		Tensor X;
		
		solve_least_squares(X, A, B, degP);
		
		Tensor residual;
		
		residual(i^degM, k^degP) = A(i^degM, j^degN)*X(j^degN, k^degP) - B(i^degM, k^degP);
		MTEST(frob_norm(residual) < 1e-10, frob_norm(residual));
		
		
		X(j^degN, k^degP) = B(i^degM, k^degP) / A(i^degM, j^degN);  //solve_least_squares(X, A, B, degP);
		
		residual(i^degM, k^degP) = A(i^degM, j^degN)*X(j^degN, k^degP) - B(i^degM, k^degP);
		MTEST(frob_norm(residual) < 1e-10, frob_norm(residual));
	}
});

static misc::UnitTest tensor_solve_w_extra_degree("Tensor", "solve with extra degrees", [](){
    Index ii,jj,kk,ll,mm,nn;
  	Tensor A = xerus::Tensor::random({2,2});
  	Tensor B = xerus::Tensor::random({2,2});
  	Tensor X({2,2});
  	Tensor tmp({2,2});

  	//non symmetric
  	xerus::solve(X, A, B,1);

  	tmp(ii,kk) = A(ii,jj)*X(jj,kk);
    TEST((tmp - B).frob_norm() < 1e-13);

    //symmetric
  	A(ii,jj) = A(ii,jj) + A(jj,ii);
  	xerus::solve(X, A, B,1);
  	tmp(ii,kk) = A(ii,jj)*X(jj,kk);
    TEST((tmp - B).frob_norm() < 1e-13);

    //higher order
  	Tensor A2 = xerus::Tensor::random({5,5,5,5});
  	Tensor B2 = xerus::Tensor::random({5,5,5,5});
  	Tensor X2({5,5,5,5});
  	Tensor tmp2({5,5,5,5});

  	xerus::solve(X2, A2, B2,2);
		tmp2(ii^2,kk^2) = A2(ii^2,jj^2)*X2(jj^2,kk^2);
		TEST((tmp2 - B2).frob_norm() < 1e-13);

  	Tensor A3 = xerus::Tensor::random({5,5,5,5});
  	Tensor B3 = xerus::Tensor::random({5,5,5});
  	Tensor X3({5,5,5});
  	Tensor tmp3({5,5,5});

		xerus::solve(X3, A3, B3,1);
		tmp3(ii^2,kk) = A3(ii^2,jj^2)*X3(jj^2,kk);
		TEST((tmp3 - B3).frob_norm() < 1e-13);
});
