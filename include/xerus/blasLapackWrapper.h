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
* @brief Header file for the blas and lapack wrapper functions.
*/

#pragma once

#include "misc/standard.h"
#include <memory>


namespace xerus {
	/**
	* @brief In this namespace the minimal wrappers for the BLAS and LAPACK functions are collected.
	* @details As an end user of xerus it should never be nessecary to call any of these functions, unless
	* a seriously low level implementation of a critical part of an algorithm is required.
	*/
	namespace blasWrapper {
		
		//----------------------------------------------- LEVEL I BLAS ----------------------------------------------------------
		
		///@brief: Computes the one norm =||x||_1
		double one_norm(const double* const _x, const size_t _n);
		
		///@brief: Computes the two norm =||x||_2
		double two_norm(const double* const _x, const size_t _n);
		
		///@brief: Computes the dot product = x^T*y
		double dot_product(const double* const _x, const size_t _n, const double* const _y);
		
		
		//----------------------------------------------- LEVEL II BLAS ---------------------------------------------------------
		
		///@brief: Perfroms x = alpha*OP(A)*y
		void matrix_vector_product(double* const _x, const size_t _m, const double _alpha, const double* const _A, const size_t _n, const bool _transposed, const double* const _y);
		
		///@brief: Performs A = alpha*x*y^T
		void dyadic_vector_product(double* _A, const size_t _m, const size_t _n, const double _alpha, const double*const _x, const double*const _y);
		
		//----------------------------------------------- LEVEL III BLAS --------------------------------------------------------
		///@brief: Performs the Matrix-Matrix product C = alpha*OP(A) * OP(B)
		void matrix_matrix_product( double* const _C,
									const size_t _leftDim,
									const size_t _rightDim,
									const double _alpha,
									const double* const _A,
									const size_t _lda,
									const bool _transposeA,
									const size_t _middleDim,
									const double* const _B,
									const size_t _ldb,
									const bool _transposeB);
		
		///@brief: Performs the Matrix-Matrix product C = alpha*OP(A) * OP(B)
		static XERUS_force_inline void matrix_matrix_product( double* const _C,
									const size_t _leftDim,
									const size_t _rightDim,
									const double _alpha,
									const double* const _A,
									const bool _transposeA,
									const size_t _middleDim,
									const double* const _B,
									const bool _transposeB) {
			
			matrix_matrix_product( _C, _leftDim, _rightDim, _alpha, _A, _transposeA ? _leftDim : _middleDim, _transposeA, _middleDim, _B, _transposeB ? _middleDim : _rightDim, _transposeB);
		}
		
		//----------------------------------------------- LAPACK ----------------------------------------------------------------
		
		///@brief: Performs (U,S,V) = SVD(A)
		void svd( double* const _U, double* const _S, double* const _Vt, const double* const _A, const size_t _m, const size_t _n);

		///@brief: Performs (U,S,V) = SVD(A). Destroys A.
		void svd_destructive( double* const _U, double* const _S, double* const _Vt, double* const _A, const size_t _m, const size_t _n);
		
		
		///@brief: splits A = Q*C, with @a _C an rxn matrix (where r is the rank of @a _A) and @a _Q orthogonal.
		std::tuple<std::unique_ptr<double[]>, std::unique_ptr<double[]>, size_t> qc(const double* const _A, const size_t _m, const size_t _n);
		
		///@brief: splits A = Q*C, with @a _C an rxn matrix (where r is the rank of @a _A) and @a _Q orthogonal. Destroys A.
		std::tuple<std::unique_ptr<double[]>, std::unique_ptr<double[]>, size_t> qc_destructive(double* const _A, const size_t _m, const size_t _n);
		
		
		///@brief: splits A = C*Q, with @a _C an rxm matrix (where r is the rank of @a _A) and @a _Q orthogonal.
		std::tuple<std::unique_ptr<double[]>, std::unique_ptr<double[]>, size_t> cq(const double* const _A, const size_t _m, const size_t _n);
		
		///@brief: splits A = C*Q, with @a _C an rxm matrix (where r is the rank of @a _A) and @a _Q orthogonal. Destroys A.
		std::tuple<std::unique_ptr<double[]>, std::unique_ptr<double[]>, size_t> cq_destructive(double* const _A, const size_t _m, const size_t _n);
		
		
		///@brief: Performs (Q,R) = QR(A)
		void qr( double* const _Q, double* const _R, const double* const _A, const size_t _m, const size_t _n);
		
		///@brief: Performs (AtoQ,R) = QR(AtoQ)
		void inplace_qr(double* const _AtoQ, double* const _R, const size_t _m, const size_t _n);
		
		///@brief: Performs (Q,R) = QR(A), destroys A in the process
		void qr_destructive( double* const _Q, double* const _R, double* const _A, const size_t _m, const size_t _n);
		
		
		
		///@brief: Performs (R,Q) = RQ(A)
		void rq( double* const _R, double* const _Q, const double* const _A, const size_t _m, const size_t _n);
		
		///@brief: Performs (R,AtoQ) = RQ(AtoQ)
		void inplace_rq( double* const _R, double* const _AtoQ, const size_t _m, const size_t _n);
		
		///@brief: Performs (R,Q) = RQ(A), destroys A in the process
		void rq_destructive( double* const _R, double* const _Q, double* const _A, const size_t _m, const size_t _n);

		
		///@brief: Solves Ax = b for x
		void solve(double* _x, const double* _A, size_t _m, size_t _n, const double* _b, size_t _p);
		
		///@brief: Solves Ax = lambda*x for x, this calls the Lapack Routine DGEEV
		void solve_ev(double* const _x, double* const _re, double* const _im, const double* const _A, const size_t _n);
		
		///@brief: Solves min ||Ax - b||_2 for x
		void solve_least_squares( double* const _x, const double* const _A, const size_t _m, const size_t _n, const double* const _b, const size_t _p);
		
		///@brief: Solves min ||Ax - b||_2 for x
		void solve_least_squares_destructive( double* const _x, double* const _A, const size_t _m, const size_t _n, double* const _b, const size_t _p);
		
		
		
		///@brief: Performs B = A^-1
		void inverse( double* const _B, const double* const _A, const size_t _m, const size_t _n);
	}

}
