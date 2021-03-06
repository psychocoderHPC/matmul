#pragma once

//-----------------------------------------------------------------------------
//! Copyright (c) 2014-2015, Benjamin Worpitz
//! All rights reserved.
//!
//! Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met :
//! * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//! * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
//! * Neither the name of the TU Dresden nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//! IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//! HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//-----------------------------------------------------------------------------

#if defined(MATMUL_BUILD_PAR_MPI_CANNON_STD) || defined(MATMUL_BUILD_PAR_MPI_CANNON_MKL) || defined(MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS)

    #include <matmul/common/Config.h>   // TElem, TIdx

    #ifdef __cplusplus
        extern "C"
        {
    #endif
    #ifdef MATMUL_BUILD_PAR_MPI_CANNON_STD
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using the Cannon algorithm with blocking MPI communication and the basic optimized sequential GEMM for local computation.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A. All processes except root will ignore the value.
        //! \param lda Specifies the leading dimension of A. All processes except root will ignore the value.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B. All processes except root will ignore the value.
        //! \param ldb Specifies the leading dimension of B. All processes except root will ignore the value.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C. All processes except root will ignore the value.
        //! \param ldc Specifies the leading dimension of C. All processes except root will ignore the value.
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_mpi_cannon_block(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);

        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using the Cannon algorithm with non-blocking MPI communication and the basic optimized sequential GEMM for local computation.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A. All processes except root will ignore the value.
        //! \param lda Specifies the leading dimension of A. All processes except root will ignore the value.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B. All processes except root will ignore the value.
        //! \param ldb Specifies the leading dimension of B. All processes except root will ignore the value.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C. All processes except root will ignore the value.
        //! \param ldc Specifies the leading dimension of C. All processes except root will ignore the value.
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_mpi_cannon_nonblock(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_MPI_CANNON_MKL
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using the Cannon algorithm with non-blocking MPI communication and Intel MKL GEMM for local computation.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A. All processes except root will ignore the value.
        //! \param lda Specifies the leading dimension of A. All processes except root will ignore the value.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B. All processes except root will ignore the value.
        //! \param ldb Specifies the leading dimension of B. All processes except root will ignore the value.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C. All processes except root will ignore the value.
        //! \param ldc Specifies the leading dimension of C. All processes except root will ignore the value.
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_mpi_cannon_nonblock_blas_mkl(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef MATMUL_BUILD_PAR_MPI_CANNON_CUBLAS
        //-----------------------------------------------------------------------------
        //! (S/D)GEMM matrix-matrix product C = alpha * A * B + beta * C using the Cannon algorithm with non-blocking MPI communication and cuBLAS GEMM for local computation.
        //!
        //! \param m Specifies the number of rows of the matrix A and of the matrix C.
        //! \param n Specifies the number of columns of the matrix B and the number of columns of the matrix C.
        //! \param k Specifies the number of columns of the matrix A and the number of rows of the matrix B.
        //! \param alpha Scalar value used to scale the product of matrices A and B.
        //! \param A Array, size lda-by-k. The leading m-by-k part of the array must contain the matrix A. All processes except root will ignore the value.
        //! \param lda Specifies the leading dimension of A. All processes except root will ignore the value.
        //! \param B Array, size ldb-by-n. The leading k-by-n part of the array must contain the matrix B. All processes except root will ignore the value.
        //! \param ldb Specifies the leading dimension of B. All processes except root will ignore the value.
        //! \param beta Scalar value used to scale matrix C.
        //! \param C Array, size ldc-by-n. The leading m-by-n part of the array must contain the matrix C. All processes except root will ignore the value.
        //! \param ldc Specifies the leading dimension of C. All processes except root will ignore the value.
        //-----------------------------------------------------------------------------
        void matmul_gemm_par_mpi_cannon_nonblock_blas_cublas(
            TIdx const m, TIdx const n, TIdx const k,
            TElem const alpha,
            TElem const * const MATMUL_RESTRICT A, TIdx const lda,
            TElem const * const MATMUL_RESTRICT B, TIdx const ldb,
            TElem const beta,
            TElem * const MATMUL_RESTRICT C, TIdx const ldc);
    #endif
    #ifdef __cplusplus
        }
    #endif
#endif
