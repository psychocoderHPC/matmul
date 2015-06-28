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

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <alpaka/alpaka.hpp>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil

    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaKernel
    {
    public:
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FCT_ACC auto operator()(
            TAcc const & acc,
            size_t const & m, size_t const & n, size_t const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, size_t const & lda,
            TElem const * const MATMUL_RESTRICT B, size_t const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, size_t const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::DimT<TAcc>::value == 2u,
                "The accelerator used for with MatMulKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const v2uiGridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            size_t const & uiGridThreadIdxX(v2uiGridThreadIdx[1u]);
            size_t const & uiGridThreadIdxY(v2uiGridThreadIdx[0u]);

            // Column and row inside the block of C to calculate.
            auto const v2uiBlockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
            size_t const & uiBlockThreadIdxX(v2uiBlockThreadIdx[1u]);
            size_t const & uiBlockThreadIdxY(v2uiBlockThreadIdx[0u]);

            // The block threads extents.
            auto const v2uiBlockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            size_t const & uiBlockThreadsExtentX(v2uiBlockThreadsExtents[1u]);
            size_t const & uiBlockThreadsExtentY(v2uiBlockThreadsExtents[0u]);
            //assert(uiBlockThreadsExtentX == uiBlockThreadsExtentY);
            size_t const & uiBlockThreadsExtent(uiBlockThreadsExtentX);

            // Shared memory used to store the current blocks of A and B.
            auto * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
            auto * const pBlockSharedB(pBlockSharedA + uiBlockThreadsExtentX*uiBlockThreadsExtentY);

            auto const uiSharedBlockIdx1d(uiBlockThreadIdxY*uiBlockThreadsExtentX + uiBlockThreadIdxX);

            // If the element is outside of the matrix, write zero into the shared block.
            bool const bInsideA = (uiGridThreadIdxY < m);
            bool const bInsideB = (uiGridThreadIdxX < n);
            bool const bInsideC = (bInsideA && bInsideB);

            TElem fCSum(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            auto const uiBlockMulCount(
                static_cast<size_t>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(uiBlockThreadsExtent))));
            for(size_t k2(0); k2<uiBlockMulCount; ++k2)
            {
                // Copy data to shared memory.
                auto const uiAIdxX(k2*uiBlockThreadsExtentX + uiBlockThreadIdxX);
                auto const uiAIdx1d(uiGridThreadIdxY*lda + uiAIdxX);
                pBlockSharedA[uiSharedBlockIdx1d] = (
                    ((!bInsideA) || (uiAIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[uiAIdx1d]);

                auto const uiBIdxY(k2*uiBlockThreadsExtentY + uiBlockThreadIdxY);
                auto const uiBIdx1d(uiBIdxY*ldb + uiGridThreadIdxX);
                pBlockSharedB[uiSharedBlockIdx1d] = (
                    ((!bInsideB) || (uiBIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[uiBIdx1d]);

                // Synchronize to make sure the sub-matrices are loaded before starting the computation.
                acc.syncBlockThreads();

                // Dyadic product within shared memory.
                for(size_t k3(0); k3<uiBlockThreadsExtent; ++k3)
                {
                    fCSum += pBlockSharedA[uiBlockThreadIdxY*uiBlockThreadsExtentX + k3]
                        * pBlockSharedB[k3*uiBlockThreadsExtentY + uiBlockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration.
                acc.syncBlockThreads();
            }

            if(bInsideC)
            {
                auto const uiIdxC1d(uiGridThreadIdxY*ldc + uiGridThreadIdxX);
                C[uiIdxC1d] = alpha * fCSum + beta * C[uiIdxC1d];
            }
        }
    };

    namespace alpaka
    {
        namespace kernel
        {
            namespace traits
            {
                //#############################################################################
                //! The trait for getting the size of the block shared extern memory for a kernel.
                //#############################################################################
                template<
                    typename TAcc>
                struct BlockSharedExternMemSizeBytes<
                    GemmAlpakaKernel,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem>
                    ALPAKA_FCT_HOST static auto getBlockSharedExternMemSizeBytes(
                        alpaka::Vec<alpaka::dim::DimT<TAcc>> const & vuiBlockThreadsExtents,
                        size_t const & m,
                        size_t const & n,
                        size_t const & k,
                        TElem const & alpha,
                        TElem const * const A,
                        size_t const & lda,
                        TElem const * const B,
                        size_t const & ldb,
                        TElem const & beta,
                        TElem * const C,
                        size_t const & ldc)
                    -> alpaka::Uint
                    {
                        boost::ignore_unused(m);
                        boost::ignore_unused(n);
                        boost::ignore_unused(k);
                        boost::ignore_unused(alpha);
                        boost::ignore_unused(A);
                        boost::ignore_unused(lda);
                        boost::ignore_unused(B);
                        boost::ignore_unused(ldb);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(C);
                        boost::ignore_unused(ldc);

                        // Reserve the buffer for the two blocks of A and B.
                        return 2u * vuiBlockThreadsExtents.prod() * sizeof(TElem);
                    }
                };
            }
        }
    }
#endif
