//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2015 Benjamin Worpitz
//!
//! This file is part of matmul.
//!
//! matmul is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Lesser General Public License as published by
//! the Free Software Foundation, either version 3 of the License, or
//! (at your option) any later version.
//!
//! matmul is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//! GNU Lesser General Public License for more details.
//!
//! You should have received a copy of the GNU Lesser General Public License
//! along with matmul.
//! If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------

#pragma once

#if defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA_MEMCPY) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_THREADS) || defined(MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_FIBERS)

    #include "Mem.h"

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <alpaka/alpaka.hpp>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil
    #include <type_traits>          // std::is_same

    struct DummyKernel
    {
        template<
            typename TAcc>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc ) const
        -> void
        {
        }
    };

    template<
        typename T_Acc
    >
    struct OptimalVectorSize;

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_GPU_CUDA
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccGpuCudaRt<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<2u>;
    };
#endif

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_SEQ_T_OMP2
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Threads<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<16u>;
    };
#endif

#ifdef  MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_B_OMP2_T_SEQ
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Blocks<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<16u>;
    };
#endif

#ifdef MATMUL_BUILD_PAR_ALPAKA_ACC_CPU_BT_OMP4
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp4<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<16u>;
    };
#endif

    template<
        typename T_Size
    >
    struct ElementMatMul
    {
        template<
            typename MatA,
            typename MatB,
            typename MatC
        >
        ALPAKA_FN_ACC
        void
        operator()(
            MatA const & matA,
            MatB const & matB,
            MatC & matC
        ) const
        {
            using Vec2 = typename MatA::IndexType;

            constexpr auto numElements = T_Size::value;
            for( TSize i(0); i < numElements; ++i )
                for( TSize k(0); k < numElements; ++k )
                {
                    auto const a = matA[Vec2(i,k)];
                    for( TSize j(0); j < numElements; ++j )
                    {
                            matC[Vec2(i,j)] += a * matB[Vec2(k,j)];
                    }
                }
        }
    };


    class GemmAlpakaElementsKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem,
            typename MatA,
            typename MatB,
            typename MatC>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            MatA const & matA, TSize const & lda,
            MatB const & matB, TSize const & ldb,
            TElem const & beta,
            MatC matC, TSize const & ldc) const
        -> void
        {
            using Dim2 = alpaka::dim::DimInt<2u>;
            using Vec2 = alpaka::Vec<Dim2, TSize>;

            using VecSize = typename OptimalVectorSize<TAcc>::type;

            using Matrix = mem::Matrix<
                mem2::ConstPtrValue<TElem>,
                Vec2
            >;

            auto const numBlocks(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));
            auto const numThreads(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));

            auto const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));

           // std::cout<<"blockIdx="<<gridBlockIdx<<std::endl;

            constexpr auto numWorkElemsPerDim = VecSize::value;

            Vec2 const workSize(
                numThreads[ 0 ] * numWorkElemsPerDim,
                numThreads[ 1 ] * numWorkElemsPerDim
            );

            //Shared memory used to store the current blocks of A and B.
            TElem * const sharedBasePointer(acc.template getBlockSharedExternMem<TElem>());
            TElem * const sharedBasePointerB(sharedBasePointer + workSize[0] * workSize[1]);
            Matrix sharedMatA(
                sharedBasePointer,
                workSize
            );

            Matrix sharedMatB(
                sharedBasePointerB,
                workSize
            );

            using MVecNN = mem::MathVec<
                TElem,
                VecSize
            >;

            MVecNN matDot;

            for( size_t j = 0; j < VecSize::value; ++j )
                for( size_t i = 0; i < VecSize::value; ++i ){
                    matDot[ Vec2(j,i) ] = 0;
                }

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const nBlocks(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(
                            workSize[1]
                        )
                    )
                )
            );

            TSize const currentThreadInA_y( blockThreadIdx[ 0 ] * numWorkElemsPerDim);
            TSize const currentThreadInB_x( blockThreadIdx[ 1 ] * numWorkElemsPerDim);
            // needs architecture based mapping
            TSize const offsetInA_y(
                gridBlockIdx[ 0 ] * workSize[ 0 ]

            );
            TSize const offsetInB_x(
                gridBlockIdx[ 1 ] * workSize[ 1 ]

            );

            for(TSize blockA_x = 0; blockA_x < nBlocks; ++blockA_x)
            {
                //TSize const offsetA_x = (blockA_x + gridBlockIdx[1])%numBlocks[1] * workSize[ 1 ];
                TSize const offsetA_x = blockA_x * workSize[ 1 ];
                Vec2 const globalBlockOffsetInA(
                    offsetInA_y,
                    offsetA_x
                );
                Vec2 const globalBlockOffsetInB(
                    offsetA_x,
                    offsetInB_x
                );
                //load shared A
                for( TSize i(0); i < numWorkElemsPerDim; ++i )
                {
                    for( TSize j(0); j < numWorkElemsPerDim; ++j )
                    {
                        Vec2 const offsetInTile(
                            currentThreadInA_y + i,
                            currentThreadInB_x + j
                        );
                        Vec2 const globalIdxA(offsetInTile + globalBlockOffsetInA);
                        Vec2 const globalIdxB(offsetInTile + globalBlockOffsetInB);

                        auto const isValidA = (globalIdxA[0]<matA.m_extent[0]) && (globalIdxA[1]<k);
                        //printf("%i %i %i %i %i\n",globalIdxA[0],matA.m_extent[0],globalIdxA[1],matA.m_extent[1],isValidA);
                        auto const isValidB = (globalIdxB[0]<matB.m_extent[0]) && (globalIdxB[1]<n);

                        sharedMatA[ offsetInTile ] = isValidA ? matA[ globalIdxA ] : static_cast<TElem>(0);
                        sharedMatB[ offsetInTile ] = isValidB ? matB[ globalIdxB ] : static_cast<TElem>(0);

                    }
                }


                alpaka::block::sync::syncBlockThreads(acc);


                // move over line in A workSize
                for( TSize k3 = 0; k3 < workSize[ 0 ]; k3 +=numWorkElemsPerDim )
                {

                    Vec2 const globalIdx_A(
                        currentThreadInA_y,
                        k3
                    );
                    Vec2 const globalIdx_B(
                        k3,
                        currentThreadInB_x
                    );

                    Matrix const tmpA(
                        sharedMatA.view(
                            Vec2(
                                globalIdx_A[ 0 ],
                                globalIdx_A[ 1 ]
                            )
                        )
                    );
                    Matrix const tmpB(
                        sharedMatB.view(
                            Vec2(
                                globalIdx_B[ 0 ],
                                globalIdx_B[ 1 ]
                            )
                        )
                    );

                    ElementMatMul<VecSize> const elemMatMul;

                    elemMatMul(tmpA,tmpB,matDot);
                }

                alpaka::block::sync::syncBlockThreads(acc);

            }

            /*for( int i(0); i<2; ++i )
                for( int j(0); j<2; ++j )
                    std::cout<<i<<","<<j<<" "<<matDot[i][j]<<std::endl;
             * */
            for( TSize i(0); i < numWorkElemsPerDim; ++i )
            {
                for( TSize j(0); j < numWorkElemsPerDim; ++j )
                {
                    Vec2 const offsetC(
                        offsetInA_y + currentThreadInA_y + i,
                        offsetInB_x + currentThreadInB_x + j
                    );
                    auto const isValid = (offsetC[0] < matC.m_extent[0]) && (offsetC[1] <  n);
                  //  printf("%i %i %i %i %i\n",offsetC[0],matC.m_extent[0],offsetC[1],matC.m_extent[1],isValid);
                    if(isValid)
                        matC[ offsetC ] = alpha * matDot[ Vec2( i, j ) ] + beta * matC[ offsetC ];

                }
            }

        }
    };

    class GemmAlpakaOmpNative
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const i(gridThreadIdx[0u]);
            /*TSize const gThreadIdX(gridThreadIdx[1u]);
            TSize const i = gThreadIdY * alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[1] + gThreadIdX;
             * */

            if(i>=n) return;

            for(TSize j = 0; j < n; ++j)
            {
                C[i*ldc + j] *= beta;
            }
            for(TSize k2 = 0; k2 < k; ++k2)
            {
                TElem const a = alpha * A[i*lda + k2];

                for(TSize j = 0; j < n; ++j)
                {
                    C[i*ldc + j] += a * B[k2*ldb + j];
                }
            }
        }
    };
    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaElementsKernel2
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const & gridThreadIdxX(gridThreadIdx[1u]);
            TSize const & gridThreadIdxY(gridThreadIdx[0u]);

            // Number of blocks
            auto const gridBlockExtents(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));


            auto const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
            TSize const & gridBlockIdxX(gridBlockIdx[1u]);
            TSize const & gridBlockIdxY(gridBlockIdx[0u]);


            auto const gridElemsExtents(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc));
            TSize const & gridElemsExtentX(gridElemsExtents[1u]);
            TSize const & gridElemsExtentY(gridElemsExtents[0u]);

            // std::cout << "##########################################################################" << gridBlockExtents.prod() << std::endl;
            // std::cout << "BlockX: " << gridBlockIdxX << " BlockY: " << gridBlockIdxY << " nBlocks: " << gridBlockExtents.prod() << std::endl;

            // Column and row inside the block of C to calculate.
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadIdxX(blockThreadIdx[1u]);
            TSize const & blockThreadIdxY(blockThreadIdx[0u]);

            // The block threads extents.
            auto const blockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadsExtentX(blockThreadsExtents[1u]);
            TSize const & blockThreadsExtentY(blockThreadsExtents[0u]);
            //assert(blockThreadsExtentX == blockThreadsExtentY);


            // The block threads extents.
            auto const blockElemsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Elems>(acc));
            TSize const & blockElemsExtentX(blockElemsExtents[1u]);
            TSize const & blockElemsExtentY(blockElemsExtents[0u]);
            //assert(blockElemsExtentX == blockElemsExtentY);


            // Number of elements per thread
            auto const threadElemsExtents(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
            TSize const & threadElemsExtentX(threadElemsExtents[1u]);
            TSize const & threadElemsExtentY(threadElemsExtents[0u]);
            //assert(threadElemsExtentX == threadElemsExtentY);

            // Map the two dimensional index space into one dimension
            auto linearizedBlockThreadIdx(alpaka::core::mapIdx<1u>(blockThreadIdx, blockThreadsExtents)[0]);


            //Shared memory used to store the current blocks of A and B.
            TElem * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadsExtents.prod() * threadElemsExtents.prod());

            //TSize const sharedBlockIdx1d(linearizedBlockThreadIdx * threadElemsExtents.prod());

            // If the element corresponding to the current thread is outside of the respective matrix.
            // Can not be calculated on this place
            // bool const insideA(gridThreadIdxY * threadElemsExtentY < blockThreadsExtentX);
            // bool const insideB(gridThreadIdxX * threadElemsExtentX < blockThreadsExtentY);
            bool const insideA(true);
            bool const insideB(true);
            bool const insideC(insideA && insideB);

            TElem dotProduct[threadElemsExtents.prod()];

            for(size_t i = 0; i < threadElemsExtents.prod(); ++i){
                dotProduct[i] = 0;
            }

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const nBlocks(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtentX * threadElemsExtentX))));

            for(TSize block_i = 0; block_i < nBlocks; ++block_i)
            {

                //std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX = ( block_i * blockElemsExtentY ) + ( blockThreadIdxX * threadElemsExtentX );
                TSize const BIdxY = ( block_i * blockElemsExtentY ) + ( blockThreadIdxY * threadElemsExtentY );

                //std::cout << "sharedBlockIdx1d: " << sharedBlockIdx1d << " AIdx1d: " << AIdx1d << " lda: " << lda << std::endl;

                for(size_t row_i = 0; row_i < threadElemsExtentX; ++row_i)
                {
                    for(size_t col_j = 0; col_j < threadElemsExtentY; ++col_j)
                    {
                        TSize const AIdx1d = ( gridThreadIdxY * threadElemsExtentY * lda ) // global row
                            + ( row_i * lda ) // elements row
                            + ( AIdxX ) // global column
                            + ( col_j ); // elements column

                        TSize const sharedBlockIdx1d = ( blockThreadIdxY * blockElemsExtentX )
                            + ( row_i * blockElemsExtentX)
                            + ( blockThreadIdxX * threadElemsExtentX )
                            + ( col_j );

                        pBlockSharedA[sharedBlockIdx1d] = A[AIdx1d];
                        //std::cout << "row_i: " << row_i << " col_j: " << col_j << " AIdxX: " << AIdxX <<" sharedBlockIdx1d: " << sharedBlockIdx1d << " written value: " << pBlockSharedA[sharedBlockIdx1d] << std::endl;
                    }

                }

                for(size_t row_i = 0; row_i < threadElemsExtentX; ++row_i)
                {
                        for(size_t col_j = 0; col_j < threadElemsExtentY; ++col_j)
                        {

                            TSize const BIdx1d = (BIdxY * ldb) // global row
                                + ( row_i * ldb ) // elements row
                                + ( gridThreadIdxX * threadElemsExtentX )
                                + ( col_j );

                            TSize const sharedBlockIdx1d = ( blockThreadIdxY * blockElemsExtentX )
                                + ( row_i * blockElemsExtentX)
                                + ( blockThreadIdxX * threadElemsExtentX )
                                + ( col_j );

                                pBlockSharedB[sharedBlockIdx1d] =  B[BIdx1d];
                                //std::cout << "row_i: " << row_i << " col_j: " << col_j << " BIdxX: " << AIdxX <<" sharedBlockIdx1d: " << sharedBlockIdx1d << " written value: " << pBlockSharedB[sharedBlockIdx1d] << std::endl;

                        }

                }


                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                alpaka::block::sync::syncBlockThreads(acc);

                // Compute the dot products within shared memory.
                for(size_t row_i = 0; row_i < threadElemsExtentX; ++row_i)
                {
                    for(size_t col_j = 0; col_j < threadElemsExtentY; ++col_j)
                    {
                        for(TSize k3 = 0; k3 < blockElemsExtentX; ++k3)
                        {

                            TSize const blockAidx = ( blockThreadIdxY * blockThreadsExtentX ) // block row
                                + ( row_i * blockElemsExtentX )
                                + ( k3 );

                            TSize const blockBidx = ( k3 * blockThreadsExtentY * threadElemsExtentY )
                                + ( blockThreadIdxX * threadElemsExtentX )
                                + ( col_j );

                            // std::cout << " i:  " << row_i
                            //           << " j:  " << col_j
                            //           << " k3: " << k3
                            //           << " A:  " << pBlockSharedA[blockAidx]
                            //           << " B:  " << pBlockSharedB[blockBidx] << std::endl;

                            dotProduct[row_i * threadElemsExtentX + col_j] +=  pBlockSharedA[blockAidx] * pBlockSharedB[blockBidx];

                        }

                    }

                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                alpaka::block::sync::syncBlockThreads(acc);
            }

            for(size_t row_i = 0; row_i < threadElemsExtentX; ++row_i)
            {
                for(size_t col_j = 0; col_j < threadElemsExtentX; ++col_j)
                {
                    TSize const CIdx1d = ( gridThreadIdxY *  threadElemsExtentY * ldc ) // global row
                        + ( row_i * ldc ) // elements row
                        + ( gridThreadIdxX * threadElemsExtentX ) // global col
                        + ( col_j); // elements col

                    // std::cout << " row_i: " << row_i
                    //           << " col_j: " << col_j
                    //           << " dotProduct: " << dotProduct[row_i * threadElemsExtentX + col_j]
                    //           << " CIdx1d: " << CIdx1d
                    //           << std::endl;

                    C[CIdx1d] = alpha * dotProduct[row_i * threadElemsExtentX + col_j] + beta * C[CIdx1d];
                }

            }

        }

    };

    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaSharedKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const & gridThreadIdxX(gridThreadIdx[1u]);
            TSize const & gridThreadIdxY(gridThreadIdx[0u]);

            // Column and row inside the block of C to calculate.
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadIdxX(blockThreadIdx[1u]);
            TSize const & blockThreadIdxY(blockThreadIdx[0u]);

            // The block threads extents.
            auto const blockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadsExtentX(blockThreadsExtents[1u]);
            TSize const & blockThreadsExtentY(blockThreadsExtents[0u]);
            //assert(blockThreadsExtentX == blockThreadsExtentY);
            TSize const & blockThreadsExtent(blockThreadsExtentX);

            // Shared memory used to store the current blocks of A and B.
            TElem * const pBlockSharedA(acc.template getBlockSharedExternMem<TElem>());
            TElem * const pBlockSharedB(pBlockSharedA + blockThreadsExtentX*blockThreadsExtentY);

            TSize const sharedBlockIdx1d(blockThreadIdxY*blockThreadsExtentX + blockThreadIdxX);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA(gridThreadIdxY < m);
            bool const insideB(gridThreadIdxX < n);
            bool const insideC(insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtent))));
            for(TSize k2(0); k2<blockMulCount; ++k2)
            {
                // Copy the current blocks of A and B into shared memory in parallel.
                // If the element of the current thread is outside of the matrix, zero is written into the shared memory.
                // This is possible because zero is a result neutral extension of the matrices regarding the dot product.
                TSize const AIdxX(k2*blockThreadsExtentX + blockThreadIdxX);
                TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                pBlockSharedA[sharedBlockIdx1d] =
                    ((!insideA) || (AIdxX>=k))
                    ? static_cast<TElem>(0)
                    : A[AIdx1d];

                TSize const BIdxY(k2*blockThreadsExtentY + blockThreadIdxY);
                TSize const BIdx1d(BIdxY*ldb + gridThreadIdxX);
                pBlockSharedB[sharedBlockIdx1d] =
                    ((!insideB) || (BIdxY>=k))
                    ? static_cast<TElem>(0)
                    : B[BIdx1d];

                // Synchronize to make sure the complete blocks are loaded before starting the computation.
                alpaka::block::sync::syncBlockThreads(acc);

                // Compute the dot products within shared memory.
                for(TSize k3(0); k3<blockThreadsExtent; ++k3)
                {
                    dotProduct += pBlockSharedA[blockThreadIdxY*blockThreadsExtentX + k3]
                        * pBlockSharedB[k3*blockThreadsExtentY + blockThreadIdxX];
                }

                // Synchronize to make sure that the preceding computation is done before loading the next blocks of A and B.
                alpaka::block::sync::syncBlockThreads(acc);
            }

            // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results.
            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
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
                    GemmAlpakaOmpNative,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem,
                        typename MatA,
                        typename MatB,
                        typename MatC
                    >
                    ALPAKA_FN_HOST static auto getBlockSharedExternMemSizeBytes(
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & vblockThreadsExtents,
                        TSize const & m,
                        TSize const & n,
                        TSize const & k,
                        TElem const & alpha,
                        MatA const & matA,
                        TSize const & lda,
                        MatB const & matB,
                        TSize const & ldb,
                        TElem const & beta,
                        MatC matC,
                        TSize const & ldc)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TSize, size::Size<TAcc>>::value,
                            "TSize and size::Size<TAcc> have to be identical!");

                        boost::ignore_unused(m);
                        boost::ignore_unused(n);
                        boost::ignore_unused(k);
                        boost::ignore_unused(alpha);
                        boost::ignore_unused(matA);
                        boost::ignore_unused(lda);
                        boost::ignore_unused(matB);
                        boost::ignore_unused(ldb);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(matC);
                        boost::ignore_unused(ldc);

                        // Reserve the buffer for the two blocks of A and B.
                        return 0; //2u * vblockThreadsExtents.prod() * sizeof(TElem);
                    }
                };

                //#############################################################################
                //! The trait for getting the size of the block shared extern memory for a kernel.
                //#############################################################################
                template<
                    typename TAcc>
                struct BlockSharedExternMemSizeBytes<
                    GemmAlpakaSharedKernel,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TElem>
                    ALPAKA_FN_HOST static auto getBlockSharedExternMemSizeBytes(
                        alpaka::Vec<alpaka::dim::Dim<TAcc>, size::Size<TAcc>> const & vblockThreadsExtents,
                        TSize const & m,
                        TSize const & n,
                        TSize const & k,
                        TElem const & alpha,
                        TElem const * const A,
                        TSize const & lda,
                        TElem const * const B,
                        TSize const & ldb,
                        TElem const & beta,
                        TElem * const C,
                        TSize const & ldc)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TSize, size::Size<TAcc>>::value,
                            "TSize and size::Size<TAcc> have to be identical!");

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
                        return 2u * vblockThreadsExtents.prod() * sizeof(TElem);
                    }
                };
            }
        }
    }

    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaNoSharedKernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const & gridThreadIdxX(gridThreadIdx[1u]);
            TSize const & gridThreadIdxY(gridThreadIdx[0u]);

            // The block threads extents.
            auto const blockThreadsExtents(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));
            TSize const & blockThreadsExtentX(blockThreadsExtents[1u]);
            TSize const & blockThreadsExtentY(blockThreadsExtents[0u]);
            TSize const & blockThreadsExtent(blockThreadsExtentX);
            //assert(blockThreadsExtentX == blockThreadsExtentY);


            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA(gridThreadIdxY < m);
            bool const insideB(gridThreadIdxX < n);
            bool const insideC(insideA && insideB);

            TElem dotProduct(0);

            // Loop over all blocks of A and B that are required to compute the C block.
            TSize const blockMulCount(
                static_cast<TSize>(
                    alpaka::math::ceil(
                        acc,
                        static_cast<float>(k)/static_cast<float>(blockThreadsExtent))));
            for(TSize k2(0); k2<blockMulCount; ++k2)
            {
                TSize const ABlockIdxX(k2*blockThreadsExtentX);
                TSize const BBlockIdxY(k2*blockThreadsExtentY);

                // Compute the dot products.
                for(TSize k3(0); k3<blockThreadsExtent; ++k3)
                {
                    TSize const AIdxX(ABlockIdxX + k3);
                    TSize const AIdx1d(gridThreadIdxY*lda + AIdxX);
                    TSize const BIdxY(BBlockIdxY + k3);
                    TSize const BIdx1d(gridThreadIdxX + BIdxY*ldb);

                    TElem const a(
                        ((!insideA) || (AIdxX>=k))
                        ? static_cast<TElem>(0)
                        : A[AIdx1d]);
                    TElem const b(
                        ((!insideB) || (BIdxY>=k))
                        ? static_cast<TElem>(0)
                        : B[BIdx1d]);
                    dotProduct += a * b;
                }
            }

            // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results.
            if(insideC)
            {
                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
    };

    //#############################################################################
    // This function only works for square blocks.
    //#############################################################################
    class GemmAlpakaNoShared2Kernel
    {
    public:
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TAcc,
            typename TElem>
        ALPAKA_FN_ACC auto operator()(
            TAcc const & acc,
            TSize const & m, TSize const & n, TSize const & k,
            TElem const & alpha,
            TElem const * const MATMUL_RESTRICT A, TSize const & lda,
            TElem const * const MATMUL_RESTRICT B, TSize const & ldb,
            TElem const & beta,
            TElem * const MATMUL_RESTRICT C, TSize const & ldc) const
        -> void
        {
            static_assert(alpaka::dim::Dim<TAcc>::value == 2u,
                "The accelerator used for the GemmAlpakaKernel has to be 2 dimensional!");

            // Column and row of C to calculate.
            auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
            TSize const & gridThreadIdxX(gridThreadIdx[1u]);
            TSize const & gridThreadIdxY(gridThreadIdx[0u]);

            // If the element corresponding to the current thread is outside of the respective matrix.
            bool const insideA(gridThreadIdxY < m);
            bool const insideB(gridThreadIdxX < n);
            bool const insideC(insideA && insideB);

            // If the element is outside of the matrix it was only a helper thread that did not calculate any meaningful results.
            if(insideC)
            {
                TElem dotProduct(0);

                // Compute the dot products.
                for(TSize k3(0); k3<k; ++k3)
                {
                    TSize const AIdx1d(gridThreadIdxY*lda + k3);
                    TSize const BIdx1d(gridThreadIdxX + k3*ldb);
                    dotProduct += A[AIdx1d] * B[BIdx1d];
                }

                TSize const CIdx1d(gridThreadIdxY*ldc + gridThreadIdxX);
                C[CIdx1d] = alpha * dotProduct + beta * C[CIdx1d];
            }
        }
    };

    namespace detail
    {
        //#############################################################################
        //! The stream type trait for the stream that should be used for the given device.
        //#############################################################################
        template<
            typename TDev,
            typename TSfinae = void>
        struct StreamType;

        //#############################################################################
        //! The stream type trait specialization for the CPU device.
        //#############################################################################
        template<>
        struct StreamType<
            alpaka::dev::DevCpu>
        {
//#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCpuSync;
/*#else
            using type = alpaka::stream::StreamCpuAsync;
#endif*/
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
        //#############################################################################
        //! The stream type trait specialization for the CUDA device.
        //#############################################################################
        template<>
        struct StreamType<
            alpaka::dev::DevCudaRt>
        {
#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCudaRtSync;
#else
            using type = alpaka::stream::StreamCudaRtAsync;
#endif
        };
#endif
    }
    //#############################################################################
    //! The stream type that should be used for the given device.
    //#############################################################################
    template<
        typename TDev>
    using Stream = typename detail::StreamType<TDev>::type;

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TKernelFnObj>
    TReturn matmul_gemm_par_alpaka(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        using Dim1 = alpaka::dim::DimInt<1u>;
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Dim3 = alpaka::dim::DimInt<3u>;

        using Vec2 = alpaka::Vec<Dim2, TSize>;


        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec<Dim1, TSize> const v2uiExtentsC(
            m
        );

        alpaka::Vec<Dim1, TSize> const elemExtent(
            static_cast<TSize>(1)
        );
        /*
        alpaka::Vec<Dim2, TSize> const elemExtent(
            static_cast<TSize>(1),
            static_cast<TSize>(1));
         */
        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim1, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

   /*     using Matrix = mem::Matrix<
            mem2::ConstPtrValue<TElem>,
            Vec2
        >;

        using ConstMatrix = mem::Matrix<
            mem2::ConstPtrConstValue<TElem>,
            Vec2
        >;

        ConstMatrix const matA(
            A,
            Vec2(
                m,
                lda
            )
        );

        ConstMatrix const matB(
            B,
            Vec2(
                k,
                ldb
            )
        );
        Matrix matC(
            C,
            Vec2(
                m,
                ldc
            )
        );*/

        auto const execDummy(
            alpaka::exec::create<TAcc>(
                workDiv,
                DummyKernel()
            )
        );
        alpaka::stream::enqueue(stream, execDummy);
        alpaka::wait::wait(stream);

        // Create the executor.
        // NOTE: We remove the __restrict__ because alpaka calls std::ref on the arguments and std::ref errors.
        // This is most probably undefined. MSVC compiles it without any warning.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            reinterpret_cast<TElem const *>(A),
            lda,
            reinterpret_cast<TElem const *>(B),
            ldb,
            beta,
            reinterpret_cast<TElem *>(C),
            ldc));

        MATMUL_TIME_START;

        // Execute the kernel.
        alpaka::stream::enqueue(stream, exec);

        // Wait for the stream to finish the operations.
        alpaka::wait::wait(stream);

        MATMUL_TIME_END;
        MATMUL_TIME_RETURN;
    }

    //-----------------------------------------------------------------------------
    //
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TKernelFnObj>
    TReturn matmul_gemm_par_alpaka_memcpy(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {

        using Dim1 = alpaka::dim::DimInt<1u>;
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Dim3 = alpaka::dim::DimInt<3u>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Get the host device.
        auto devHost(alpaka::dev::DevManCpu::getDevByIdx(0u));

        // Select a device to execute on.
        alpaka::dev::Dev<TAcc> devAcc(
            alpaka::dev::DevMan<TAcc>::getDevByIdx(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        alpaka::Vec<Dim2, TSize> const v2uiExtentsA(
            m,
            k);

        alpaka::Vec<Dim2, TSize> const v2uiExtentsB(
            k,
            n);

        // Result matrix is MxN. We create one worker per result matrix cell.
        alpaka::Vec<Dim2, TSize> const v2uiExtentsC(
            m,
            n);

        alpaka::Vec<Dim1, TSize> const v2uiExtentsC2(
            m
        );

        alpaka::Vec<Dim1, TSize> const elemExtent(
            static_cast<TSize>(1)
        );

        // Wrap the Pointers into memory buffer objects.
        using BufWrapperIn = alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            TElem const,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapperIn bufAHost(A, devHost, v2uiExtentsA, lda * sizeof(TElem));
        BufWrapperIn bufBHost(B, devHost, v2uiExtentsB, ldb * sizeof(TElem));
        using BufWrapperOut = alpaka::mem::view::ViewPlainPtr<
            std::decay<decltype(devHost)>::type,
            TElem,
            alpaka::dim::DimInt<2u>,
            TSize>;
        BufWrapperOut bufCHost(C, devHost, v2uiExtentsC, ldc * sizeof(TElem));

        // Allocate the buffers on the accelerator and copy Host -> Acc.
        // TODO: Test if interleaved is better then alloc first, copy later.
        // Because alloc causes a device sync this may hinder the copies.
        auto bufAAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsA));
        alpaka::mem::view::copy(stream, bufAAcc, bufAHost, v2uiExtentsA);
        auto bufBAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsB));
        alpaka::mem::view::copy(stream, bufBAcc, bufBHost, v2uiExtentsB);
        auto bufCAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsC));
        alpaka::mem::view::copy(stream, bufCAcc, bufCHost, v2uiExtentsC);

        alpaka::Vec<Dim1, TSize> const M(m);
        // Let alpaka calculate good block and grid sizes given our full problem extents.
        alpaka::workdiv::WorkDivMembers<Dim1, TSize> const workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC2,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;
        // Create the executor.
        // NOTE: We remove the __restrict__ because alpaka calls std::ref on the arguments and std::ref errors.
        // This is most probably undefined. MSVC compiles it without any warning.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            reinterpret_cast<TElem const *>(alpaka::mem::view::getPtrNative(bufAAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufAAcc)/sizeof(TElem),
            reinterpret_cast<TElem const *>(alpaka::mem::view::getPtrNative(bufBAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufBAcc)/sizeof(TElem),
            beta,
            reinterpret_cast<TElem *>(alpaka::mem::view::getPtrNative(bufCAcc)),
            alpaka::mem::view::getPitchBytes<1>(bufCAcc)/sizeof(TElem)));

        alpaka::wait::wait(stream);
        MATMUL_TIME_START;

        // Execute the kernel.
        alpaka::stream::enqueue(stream, exec);

#ifdef MATMUL_RETURN_COMPUTATION_TIME
        alpaka::wait::wait(stream);
#endif
        MATMUL_TIME_END;

        // Copy back the result.
        alpaka::mem::view::copy(stream, bufCHost, bufCAcc, v2uiExtentsC);

        // Wait for the stream to finish the operations.
        alpaka::wait::wait(stream);

        MATMUL_TIME_RETURN;
    }
#endif
