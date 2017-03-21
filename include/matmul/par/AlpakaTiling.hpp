//-----------------------------------------------------------------------------
//! \file
//! Copyright 2013-2016 Benjamin Worpitz, Rene Widera
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

    #include "matmul/common/AlpakaHelper.hpp"

    #include <matmul/common/Mat.h>  // matmul_mat_gemm_early_out

    #include <alpaka/alpaka.hpp>

    #include <stdio.h>              // printf
    #include <math.h>               // ceil
    #include <type_traits>          // std::is_same

    template<
        typename T_Acc
    >
    struct OptimalVectorSize
    {
        using type = alpaka::dim::DimInt<1u>;
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccGpuCudaRt<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<GPU_ELEM_NUM>;
    };
#endif

#ifndef OMP_ELEM_NUM
    #define OMP_ELEM_NUM 256u
#endif

#ifndef OMP_DIVISOR
    #define OMP_DIVISOR 1u
#endif

#ifndef REAL_SHARED_MEMORY
    #define REAL_SHARED_MEMORY 1
#endif

#define VECTOR_PRAGMA \
    /*_Pragma ("vector aligned")*/ \
    /*_Pragma ("unroll (8)")*/ \
    _Pragma ("ivdep") \
    _Pragma ("GCC ivdep")



#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Threads<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<OMP_ELEM_NUM>;
    };
#endif

#ifdef  ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp2Blocks<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<OMP_ELEM_NUM>;
    };
#endif

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED
    template<
        typename... T_Args
    >
    struct OptimalVectorSize<
        alpaka::acc::AccCpuOmp4<
            T_Args...
        >
    >
    {
        using type = alpaka::dim::DimInt<OMP_ELEM_NUM>;
    };
#endif

    //#############################################################################
    //!
    //#############################################################################
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
            MatC* matC,
            size_t const jumpLength,
            size_t const results
        ) const
        {
            using Vec2 = typename MatA::IndexType;

            VECTOR_PRAGMA
            for( TSize j(0); j < results; ++j )
            {
                    //matC[Vec2(i,j)] += a * matB[Vec2(k,j)];
                    matC[j] += matB * matA[Vec2(j*jumpLength,0)];
            }
        }
    };

    //#############################################################################
    //! An alpaka kernel implementing an adaptive tiling scheme.
    //#############################################################################
    class GemmAlpakaTiling
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
            MatA const & MATMUL_RESTRICT matA,
            MatB const & MATMUL_RESTRICT matB,
            TElem const & beta,
            MatC MATMUL_RESTRICT matC) const
        -> void
        {
            using Dim2 = alpaka::dim::DimInt<2u>;
            using Vec2 = alpaka::vec::Vec<Dim2, TSize>;

            using VecSize = typename OptimalVectorSize<TAcc>::type;

            using Matrix = alpakaHelper::Matrix<
                alpakaHelper2::ConstPtrValue<TElem>,
                Vec2
            >;
            using ConstMatrix = alpakaHelper::Matrix<
                alpakaHelper2::ConstPtrConstValue<TElem>,
                Vec2
            >;

            auto const numBlocks(alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc));
            auto const numThreads(alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc));

            auto const gridBlockIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));
            auto const blockThreadIdx(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc));

            constexpr auto numWorkElemsPerDim = static_cast<TSize>(VecSize::value);

            Vec2 const workSize(
                numThreads[ 0 ] * numWorkElemsPerDim,
                numThreads[ 1 ] * numWorkElemsPerDim
            );

            size_t const linearThreadIdx = alpaka::idx::mapIdx<1u>(
                blockThreadIdx,
                numThreads
            )[0];

            size_t const linerThreadCount = numThreads[0] * numThreads[1];

            size_t const linearWorkSize = workSize[0] * workSize[1];

            Vec2 threadIndex(
                linearThreadIdx / workSize[0],
                linearThreadIdx % workSize[1]
            );

            size_t const jumpLength = linerThreadCount / workSize[1];

            //Shared alpakaHelperory used to store the current blocks of A and B.
            TElem * const sharedBasePointer(alpaka::block::shared::dyn::getMem<TElem>(acc));
            TElem * const sharedBasePointerB(sharedBasePointer + workSize[0] * workSize[1]);
#if REAL_SHARED_MEMORY == 1
            Matrix sharedMatA(
                sharedBasePointer,
                workSize
            );
            Matrix sharedMatB(
                sharedBasePointerB,
                workSize
            );
#endif

            using MVecNN = alpakaHelper::MathVec<
                TElem,
                VecSize
            >;

            TElem matDot[numWorkElemsPerDim*numWorkElemsPerDim];
            VECTOR_PRAGMA
            for(TSize j(0); j < numWorkElemsPerDim*numWorkElemsPerDim; ++j)
            {
                matDot[j] = 0;
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

            VECTOR_PRAGMA
            for(TSize blockA_x = 0; blockA_x < nBlocks; ++blockA_x)
            {
                TSize const offsetA_x = blockA_x * workSize[ 1 ];
                Vec2 const globalBlockOffsetInA(
                    offsetInA_y,
                    offsetA_x
                );
                Vec2 const globalBlockOffsetInB(
                    offsetA_x,
                    offsetInB_x
                );
#if REAL_SHARED_MEMORY == 1
                //load shared A & B
                VECTOR_PRAGMA
                for( TSize i(0); i < numWorkElemsPerDim * numWorkElemsPerDim; ++i )
                {
                    auto offsetInTile_Y = threadIndex[0] + i * jumpLength;
                    auto lineSharedA = &(sharedMatA.m_ptr[offsetInTile_Y * sharedMatA.m_extent[1]]);
                    auto lineSharedB = &(sharedMatB.m_ptr[offsetInTile_Y * sharedMatB.m_extent[1]]);
                    auto lineMatA = &(matA.m_ptr[(offsetInTile_Y + offsetInA_y) * matA.m_extent[1]]);
                    auto lineMatB = &(matB.m_ptr[(offsetInTile_Y + offsetA_x) * matB.m_extent[1]]);
                    Vec2 const offsetInTile(
                        offsetInTile_Y,
                        threadIndex[1]
                    );
                    Vec2 const globalIdxA(offsetInTile + globalBlockOffsetInA);
                    Vec2 const globalIdxB(offsetInTile + globalBlockOffsetInB);

                    auto const isValidA = (globalIdxA[0]<matA.m_extent[0]) && (globalIdxA[1]<k);
                    auto const isValidB = (globalIdxB[0]<matB.m_extent[0]) && (globalIdxB[1]<n);

                    //sharedMatA[ offsetInTile ] = isValidA ? matA[ globalIdxA ] : static_cast<TElem>(0);
                    //sharedMatB[ offsetInTile ] = isValidB ? matB[ globalIdxB ] : static_cast<TElem>(0);
                    lineSharedA[offsetInTile[1]] = isValidA ? lineMatA[ globalIdxA[1] ] : static_cast<TElem>(0);
                    lineSharedB[offsetInTile[1]] = isValidB ? lineMatB[ globalIdxB[1] ] : static_cast<TElem>(0);

                }
                alpaka::block::sync::syncBlockThreads(acc);
#else
                //create view of A & B
                ConstMatrix sharedMatA(
                    matA.view(globalBlockOffsetInA)
                );
                ConstMatrix sharedMatB(
                    matB.view(globalBlockOffsetInB)
                );
#endif
                // move over line in A workSize
                // move over line in A workSize
                for(TSize k3(0); k3 < workSize[ 1 ]; ++k3)
                {

                    Vec2 const globalIdx_A(
                        threadIndex[0],
                        k3
                    );
                    Vec2 const globalIdx_B(
                        k3,
                        threadIndex[1]
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

                    elemMatMul(tmpA,tmpB[Vec2(size_t(0),size_t(0))],matDot,jumpLength,numWorkElemsPerDim*numWorkElemsPerDim);
                }
                alpaka::block::sync::syncBlockThreads(acc);

            }

                VECTOR_PRAGMA
                for(TSize j(0); j < numWorkElemsPerDim*numWorkElemsPerDim; ++j)
                {
                    Vec2 const offsetC(
                        offsetInA_y + threadIndex[0] + j * jumpLength,
                        offsetInB_x + threadIndex[1]
                    );
                    auto const isValid = (offsetC[0] < matC.m_extent[0]) && (offsetC[1] <  n);

                    if(isValid)
                        matC[ offsetC ] = alpha * matDot[ j ] + beta * matC[ offsetC ];

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
                struct BlockSharedMemDynSizeBytes<
                    GemmAlpakaTiling,
                    TAcc>
                {
                    //-----------------------------------------------------------------------------
                    //! \return The size of the shared memory allocated for a block.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TVec,
                        typename MatA,
                        typename MatB,
                        typename MatC
                    >
                    ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                        GemmAlpakaTiling const & gemmAlpakaTiling,
                        TVec const & blockThreadExtent,
                        TVec const & threadElemExtent,
                        TSize const & m,
                        TSize const & n,
                        TSize const & k,
                        TElem const & alpha,
                        MatA const & MATMUL_RESTRICT matA,
                        MatB const & MATMUL_RESTRICT matB,
                        TElem const & beta,
                        MatC matC)
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
                        boost::ignore_unused(matB);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(matC);

                        // Reserve the buffer for the two blocks of A and B.
#if REAL_SHARED_MEMORY == 1
                        return 2u * blockThreadExtent.prod() * threadElemExtent.prod() * sizeof(TElem);
#else
                        return 0;
#endif
                    }
                };

            }
        }
    }


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
#if (MATMUL_DEBUG >= MATMUL_DEBUG_FULL)
            using type = alpaka::stream::StreamCpuSync;
#else
            using type = alpaka::stream::StreamCpuAsync;
#endif
        };

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
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
    TReturn matmul_gemm_par_alpaka_tiling(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::vec::Vec<Dim2, TSize>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Select a device to execute on.
        auto devAcc(
            alpaka::pltf::getDevByIdx< alpaka::pltf::Pltf< alpaka::dev::Dev<TAcc> > >(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const v2uiExtentsC(
            m,
            n);

        Vec2 const elemExtent(
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value),
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value)
        );

        // Let alpaka calculate good block and grid sizes given our full problem extents.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        const alpaka::vec::Vec<Dim2, TSize> threads (TSize(16), TSize(16));
        const alpaka::vec::Vec<Dim2, TSize> blocks  (m/threads[0]/elemExtent[0], n/threads[1]/elemExtent[1]);
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(blocks,threads,elemExtent);
#else
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // We need to check, whether this workdiv is too big for the shared memory
        while ( 2u * workDiv.m_blockThreadExtent.prod() * workDiv.m_threadElemExtent.prod() * sizeof(TElem) > prop.sharedMemPerBlock) // 64KB or 112KB
        {
            workDiv.m_gridBlockExtent[0] *= 2;
            workDiv.m_gridBlockExtent[1] *= 2;
            workDiv.m_blockThreadExtent[0] /= 2;
            workDiv.m_blockThreadExtent[1] /= 2;
        }
#endif
        size_t used_shared_memory = 2u * workDiv.m_blockThreadExtent.prod() * workDiv.m_threadElemExtent.prod() * sizeof(TElem);
        std::cout << std::endl << "Workdiv: "
                  << workDiv.m_gridBlockExtent[0] << "*" << workDiv.m_gridBlockExtent[1] << " : "
                  << workDiv.m_blockThreadExtent[0] << "*" << workDiv.m_blockThreadExtent[1] << " : "
                  << workDiv.m_threadElemExtent[0] << "*" << workDiv.m_threadElemExtent[1]
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                  << " (Shared Memory: " << used_shared_memory << " of "
                  << prop.sharedMemPerBlock << ")" << std::endl;
#else
                  << std::endl;
#endif
        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        using Matrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrValue<TElem>,
            Vec2
        >;

        using ConstMatrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrConstValue<TElem>,
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
        );

        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            matA,
            matB,
            beta,
            matC));
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
    TReturn matmul_gemm_par_alpaka_memcpy_tiling(
        TSize const m, TSize const n, TSize const k,
        TElem const alpha,
        TElem const * const MATMUL_RESTRICT A, TSize const lda,
        TElem const * const MATMUL_RESTRICT B, TSize const ldb,
        TElem const beta,
        TElem * const MATMUL_RESTRICT C, TSize const ldc)
    {
        std::cout << "Using explicit memcpy" << std::endl;
        using Dim2 = alpaka::dim::DimInt<2u>;
        using Vec2 = alpaka::vec::Vec<Dim2, TSize>;

        if(matmul_mat_gemm_early_out(m, n, k, alpha, beta))
        {
            MATMUL_TIME_RETURN_EARLY_OUT;
        }

        // Get the host device.
        auto devHost(
            alpaka::pltf::getDevByIdx< alpaka::pltf::Pltf< alpaka::dev::DevCpu > >(0u));

        // Select a device to execute on.
        auto devAcc(
            alpaka::pltf::getDevByIdx< alpaka::pltf::Pltf< alpaka::dev::Dev<TAcc> > >(0));

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        Vec2 const v2uiExtentsA(
            m,
            k);

        Vec2 const v2uiExtentsB(
            k,
            n);

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const v2uiExtentsC(
            m,
            n);

        Vec2 const elemExtent(
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value),
            static_cast<TSize>(OptimalVectorSize<TAcc>::type::value)
        );


        // Wrap the Pointers into memory buffer objects.
        using DevHost = std::decay<decltype(devHost)>::type;
        using BufWrapperIn = alpaka::mem::view::ViewPlainPtr<
            DevHost,
            TElem const,
            Dim2,
            TSize>;
        constexpr TSize elemSize(static_cast<TSize>(sizeof(TElem)));
        TSize const pitchBytesXAHost = lda * elemSize;
        Vec2 const pitchBytesAHost(k * pitchBytesXAHost, pitchBytesXAHost);
        BufWrapperIn bufAHost(A, devHost, v2uiExtentsA, pitchBytesAHost);
        TSize const pitchBytesXBHost = ldb * elemSize;
        Vec2 const pitchBytesBHost(n * pitchBytesXBHost, pitchBytesXBHost);
        BufWrapperIn bufBHost(B, devHost, v2uiExtentsB, pitchBytesBHost);
        using BufWrapperOut = alpaka::mem::view::ViewPlainPtr<
            DevHost,
            TElem,
            Dim2,
            TSize>;
        TSize const pitchBytesXCHost = ldc * elemSize;
        Vec2 const pitchBytesCHost(n * pitchBytesXCHost, pitchBytesXCHost);
        BufWrapperOut bufCHost(C, devHost, v2uiExtentsC, pitchBytesCHost);

        // Allocate the buffers on the accelerator and copy Host -> Acc.
        // TODO: Test if interleaved is better then alloc first, copy later.
        // Because alloc causes a device sync this may hinder the copies.
        auto bufAAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsA));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < v2uiExtentsA.prod(); i++)
            alpaka::mem::view::getPtrNative(bufAAcc)[i] = 0;
#endif
        alpaka::mem::view::copy(stream, bufAAcc, bufAHost, v2uiExtentsA);
        auto bufBAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsB));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < v2uiExtentsB.prod(); i++)
            alpaka::mem::view::getPtrNative(bufBAcc)[i] = 0;
#endif
        alpaka::mem::view::copy(stream, bufBAcc, bufBHost, v2uiExtentsB);
        auto bufCAcc(alpaka::mem::buf::alloc<TElem, TSize>(devAcc, v2uiExtentsC));
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < v2uiExtentsC.prod(); i++)
            alpaka::mem::view::getPtrNative(bufCAcc)[i] = 0;
#endif
        alpaka::mem::view::copy(stream, bufCAcc, bufCHost, v2uiExtentsC);

        // Let alpaka calculate good block and grid sizes given our full problem extents.
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        const alpaka::vec::Vec<Dim2, TSize> threads (TSize(16), TSize(16));
        const alpaka::vec::Vec<Dim2, TSize> blocks  (m/threads[0]/elemExtent[0], n/threads[1]/elemExtent[1]);
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(blocks,threads,elemExtent);
#else
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(
            alpaka::workdiv::getValidWorkDiv<TAcc>(
                devAcc,
                v2uiExtentsC,
                elemExtent,
                false,
                alpaka::workdiv::GridBlockExtentSubDivRestrictions::EqualExtent));
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // We need to check, whether this workdiv is too big for the shared memory
        while ( 2u * workDiv.m_blockThreadExtent.prod() * workDiv.m_threadElemExtent.prod() * sizeof(TElem) > prop.sharedMemPerBlock) // 64KB or 112KB
        {
            workDiv.m_gridBlockExtent[0] *= 2;
            workDiv.m_gridBlockExtent[1] *= 2;
            workDiv.m_blockThreadExtent[0] /= 2;
            workDiv.m_blockThreadExtent[1] /= 2;
        }
#endif
        size_t used_shared_memory = 2u * workDiv.m_blockThreadExtent.prod() * workDiv.m_threadElemExtent.prod() * sizeof(TElem);
        std::cout << std::endl << "Workdiv: "
                  << workDiv.m_gridBlockExtent[0] << "*" << workDiv.m_gridBlockExtent[1] << " : "
                  << workDiv.m_blockThreadExtent[0] << "*" << workDiv.m_blockThreadExtent[1] << " : "
                  << workDiv.m_threadElemExtent[0] << "*" << workDiv.m_threadElemExtent[1]
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                  << " (Shared Memory: " << used_shared_memory << " of "
                  << prop.sharedMemPerBlock << ")" << std::endl;
#else
                  << std::endl;
#endif
        using Matrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrValue<TElem>,
            Vec2
        >;

        using ConstMatrix = alpakaHelper::Matrix<
            alpakaHelper2::ConstPtrConstValue<TElem>,
            Vec2
        >;

        ConstMatrix const matA(
            alpaka::mem::view::getPtrNative(bufAAcc),
            Vec2(
                m,
                alpaka::mem::view::getPitchBytes<1>(bufAAcc) / elemSize
            )
        );

        ConstMatrix const matB(
            alpaka::mem::view::getPtrNative(bufBAcc),
            Vec2(
                k,
                alpaka::mem::view::getPitchBytes<1>(bufBAcc) / elemSize
            )
        );
        Matrix matC(
            alpaka::mem::view::getPtrNative(bufCAcc),
            Vec2(
                m,
                alpaka::mem::view::getPitchBytes<1>(bufCAcc) / elemSize
            )
        );

        // Create an instance of the kernel functor.
        TKernelFnObj kernel;

        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            alpha,
            matA,
            matB,
            beta,
            matC));

#ifdef MATMUL_RETURN_COMPUTATION_TIME
        alpaka::wait::wait(stream);
#endif
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
