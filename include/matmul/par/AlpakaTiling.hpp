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

#define BLOCK_SIZE GPU_TILE_SIZE
#define THREADS GPU_THREADS_NUM

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
    _Pragma ("unroll") \
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

    template<
        typename T_Type,
        size_t T_size
    >
    struct Array{
        T_Type m_data[T_size];

        template<
            typename T_Idx
        >
        ALPAKA_FN_HOST_ACC
        const T_Type &
        operator[](
            const T_Idx idx
        ) const {
            return m_data[idx];
        }

        template<
            typename T_Idx
        >
        ALPAKA_FN_HOST_ACC
        T_Type &
        operator[](
            const T_Idx idx
        ){
            return m_data[idx];
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
            TSize const & wA, TSize const & wB,
            TElem const & alpha,
            const MatA *const MATMUL_RESTRICT A,
            const MatB *const MATMUL_RESTRICT B,
            TElem const & beta,
            MatC* const MATMUL_RESTRICT C) const
        -> void
        {

            auto const blockIndex(alpaka::idx::getIdx<alpaka::Grid, alpaka::Blocks>(acc));

            const TSize threadIndex = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
            TSize tx2 = threadIndex % BLOCK_SIZE;
            TSize ty2 = threadIndex / BLOCK_SIZE;

            constexpr TSize jumpLength = (THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;
            constexpr TSize ELEM = BLOCK_SIZE / jumpLength;
            constexpr TSize ELEM_X = (BLOCK_SIZE + THREADS - 1u)  / THREADS;

            TSize aBegin = wA * blockIndex[0] * BLOCK_SIZE;
            TSize aEnd   = aBegin + m - 1;
            TSize aStep  = BLOCK_SIZE;

            TSize bBegin = BLOCK_SIZE * blockIndex[1];
            TSize bStep  = wB * BLOCK_SIZE;

            Array<
                Array<
                    TElem,
                    ELEM_X
                >,
                ELEM
             > Csub;
            for(int y=0;y<ELEM;++y)
            {
                for(TSize x=0;x<ELEM_X;++x)
                    Csub[y][x]=0;
            }
            TElem * const __restrict As(alpaka::block::shared::dyn::getMem<TElem>(acc));
            TElem * const __restrict Bs(As + BLOCK_SIZE * BLOCK_SIZE);
#if 0
            auto& As = ::alpaka::block::shared::st::allocVar<
                Array<
                    TElem,
                    BLOCK_SIZE * BLOCK_SIZE
                >,
                0
            >(acc);
            auto& Bs = ::alpaka::block::shared::st::allocVar<
                Array<
                    TElem,
                    BLOCK_SIZE * BLOCK_SIZE
                >,
                1
            >(acc);
#endif
            VECTOR_PRAGMA
            for (TSize a = aBegin, b = bBegin;
                 a <= aEnd;
                 a += aStep, b += bStep)
            {
                VECTOR_PRAGMA
                for(TSize y=0;y<ELEM;++y)
                {
                    VECTOR_PRAGMA
                    for(TSize x=0;x<ELEM_X;++x)
                    {
                        TSize ty = ty2 + y * jumpLength;
                        As[ty * BLOCK_SIZE + tx2 + x * THREADS] = A[a + wA * ty + tx2 + x*THREADS];
                        Bs[ty * BLOCK_SIZE + tx2 + x * THREADS] = B[b + wB * ty + tx2 + x*THREADS];
                    }
                }
                alpaka::block::sync::syncBlockThreads(acc);

                VECTOR_PRAGMA
                for (TSize k = 0; k < BLOCK_SIZE; ++k)
                {
                    const TElem * const bs = &(Bs[k * BLOCK_SIZE + tx2]);
                    VECTOR_PRAGMA
                    for(TSize y=0;y<ELEM;++y)
                    {
                        const TSize ty = ty2 + y * jumpLength;
                        const TElem as = As[ty * BLOCK_SIZE + k];
                        VECTOR_PRAGMA
                        for(TSize x=0;x<ELEM_X;++x)
                        {
                            Csub[y][x] += as * bs[x*THREADS];
                        }
                    }
                }

                alpaka::block::sync::syncBlockThreads(acc);
            }

            const TSize c = wB * blockIndex[0] * BLOCK_SIZE + BLOCK_SIZE * blockIndex[1] + tx2;
            VECTOR_PRAGMA
            for(TSize y=0;y<ELEM;++y)
            {
                const TSize ty = ty2 + y * jumpLength;
                VECTOR_PRAGMA
                for(TSize x=0;x<ELEM_X;++x)
                {
                    const TSize cOffset = c + wB * ty + x*THREADS;
                    C[cOffset] = alpha * Csub[y][x] + beta * C[cOffset];
                }
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
                        TSize const & wA,
                        TSize const & wB,
                        TElem const & alpha,
                        const MatA *const MATMUL_RESTRICT matA,
                        const MatB *const MATMUL_RESTRICT matB,
                        TElem const & beta,
                        MatC* const matC)
                    -> size::Size<TAcc>
                    {
                        static_assert(
                            std::is_same<TSize, size::Size<TAcc>>::value,
                            "TSize and size::Size<TAcc> have to be identical!");

                        boost::ignore_unused(m);
                        boost::ignore_unused(k);
                        boost::ignore_unused(alpha);
                        boost::ignore_unused(matA);
                        boost::ignore_unused(matB);
                        boost::ignore_unused(beta);
                        boost::ignore_unused(matC);

                        // Reserve the buffer for the two blocks of A and B.
                        return 2u * BLOCK_SIZE * BLOCK_SIZE * sizeof(TElem);;
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
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        if(sizeof(TElem)==8u)
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif

        // Get a stream on this device.
        Stream<alpaka::dev::Dev<TAcc>> stream(devAcc);

        // Result matrix is MxN. We create one worker per result matrix cell.
        Vec2 const v2uiExtentsC(
            m,
            n);

        constexpr TSize jumpLength = (THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        constexpr TSize ELEM = BLOCK_SIZE / jumpLength;
        constexpr TSize ELEM_X = (BLOCK_SIZE + THREADS - 1u)  / THREADS;

        Vec2 const elemExtent(
            static_cast<TSize>(ELEM),
            static_cast<TSize>(ELEM_X)
        );

        // Let alpaka calculate good block and grid sizes given our full problem extents.
        const alpaka::vec::Vec<Dim2, TSize> threads (TSize(1), TSize(THREADS));
        const alpaka::vec::Vec<Dim2, TSize> blocks  (m/TSize(BLOCK_SIZE), n/TSize(BLOCK_SIZE));
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(blocks,threads,elemExtent);


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


        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            lda,
            ldb,
            alpha,
            A,
            B,
            beta,
            C));
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
        TElem const * const A, TSize const lda,
        TElem const * const B, TSize const ldb,
        TElem const beta,
        TElem * const C, TSize const ldc)
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

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
        if(sizeof(TElem)==8u)
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif

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

        constexpr TSize jumpLength = (THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        constexpr TSize ELEM = BLOCK_SIZE / jumpLength;
        constexpr TSize ELEM_X = (BLOCK_SIZE + THREADS - 1u)  / THREADS;

        Vec2 const elemExtent(
            static_cast<TSize>(ELEM),
            static_cast<TSize>(ELEM_X)
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

        const alpaka::vec::Vec<Dim2, TSize> threads (TSize(1), TSize(THREADS));
        const alpaka::vec::Vec<Dim2, TSize> blocks  (m/TSize(BLOCK_SIZE), n/TSize(BLOCK_SIZE));
        alpaka::workdiv::WorkDivMembers<Dim2, TSize> workDiv(blocks,threads,elemExtent);


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

        // Create the executor.
        auto const exec(alpaka::exec::create<TAcc>(
            workDiv,
            kernel,
            m,
            n,
            k,
            lda,
            ldb,
            alpha,
            A,
            B,
            beta,
            C));

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
