
#pragma once

#include <alpaka/alpaka.hpp>

namespace mem
{

using IndexType = size_t;
using Dim2 = alpaka::dim::DimInt<2u>;
using Vec2 = alpaka::Vec<Dim2,IndexType>;

template<
    typename T
>
struct IdentityAccess
{
    ALPAKA_FN_ACC
    auto
    operator()(
        Vec2 const & extent,
        Vec2 const & idx
    ) const
    -> IndexType const
    {
        IndexType const col( idx[ 1 ] );
        IndexType const pitch( extent[ 1 ] );
        IndexType const row( pitch * idx[ 0 ] );
        return row + col;
    }
};

template<
    typename T
>
struct TransposeAccess
{
    ALPAKA_FN_ACC
    auto
    operator()(
        Vec2 const & extent,
        Vec2 const & idx
    ) const
    -> IndexType const
    {
        IndexType const col( idx[ 0 ] );
        IndexType const pitch( extent[ 1 ] );
        IndexType const row( pitch * idx[ 1 ] );
        return row + col;
    }
};

template<
    typename T,
    template <typename> class T_Access = IdentityAccess
>
struct Matrix
{
    using Access = const T_Access<T>;

    ALPAKA_FN_ACC
    Matrix(
        T * const ptr,
        Vec2 const & extent
    ) :
        m_ptr(ptr),
        m_extent(extent)
    {
    }

    ALPAKA_FN_ACC
    Matrix(
        const T * const ptr,
        Vec2 const & extent
    ) :
        m_ptr( const_cast<T*>( ptr ) ),
        m_extent(extent)
    {
    }

    ALPAKA_FN_ACC
    auto
    operator[](
        Vec2 const & idx
    )
    -> T &
    {
        const IndexType linearIndex( Access( )( m_extent, idx ) );
        return m_ptr[ linearIndex ];
    }

    ALPAKA_FN_ACC
    auto
    operator[](
        Vec2 const & idx
    ) const
    -> T const &
    {
        const IndexType linearIndex( Access( )( m_extent, idx ) );
        return m_ptr[ linearIndex ];
    }

    T * const m_ptr;
    Vec2 const m_extent;
};

template<
    typename T,
    typename T_Dim
>
struct MathVec
{

    using ThisType = MathVec<
        T,
        T_Dim
    >;
    static constexpr auto dim = T_Dim::value;
    using DimIndexType = size_t; //use alpaka trait

    // data storage
    T m_ptr[ dim ];

    ALPAKA_FN_ACC
    MathVec( )
    { }

 /*   template<
        typename ...T_Args
    >
    ALPAKA_FN_ACC
    MathVec(
        T_Args & ...args
    ) :
    m_ptr{
        args...
    }
    { }
*/
    ALPAKA_FN_ACC
    auto
    operator[](
        DimIndexType const idx
    ) const
    -> T const &
    {
        return m_ptr[ idx ];
    }

    ALPAKA_FN_ACC
    auto
    operator[](
        DimIndexType const idx
    )
    -> T &
    {
        return m_ptr[ idx ];
    }

    ALPAKA_FN_ACC
    auto
    operator*(
        ThisType const & other
    )
    -> ThisType
    {
        ThisType tmp;
        for(size_t j{0};j<dim;++j)
            tmp[ j ] = m_ptr[ j ] * other[ j ];
        return tmp;
    }

    ALPAKA_FN_ACC
    auto
    operator+=(
        ThisType const & other
    )
    -> ThisType &
    {
        for(size_t j{0};j<dim;++j)
            m_ptr[ j ] += other[ j ];
        return *this;
    }

    ALPAKA_FN_ACC
    auto
    operator=(
        ThisType const & other
    )
    -> ThisType &
    {
        for(size_t j{0};j<dim;++j)
            m_ptr[ j ] = other[ j ];
        return *this;
    }

};

}
