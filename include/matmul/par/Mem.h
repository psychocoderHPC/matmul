
#pragma once

#include <alpaka/alpaka.hpp>


namespace mem2
{

template<
    typename T
>
struct IdentityAccess
{

    template<
        typename T_IndexType
    >
    ALPAKA_FN_ACC
    auto
    operator()(
        T_IndexType const & extent,
        T_IndexType const & idx
    ) const
    -> alpaka::size::Size<T_IndexType> const
    {
        using LinearIndexType = alpaka::size::Size<T_IndexType>;
        LinearIndexType const col( idx[ 1 ] );
        LinearIndexType const pitch( extent[ 1 ] );
        LinearIndexType const row( pitch * idx[ 0 ] );
        return row + col;
    }
};

template<
    typename T
>
struct TransposeAccess
{

    template<
        typename T_IndexType
    >
    ALPAKA_FN_ACC
    auto
    operator()(
        T_IndexType const & extent,
        T_IndexType const & idx
    ) const
    -> alpaka::size::Size<T_IndexType> const
    {
        using LinearIndexType = alpaka::size::Size<T_IndexType>;
        LinearIndexType const col( idx[ 0 ] );
        LinearIndexType const pitch( extent[ 1 ] );
        LinearIndexType const row( pitch * idx[ 1 ] );
        return row + col;
    }
};


template<
    typename T
>
struct ConstPtrConstValue
{
    using Value = T;
    using ValuePtr = T const * const;
    using ValueRef = T const &;
    using ValueConstRef = T const &;

    ALPAKA_FN_HOST_ACC
    ConstPtrConstValue(
        ValuePtr ptr
    ) : m_ptr(ptr)
    { }

    ValuePtr m_ptr;
};

template<
    typename T
>
struct ConstPtrValue
{
    using Value = T;
    using ValuePtr = T * const;
    using ValueRef = T &;
    using ValueConstRef = T const &;

    ALPAKA_FN_HOST_ACC
    ConstPtrValue(
        ValuePtr ptr
    ) : m_ptr(ptr)
    { }

    ValuePtr m_ptr;
};

} //namepsace mem2

namespace mem
{

template<
    typename T_PtrStorage,
    typename T_IndexType,
    typename T_Access = mem2::IdentityAccess< typename T_PtrStorage::Value >
>
struct Matrix : protected T_PtrStorage
{
    using PtrStorage = T_PtrStorage;
    using IndexType = T_IndexType;
    using Value = typename PtrStorage::Value;
    using ValuePtr = typename PtrStorage::ValuePtr;
    using ValueRef = typename PtrStorage::ValueRef;
    using ValueConstRef = typename PtrStorage::ValueConstRef;
    using ThisType = Matrix<
        PtrStorage,
        IndexType,
        T_Access
    >;

    using Access = const T_Access;


    ALPAKA_FN_HOST_ACC
    Matrix(
        ValuePtr ptr,
        T_IndexType const & extent
    ) :
        PtrStorage( ptr ),
        m_extent(extent)
    {
    }

    ALPAKA_FN_HOST_ACC
    auto
    operator[](
        T_IndexType const & idx
    )
    -> ValueRef
    {
        auto const linearIndex( Access( )( m_extent, idx ) );
        return this->m_ptr[ linearIndex ];
    }

    ALPAKA_FN_ACC
    auto
    operator[](
        T_IndexType const & idx
    ) const
    -> ValueConstRef
    {
        auto const linearIndex( Access( )( m_extent, idx ) );
        return this->m_ptr[ linearIndex ];
    }

    ALPAKA_FN_HOST_ACC
    auto
    view(
        T_IndexType const & offset
    ) const
    -> ThisType
    {
        auto const linearIndex( Access( )( m_extent, offset ) );
        return ThisType(
            static_cast<ValuePtr>(this->m_ptr +  linearIndex ),
            m_extent
        );
    }

    T_IndexType const m_extent;
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

    // data storage
    T m_ptr[ dim ][ dim ];

    ALPAKA_FN_ACC
    MathVec( )
    { }

    template<
        typename Vec2
    >
    ALPAKA_FN_ACC
    auto
    operator[](
        Vec2 const & idx
    ) const
    -> T const &
    {
        return m_ptr[ idx[0] ][ idx[1] ];;
    }

    template<
        typename Vec2
    >
    ALPAKA_FN_ACC
    auto
    operator[](
        Vec2 const & idx
    )
    -> T &
    {
        return m_ptr[ idx[0] ][ idx[1] ];
    }
};

}
