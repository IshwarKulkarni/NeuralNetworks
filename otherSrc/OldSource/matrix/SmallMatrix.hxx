#ifndef SMALLMATRIX2D_INCLUDED
#define SMALLMATRIX2D_INCLUDED

#include "Matrix.hxx"

template<typename Type>   // an H row x W col matrix that is in place
class SmallMatrix2D: public Matrix2D<Type> // and so neither copies data nor deletes it
{
public:
    SmallMatrix2D(uint W, uint H, Type* data): Matrix2D(W,H,data)
    {
    }

    virtual ~SmallMatrix2D()
    {
        // nothing, don't delete data, we dont own it
    }
    
    template<typename TypeOne, typename TypeTwo>
    inline auto SetAsProductOf(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> decltype(one.Get(0,0) * two.Get(0,0))*
    {
        THROW_IF( one.Width() != two.Height(), DimensionException, "Dimensions mismatch in multiplication");
        THROW_IF( NumElems() != one.Width()*two.Height(), DimensionException, "Product does not fit in this matrix");

        for (uint i = 0; i < one.Height(); ++i)
            for (uint j = 0; j < two.Width(); j++)
                for (uint k = 0; k < one.Width(); k++)
                    (*this)(i,j) += one(i,k) * to(k,j);
    }
};

//template<typename Type>
//std::ostream& operator<<(std::ostream& strm, const Matrix2D<Type>& M)
//{
//    strm << "\nMatrix2D Id: "<< M.GetId() << "\n";
//    for ( uint x = 0; x < M.Width(); ++x )
//    {
//        for ( uint y = 0; y < M.Height(); ++y ) 
//            strm << std::setw(8) << std::setprecision(4) << M(x,y) ;
//        strm << "\n";
//    }
//
//    return strm;
//}

template<typename TypeOne, typename TypeTwo>
inline auto operator^(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> Matrix2D<decltype(one.Get(0,0) * two.Get(0,0))>
{
    THROW_IF( one.Width() != two.Height(), DimensionException, "Dimensions mismatch in multiplication");

    typedef decltype(one.Get(0,0) * two.Get(0,0)) ReturnDataType;
    Matrix2D<ReturnDataType> ret(two.Width(), one.Height()); // H1 x W1 * H2 x W2 => H1 * W2;

    for (uint i = 0; i < one.Height(); ++i)
        for (uint j = 0; j < two.Width(); j++)
            for (uint k = 0; k < one.Width(); k++)
                ret(i,j) += one(i,k) * to(k,j);
    
    return ret;
}


template<typename TypeOne, typename TypeTwo>
inline auto operator+(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> Matrix2D<decltype(one.Get(0,0) + two.Get(0,0))>
{
    THROW_IF( two.Height() != one.Height() || two.Width() != one.Width() , DimensionException, "Dimensions mismatch is addition");

    Matrix2D<decltype( one.Get(0,0) + two.Get(0,0))> ret(one.Width(),one.Height());
    for (uint i = 0; i < one.Width()*one.Height(); i++)
        ret[i] = one[i]+two[i];

    return ret;
}


template<typename TypeOne, typename TypeTwo>
inline auto operator-(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> Matrix2D<decltype(one.Get(0,0) - two.Get(0,0))>
{
    THROW_IF( two.Height() != one.Height() || two.Width() != one.Width() , DimensionException, "Dimensions mismatch is addition");

    Matrix2D<decltype( one.Get(0,0) + two.Get(0,0))> ret(one.Width(),one.Height());
    for (uint i = 0; i < one.Width()*one.Height(); i++)
        ret[i] = one[i] - two[i];

    return ret;
}



template<typename TypeOne, typename TypeTwo>
inline auto operator*(const Matrix2D<TypeOne>& one, TypeTwo c) -> Matrix2D<decltype(one.Get(0,0) * c)>
{
    typedef decltype(one.Get(0,0) * c) ReturnDataType;
    Matrix2D<ReturnDataType> ret(one); 
    
    ret.ForEachElement([&](auto* V) { V*=c; true;});

    return ret;
}




#endif