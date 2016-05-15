#ifndef __LINEARALGEBRA_HXX__
#define __LINEARALGEBRA_HXX__

#include "matrix/Matrix.hxx"

template<typename T1, typename T2>
auto operator*(const T1& one, const T2& two) -> decltype(one[0] * two[0]) // inner product
{
    if (one.size() && two.size() && one.size() <= two.size())
    {
        auto sum = one[0] * two[0];
        for (size_t i = 1; i < one.size(); i++)
            sum += one[i] * two[i];

        return sum;
    }

    return 0;
}

template <typename Tuple1, typename Tuple2>
double L2Norm(const Tuple1& t1, const Tuple2& t2)
{
    double d = 0.;
    for (uint i = 0; i < t1.size(); ++i)
    {
        auto diff = t1[i] - t2[i];
        d += diff*diff;
    }
    return sqrt(d);
}

template<typename Tuple1, typename Tuple2>
void operator +=(Tuple1& t1, const Tuple2& t2)
{
    for (uint i = 0; i < t1.size(); ++i)
        t1[i] += t2[i];
}


template<typename Tuple1, typename Tuple2>
void operator -=(Tuple1& t1, const Tuple2& t2)
{
    for (uint i = 0; i < t1.size(); ++i)
        t1[i] -= t2[i];
}

template<typename Tuple1>
void operator /=(Tuple1& t1, double d)
{
    for (uint i = 0; i < t1.size(); ++i)
        t1[i] /= d;
}

template<typename MatrixType>
MatrixType Covariance(const MatrixType& mat)
{
    MatrixType means(mat.Width(), 1);
    for (size_t i = 0; i < mat.Height(); i++)
        means[0] += mat[i];

    means /= mat.Height();

    for (size_t i = 0; i < mat.Height(); i++)
        for (size_t k = 0; k < mat.Width(); k++)
            mat[i][k] -= means[k];

    MatrixType covarianceMatrix(mat.Width(), mat.Width());
    for (size_t k = 0; k < mat.Height(); k++)
        for (size_t i = 0; i < mat.Width(); i++)
            for (size_t j = i; j < mat.Width(); j++)
                covarianceMatrix[i][j] +=  (mat[k][i] * mat[k][j]),
                covarianceMatrix[j][i] = covarianceMatrix[i][j];

    //for(auto& row : covarianceMatrix) 
    //    for (auto& val : row)
    covarianceMatrix.ForEachElement([&](MatrixType::ElemType& e) { e /= (mat.Height() - 1); return true; });

    auto correlationMatrix = covarianceMatrix;
    for (size_t i = 0; i < mat.Width(); i++)
    {
        auto sdI = sqrt(covarianceMatrix[i][i]);
        for (size_t j = i; j < mat.Width(); j++)
            correlationMatrix[i][j] /= (sdI * sqrt(covarianceMatrix[j][j])),
            correlationMatrix[j][i] = correlationMatrix[i][j];
    }
    return covarianceMatrix;
}






#endif
