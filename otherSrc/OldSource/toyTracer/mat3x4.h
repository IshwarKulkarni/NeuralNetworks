/***************************************************************************
* Mat3x4.h                                                                 *
*                                                                          *
* Mat3x4 is a 3x4 matrix class, with associated operators.  Mathematically,*
* this type of matrix behaves exactly like the 4x4 matrix formed by        *
* appending the last row of the 4x4 identity matrix to it, and appending   *
* a fourth element of 1 to each 3D vector it is multiplied with.  However, *
* we represent such a matrix as a 3x3 matrix coupled with a 3D vector.     *
* Indexing by row and column is therefore limited to the upper left 3x3    *
* portion of the matrix.                                                   *
*                                                                          *
* History:                                                                 *
*   10/16/2010  Added *= operators.                                        *
*   10/10/2004  Added Inverse function.                                    *
*   10/06/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __MAT3X4_INCLUDED__
#define __MAT3X4_INCLUDED__

#include <iostream>
#include "mat3x3.h"
#include "vec3.h"

struct Mat3x4 {
    inline Mat3x4() : mat(Mat3x3::Identity()) {}
    inline Mat3x4( const Mat3x3 &m                ) : mat(m) {}
    inline Mat3x4( const Mat3x3 &m, const Vec3 &v ) : mat(m), vec(v) {}
    inline ~Mat3x4() {}
    inline static Mat3x4 Identity();
    Mat3x3 mat;
    Vec3   vec;
    };

inline Mat3x4 operator+( const Mat3x4 &A, const Mat3x4 &B )
    {
    return Mat3x4( A.mat + B.mat, A.vec + B.vec );
    }

inline Mat3x4 operator-( const Mat3x4 &A, const Mat3x4 &B )
    {
    return Mat3x4( A.mat - B.mat, A.vec - B.vec );
    }

inline Mat3x4 operator*( double c, const Mat3x4 &M )
    {
    return Mat3x4( c * M.mat, c * M.vec );
    }

inline Mat3x4 operator*( const Mat3x4 &M, double c )
    {
    return c * M;
    }

inline Vec3 operator*( const Mat3x4 &M, const Vec3 &A )
    {
    return Vec3( M.mat * A + M.vec );
    }

inline Mat3x4 operator*( const Mat3x4 &A, const Mat3x4 &B )
    {
    return Mat3x4( A.mat * B.mat, A.mat * B.vec + A.vec );
    }

inline Mat3x4 operator/( const Mat3x4 &M, double c )
    {
    return (1/c) * M;
    }

inline Mat3x4 &operator+=( Mat3x4 &A, const Mat3x4 &B )
    {
    A = A + B;
    return A;
    }

inline Mat3x4 &operator-=( Mat3x4 &A, const Mat3x4 &B )
    {
    A = A - B;
    return A;
    }

inline Mat3x4 &operator*=( Mat3x4 &A, const Mat3x4 &B )
    {
    A = A * B;
    return A;
    }

inline Vec3 &operator*=( Vec3 &v, const Mat3x4 &M )
    {
    v = M * v;
    return v;
    }

// Allow operations with 3x3 matrices by assuming they correspond to
// 3x4 matrices with zero translations.  Perform these explicitly, rather
// than allowing coersion from 3x3 to 3x4, to avoid unnecessary arithemtic.
inline Mat3x4 operator*( const Mat3x3 &A, const Mat3x4 &B )
    {
    // Assume translation of the A matrix is zero.
    return Mat3x4( A * B.mat, A * B.vec  );
    }

// Same as above, but with the 3x3 matrix on the right.
inline Mat3x4 operator*( const Mat3x4 &A, const Mat3x3 &B )
    {
    // Assume translation of the B matrix is zero.
    return Mat3x4( A.mat * B, A.vec );
    }

// While the inverse of a non-square matrix is technically undefined,
// it's convenient to treat a 3x4 matrix as though it is actually 4x4
// by augmenting it with the last row of the 4x4 identity matrix.
// The inverse of the resulting 4x4 has the same structure (with 
// 0, 0, 0, 1 in the last row) so we simply return the upper 3x4
// portion as the "inverse" of the original matrix.
inline Mat3x4 Inverse( const Mat3x4 &M )
    {
    const Mat3x3 W( Inverse( M.mat ) ); // Compute inverse of 3x3 portion.
    return Mat3x4( W, -( W * M.vec ) );
    }

inline Mat3x4 Mat3x4::Identity()
    {
    Mat3x4 I;
    I.mat(0,0) = 1.0;
    I.mat(1,1) = 1.0;
    I.mat(2,2) = 1.0;
    return I;
    }

// Return a 3x4 matrix that performs a translation.
inline Mat3x4 Translate( double x, double y, double z )
    {
    return Mat3x4( Mat3x3::Identity(), Vec3( x, y, z ) );
    }

// Return a 3x4 matrix that performs a rotation about a given axis
// through a given point.  The rotation is right-handed about this axis
// and "angle" is taken to be in radians.
extern Mat3x4 Rotation(
    const Vec3 &Axis,
    const Vec3 &Point,
    double angle
    );

// A simple output operator for Mat3x4.  Writes a 3x4 matrix in the same Matlab-like
// format that the toytracer parser can read.
inline std::ostream &operator<<( std::ostream &out, const Mat3x4 &M )
    {
    out << "("
        << M.mat(0,0) << ", " << M.mat(0,1) << ", " << M.mat(0,2) << ", " << M.vec.x << "; "
        << M.mat(1,0) << ", " << M.mat(1,1) << ", " << M.mat(1,2) << ", " << M.vec.y << "; "
        << M.mat(2,0) << ", " << M.mat(2,1) << ", " << M.mat(2,2) << ", " << M.vec.z
        << ")";
    return out;
    }

#endif

