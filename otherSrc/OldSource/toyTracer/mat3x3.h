/***************************************************************************
* Mat3x3.h                                                                 *
*                                                                          *
* Mat3x3 is a 3x3 matrix class, with associated operators.                 *
*                                                                          *
* History:                                                                 *
*   04/25/2010  Added matrix norms and shear matrices.                     *
*   04/16/2010  Changed "Adjoint" to "Adjugate", added Rotation.           *
*   10/16/2004  Added ^ operator for multiplying by the transpose.         *
*   10/10/2004  Added Inverse function.                                    *
*   04/07/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __MAT3X3_INCLUDED__
#define __MAT3X3_INCLUDED__

#include <iostream>
#include "vec3.h"

struct Mat3x3 {
    inline Mat3x3();
    inline Mat3x3( const Vec3 &A, const Vec3 &B, const Vec3 &C );
    inline ~Mat3x3() {}
    inline       double &operator()( int i, int j )       { return m[i][j]; }
    inline const double &operator()( int i, int j ) const { return m[i][j]; }
    inline static Mat3x3 Identity();
    inline Vec3 row( int i ) const { return Vec3( m[i][0], m[i][1], m[i][2] ); }
    inline Vec3 col( int j ) const { return Vec3( m[0][j], m[1][j], m[2][j] ); }
    double m[3][3];
    };

inline Mat3x3::Mat3x3()
    {
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ ) m[i][j] = 0.0;
    }

// Construct the matrix from three vectors that are taken to be columns.
inline Mat3x3::Mat3x3( const Vec3 &A, const Vec3 &B, const Vec3 &C )
    {
    m[0][0] = A.x;  m[0][1] = B.x;  m[0][2] = C.x;
    m[1][0] = A.y;  m[1][1] = B.y;  m[1][2] = C.y;
    m[2][0] = A.z;  m[2][1] = B.z;  m[2][2] = C.z;
    }

inline Mat3x3 operator+( const Mat3x3 &A, const Mat3x3 &B )
    {
    Mat3x3 C;
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ ) C(i,j) = A(i,j) + B(i,j);
    return C;
    }

inline Mat3x3 operator-( const Mat3x3 &A, const Mat3x3 &B )
    {
    Mat3x3 C;
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ ) C(i,j) = A(i,j) - B(i,j);
    return C;
    }

inline Mat3x3 operator*( double c, const Mat3x3 &M )
    {
    Mat3x3 A;
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ ) A(i,j) = c * M(i,j);
    return A;    
    }

inline Mat3x3 operator*( const Mat3x3 &M, double c )
    {
    return c * M;
    }

// Transform the column vector A, multiplying on the left by matrix M.
inline Vec3 operator*( const Mat3x3 &M, const Vec3 &A )
    {
    return Vec3(
        M(0,0) * A.x + M(0,1) * A.y + M(0,2) * A.z,
        M(1,0) * A.x + M(1,1) * A.y + M(1,2) * A.z,
        M(2,0) * A.x + M(2,1) * A.y + M(2,2) * A.z
        );
    }

 // Multiply the vector A on the left by the TRANSPOSE of the matrix.
inline Vec3 operator^( const Mat3x3 &M, const Vec3 &A )
    {
    return Vec3(
        M(0,0) * A.x + M(1,0) * A.y + M(2,0) * A.z,
        M(0,1) * A.x + M(1,1) * A.y + M(2,1) * A.z,
        M(0,2) * A.x + M(1,2) * A.y + M(2,2) * A.z
        );
    }

// Standard matrix multiplication.
inline Mat3x3 operator*( const Mat3x3 &A, const Mat3x3 &B )
    {    
    Mat3x3 C;
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
        C(i,j) = A(i,0) * B(0,j) + A(i,1) * B(1,j) + A(i,2) * B(2,j);
    return C;
    }

inline Mat3x3 operator/( const Mat3x3 &M, double c )
    {
    return (1/c) * M;
    }

// Compute the outer product of two vectors, which results in a matrix.
inline Mat3x3 operator%( const Vec3 &A, const Vec3 &B )
    {
    Mat3x3 M;
    M(0,0) = A.x * B.x;  M(0,1) = A.x * B.y;  M(0,2) = A.x * B.z;
    M(1,0) = A.y * B.x;  M(1,1) = A.y * B.y;  M(1,2) = A.y * B.z;
    M(2,0) = A.z * B.x;  M(2,1) = A.z * B.y;  M(2,2) = A.z * B.z;
    return M;
    }

// Determinant.
inline double det( const Mat3x3 &M )
    {
    return
        M(0,0) * ( M(1,1) * M(2,2) - M(1,2) * M(2,1) )
      - M(0,1) * ( M(1,0) * M(2,2) - M(1,2) * M(2,0) )
      + M(0,2) * ( M(1,0) * M(2,1) - M(1,1) * M(2,0) );
    }

inline Mat3x3 Transpose( const Mat3x3 &M )
    {
    Mat3x3 W;
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ ) W(i,j) = M(j,i);
    return W;
    }

inline Mat3x3 Mat3x3::Identity()
    {
    Mat3x3 I;
    I(0,0) = 1.0;
    I(1,1) = 1.0;
    I(2,2) = 1.0;
    return I;
    }

// Rotation about the X-axis.
inline Mat3x3 Rotate_X( double angle )
    {
    Mat3x3 M( Mat3x3::Identity() );
    M(1,1) = cos( angle );  M(1,2) = -sin( angle );
    M(2,1) = sin( angle );  M(2,2) =  cos( angle );
    return M;
    }

// Rotation about the Y-axis.
inline Mat3x3 Rotate_Y( double angle )
    {
    Mat3x3 M( Mat3x3::Identity() );
    M(0,0) = cos( angle );  M(0,2) = -sin( angle );
    M(2,0) = sin( angle );  M(2,2) =  cos( angle );
    return M;
    }

// Rotation about the Z-axis.
inline Mat3x3 Rotate_Z( double angle )
    {    
    Mat3x3 M( Mat3x3::Identity() );
    M(0,0) = cos( angle );  M(0,1) = -sin( angle );
    M(1,0) = sin( angle );  M(1,1) =  cos( angle );
    return M;
    }

// Allows non-uniform scaling along all three coordinate axes.
inline Mat3x3 Scale( double x, double y, double z )
    {
    Mat3x3 M;
    M(0,0) = x;
    M(1,1) = y;
    M(2,2) = z;
    return M;
    }

// Shear parallel to the X-axis by the given amounts along
// both the Y and Z axes.
inline Mat3x3 Shear_X( double shear_y, double shear_z = 0.0 )
    {
    Mat3x3 M( Mat3x3::Identity() );
    M(0,1) = shear_y;
    M(0,2) = shear_z;
    return M;
    }

// Shear parallel to the Y-axis by the given amounts along
// both the X and Z axes.
inline Mat3x3 Shear_Y( double shear_x, double shear_z = 0.0 )
    {    
    Mat3x3 M( Mat3x3::Identity() );
    M(1,0) = shear_x;
    M(1,2) = shear_z;
    return M;
    }

// Shear parallel to the Z-axis by the given amounts along
// both the X and Y axes.
inline Mat3x3 Shear_Z( double shear_x, double shear_y = 0.0 )
    {    
    Mat3x3 M( Mat3x3::Identity() );
    M(2,0) = shear_x;
    M(3,1) = shear_y;
    return M;
    }

// Compute the infinity-norm of the matrix M, which is the
// maximum absolute row sum.
extern double InfinityNorm(
    const Mat3x3 &M
    );

// Compute the 1-norm of the matrix M, which is the
// maximum absolute column sum.
extern double OneNorm(
    const Mat3x3 &M
    );

// Compute the Adjugate of the given matrix, which is closely related to
// the inverse of the matrix, and it always exists.  To obtain the inverse
// (when it exists), we take the transpose and divide by the determinant.
extern Mat3x3 Adjugate(
    const Mat3x3 &M
    );

// Compute the inverse of a matrix using its Adjugate 
// and determinant.
inline Mat3x3 Inverse( const Mat3x3 &M )
    {
    return Transpose( Adjugate( M ) ) / det( M );
    }

// Rotation about an arbitrary axis.
extern Mat3x3 Rotation(
    const Vec3 &Axis,
    double angle
    );

// A simple output operator for Mat3x3.
inline std::ostream &operator<<( std::ostream &out, const Mat3x3 &M )
    {
    out << "("
        << M(0,0) << ", " << M(0,1) << ", " << M(0,2) << "; "
        << M(1,0) << ", " << M(1,1) << ", " << M(1,2) << "; "
        << M(2,0) << ", " << M(2,1) << ", " << M(2,2) 
        << ")";
    return out;
    }

#endif


