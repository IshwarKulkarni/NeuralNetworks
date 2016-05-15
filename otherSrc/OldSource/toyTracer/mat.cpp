/***************************************************************************
* mat.cpp                                                                  *
*                                                                          *
* Matrix operations that are not performed by inline functions in the      *
* header files.                                                            *
*                                                                          *
* History:                                                                 *
*   04/25/2010  Added 3x3 matrix norms.                                    *
*   04/18/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <cmath>
#include <iostream>
#include "mat3x3.h"
#include "mat3x4.h"
#include "util.h"

// The "adjugate" of a matrix is closely realted to its inverse, but unlike the
// inverse, it is guaranteed to exist.  To obtain the inverse matrix from the
// adjugate (when the former exists), we take the transpose and divide by the
// determinant.  The adjugate is also known as the "classical adjoint" matrix.
Mat3x3 Adjugate( const Mat3x3 &M )  
    {
    Mat3x3 A;
    A(0,0) = M(1,1) * M(2,2) - M(1,2) * M(2,1);
    A(0,1) = M(1,2) * M(2,0) - M(1,0) * M(2,2);
    A(0,2) = M(1,0) * M(2,1) - M(1,1) * M(2,0);
 
    A(1,0) = M(0,2) * M(2,1) - M(0,1) * M(2,2);
    A(1,1) = M(0,0) * M(2,2) - M(0,2) * M(2,0);
    A(1,2) = M(0,1) * M(2,0) - M(0,0) * M(2,1);

    A(2,0) = M(0,1) * M(1,2) - M(0,2) * M(1,1);
    A(2,1) = M(0,2) * M(1,0) - M(0,0) * M(1,2);
    A(2,2) = M(0,0) * M(1,1) - M(0,1) * M(1,0);
    return A;
    }

// Returns a 3x3 matrix that performs a rotation about an arbitrary axis.
// The rotation is right-handed about this axis and "angle" is taken to be in
// radians.  The only error that can occur is when "axis" is the zero-vector.
// The axis vector need not be normalized.
Mat3x3 Rotation( const Vec3 &Axis, double angle )
    {
    // Compute a unit quaternion (a,b,c,d) that performs the rotation.

    double t( LengthSquared( Axis ) );
    if( t == 0 ) return Mat3x3();
    t = sin( angle * 0.5 ) / sqrt( t );

    // Fill in the entries of the quaternion.

    const double
        a( cos( angle * 0.5 ) ),
        b( t * Axis.x ),
        c( t * Axis.y ),
        d( t * Axis.z );

    // Compute all the double products of a, b, c, and d, except a * a.

    const double
        bb( b * b ),
        cc( c * c ),
        dd( d * d ),
        ab( a * b ),
        ac( a * c ),
        ad( a * d ),
        bc( b * c ),
        bd( b * d ),
        cd( c * d );

    // Fill in the entries of the rotation matrix.

    Mat3x3 R;

    R(0,0) = 1 - 2 * ( cc + dd );
    R(0,1) =     2 * ( bc + ad );
    R(0,2) =     2 * ( bd - ac );

    R(1,0) =     2 * ( bc - ad );
    R(1,1) = 1 - 2 * ( bb + dd );
    R(1,2) =     2 * ( cd + ab );

    R(2,0) =     2 * ( bd + ac );
    R(2,1) =     2 * ( cd - ab );
    R(2,2) = 1 - 2 * ( bb + cc );

    return R;
    }

// Return a 3x4 matrix that performs a rotation about a given axis
// through a given point.  The rotation is right-handed about this axis
// and "angle" is taken to be in radians.
Mat3x4 Rotation( const Vec3 &Axis, const Vec3 &Point, double angle )
    {
    const Mat3x3 R( Rotation( Axis, angle ) ); // Simple 3x3 rotation.

    // Compute the last column of the matrix (the translation) using the
    // 3x3 rotation matrix.  The full 4x4 matrix would be
    //
    //       | I   P | | R   0 | | I  -P |   | R   P - RP |
    //       |       | |       | |       | = |            |
    //       | 0   1 | | 0   1 | | 0   1 |   | 0      1   |
    //
    // Therefore, the desired column is P - R P, where P is the "Point".

    return Mat3x4( R, Point - R * Point );
    }

// Compute the infinity-norm of the matrix M.
double InfinityNorm( const Mat3x3 &M )
    {
    double r0( fabs( M(0,0) ) + fabs( M(0,1) ) + fabs( M(0,2) ) );
    double r1( fabs( M(1,0) ) + fabs( M(1,1) ) + fabs( M(1,2) ) );
    double r2( fabs( M(2,0) ) + fabs( M(2,1) ) + fabs( M(2,2) ) );
    return max( r0, r1, r2 );
    }

// Compute the 1-norm of the matrix M.
double OneNorm( const Mat3x3 &M )
    {
    double c0( fabs( M(0,0) ) + fabs( M(1,0) ) + fabs( M(2,0) ) );
    double c1( fabs( M(0,1) ) + fabs( M(1,1) ) + fabs( M(2,1) ) );
    double c2( fabs( M(0,2) ) + fabs( M(1,2) ) + fabs( M(2,2) ) );
    return max( c0, c1, c2 );
    }


