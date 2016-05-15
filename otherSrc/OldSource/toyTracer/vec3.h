/***************************************************************************
* vec3.h                                                                   *
*                                                                          *
* Vec3 is a trivial encapsulation of 3D floating-point coordinates.        *
* It has all of the obvious operators defined as inline functions.         *
*                                                                          *
* History:                                                                 *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __VEC3_INCLUDED__
#define __VEC3_INCLUDED__

#include <cmath>
#include <iostream>

struct Vec3 {
    inline Vec3() : x(0.0), y(0.0), z(0.0) {}
    inline Vec3( double a, double b, double c ) : x(a), y(b), z(c) {}
    inline ~Vec3() {}
    inline void Zero() { x = 0.0; y = 0.0; z = 0.0; }
    double x;
    double y;
    double z;
    static inline Vec3 X_axis() { return Vec3( 1.0, 0.0, 0.0 ); }
    static inline Vec3 Y_axis() { return Vec3( 0.0, 1.0, 0.0 ); }
    static inline Vec3 Z_axis() { return Vec3( 0.0, 0.0, 1.0 ); }
    };

inline double LengthSquared( const Vec3 &A )
    {
    return A.x * A.x + A.y * A.y + A.z * A.z;
    }

inline double Length( const Vec3 &A )
    {
    return sqrt( LengthSquared( A ) );
    }

inline Vec3 operator+( const Vec3 &A, const Vec3 &B )
    {
    return Vec3( A.x + B.x, A.y + B.y, A.z + B.z );
    }

inline Vec3 operator-( const Vec3 &A, const Vec3 &B )
    {
    return Vec3( A.x - B.x, A.y - B.y, A.z - B.z );
    }

// Unary minus operator.
inline Vec3 operator-( const Vec3 &A )
    {
    return Vec3( -A.x, -A.y, -A.z );
    }

inline Vec3 operator*( double a, const Vec3 &A )
    {
    return Vec3( a * A.x, a * A.y, a * A.z );
    }

inline Vec3 operator*( const Vec3 &A, double a )
    {
    return Vec3( a * A.x, a * A.y, a * A.z );
    }

// Inner product.
inline double operator*( const Vec3 &A, const Vec3 &B )
    {
    return (A.x * B.x) + (A.y * B.y) + (A.z * B.z);
    }

inline Vec3 operator/( const Vec3 &A, double c )
    {
    return Vec3( A.x / c, A.y / c, A.z / c );
    }

// Cross product of A and B.
inline Vec3 operator^( const Vec3 &A, const Vec3 &B )
    {
    return Vec3( 
        A.y * B.z - A.z * B.y,
        A.z * B.x - A.x * B.z,
        A.x * B.y - A.y * B.x
        );
    }

inline Vec3 & operator+=( Vec3 &A, const Vec3 &B )
    {
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    return A;
    }

inline Vec3 & operator-=( Vec3 &A, const Vec3 &B )
    {
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    return A;
    }

inline Vec3 & operator*=( Vec3 &A, double a )
    {
    A.x *= a;
    A.y *= a;
    A.z *= a;
    return A;
    }

inline Vec3 & operator/=( Vec3 &A, double a )
    {
    A.x /= a;
    A.y /= a;
    A.z /= a;
    return A;
    }

// Remove the component of A that is parallel to B.
inline Vec3 operator/( const Vec3 &A, const Vec3 &B )
    {
    const double x( LengthSquared( B ) );
    return ( x > 0.0 ) ? ( A - (( A * B ) / x) * B ) : A;
    }

inline bool operator==( const Vec3 &A, const Vec3 &B )
    {
	return ( A.x == B.x ) && ( A.y == B.y ) && ( A.z == B.z );
    }

inline bool operator!=( const Vec3 &A, const Vec3 &B )
    {
	return ( A.x != B.x ) || ( A.y != B.y ) || ( A.z != B.z );
    }

// Return a normalized version of A.
inline Vec3 Unit( const Vec3 &A )
    {
    const double d( LengthSquared( A ) );
    return d > 0.0 ? A / sqrt(d) : Vec3(0,0,0);
    }

// Return a normalized version of A and set "dist" to the length of
// the original vector.
inline Vec3 Unit( const Vec3 &A, double &dist )
    {
    dist = Length( A );
    return dist > 0.0 ? A / dist : Vec3(0,0,0);
    }

// Return a normalized version of (x,y,z).
inline Vec3 Unit( double x, double y, double z )
    {
    const double d( (x * x) + (y * y) + (z * z) );
    return d > 0.0 ? Vec3(x,y,z) / sqrt(d) : Vec3(0,0,0);
    }

// Return a non-zero vector orthogonal to A.
inline Vec3 OrthogonalTo( const Vec3 &A )
    {
    if( A.x == 0 ) return Vec3( 1, 0, 0 );
    return Vec3( A.y, -A.x, 0 );
    }

// Euclidean distance from A to B.
inline double dist( const Vec3 &A, const Vec3 &B )
    {
    return Length( A - B ); 
    }

inline std::ostream &operator<<( std::ostream &out, const Vec3 &A )
    {
    out << "(" << A.x << ", " << A.y << ", " << A.z << ")";
    return out;
    }

#endif


