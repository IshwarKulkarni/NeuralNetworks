/***************************************************************************
* Vec2.h                                                                   *
*                                                                          *
* Vec2 is a trivial encapsulation of 2D floating-point coordinates.        *
* It has all of the obvious operators defined as inline functions.         *
*                                                                          *
* History:                                                                 *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __VEC2_INCLUDED__
#define __VEC2_INCLUDED__

#include <cmath>
#include <iostream>

struct Vec2 {
    inline Vec2() : x(0.), y(0.) {}
    inline Vec2( double a, double b ) : x(a), y(b) {}
    inline ~Vec2() {}
    inline void Zero() { x = 0.0; y = 0.0; }
    double x;
    double y;
    };

inline double LengthSquared( const Vec2 &A )
    {
    return A.x * A.x + A.y * A.y;
    }

inline double Length( const Vec2 &A )
    {
    return sqrt( LengthSquared( A ) );
    }

inline Vec2 operator+( const Vec2 &A, const Vec2 &B )
    {
    return Vec2( A.x + B.x, A.y + B.y );
    }

inline Vec2 operator-( const Vec2 &A, const Vec2 &B )
    {
    return Vec2( A.x - B.x, A.y - B.y );
    }

// Unary minus operator.
inline Vec2 operator-( const Vec2 &A )
    {
    return Vec2( -A.x, -A.y );
    }

inline Vec2 operator*( double a, const Vec2 &A )
    {
    return Vec2( a * A.x, a * A.y );
    }

inline Vec2 operator*( const Vec2 &A, double a )
    {
    return Vec2( a * A.x, a * A.y );
    }

// Inner product of A and B.
inline double operator*( const Vec2 &A, const Vec2 &B )
    {
    return (A.x * B.x) + (A.y * B.y);
    }

inline Vec2 operator/( const Vec2 &A, double c )
    {
    return Vec2( A.x / c, A.y / c );
    }

// Pad A & B with a zero z component, then return the z-component
// of the cross product of the resulting 3D vectors.
inline double operator^( const Vec2 &A, const Vec2 &B )
    {
    return A.x * B.y - A.y * B.x;
    }

inline Vec2 & operator+=( Vec2 &A, const Vec2 &B )
    {
    A.x += B.x;
    A.y += B.y;
    return A;
    }

inline Vec2 & operator-=( Vec2 &A, const Vec2 &B )
    {
    A.x -= B.x;
    A.y -= B.y;
    return A;
    }

inline Vec2 & operator*=( Vec2 &A, double a )
    {
    A.x *= a;
    A.y *= a;
    return A;
    }

inline Vec2 & operator/=( Vec2 &A, double a )
    {
    A.x /= a;
    A.y /= a;
    return A;
    }

// Remove component of A that is parallel to B.
inline Vec2 operator/( const Vec2 &A, const Vec2 &B )
    {
    const double x( LengthSquared( B ) );
    return ( x > 0.0 ) ? ( A - (( A * B ) / x) * B ) : A;
    }

// Return a normalized version of the vector A.
inline Vec2 Unit( const Vec2 &A )
    {
    const double d( LengthSquared( A ) );
    return d > 0.0 ? A / sqrt(d) : Vec2(0,0);
    }

// Return a normalized version of the vector (x,y).
inline Vec2 Unit( double x, double y )
    {
    const double d( (x * x) + (y * y) );
    return d > 0.0 ? Vec2(x,y) / sqrt(d) : Vec2(0,0);
    }

// Euclidean distance from A to B.
inline double dist( const Vec2 &A, const Vec2 &B )
    {
    return Length( A - B ); 
    }

inline std::ostream & operator<<( std::ostream &out, const Vec2 &A )
    {
    out << "(" << A.x << ", " << A.y << ")";
    return out;
    }

#endif

