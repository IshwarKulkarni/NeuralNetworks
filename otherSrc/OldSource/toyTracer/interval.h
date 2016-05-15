/***************************************************************************
* Interval.h                                                               *
*                                                                          *
* The Interval class defines a closed interval on the real line, defined   *
* by a minimum and a maximum value.  Numerous basic operations are defined *
* for intervals, such as expanding them to include a value or another      *
* interval.  Intervals are convenient for defining rectangles (such as the *
* view window) and boxes (such as axis-aligned bounding boxes).            *
*                                                                          *
* History:                                                                 *
*   05/05/2010  Added standard interval arithmetic.                        *
*   12/10/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __INTERVAL_INCLUDED__
#define __INTERVAL_INCLUDED__

#include <iostream>
#include "constants.h"

// An interval of real numbers.  An interval is considered degenerate when
// the minimum value is greater than the maximum value.
struct Interval { 
    inline Interval() : min(Infinity), max(-Infinity) {}
    inline Interval( double a ) : min(a), max(a) {}
    inline Interval( double a, double b ) : min(a), max(b) {}
    inline ~Interval() {}
    inline bool Contains( double x ) const { return ( min <= x ) && ( x <= max ); }
    inline double Length() const { return max > min ? max - min : 0.0; }
    static inline Interval Null() { return Interval( Infinity, -Infinity ); }
    double min;
    double max;
    };

// Is the interval I degenerate?
inline bool IsNull( const Interval &I )
    {    
    return I.max < I.min;
    }

// Expand interval A to include interval B.
inline Interval &operator<<( Interval &A, const Interval &B ) 
    {
    if( B.min < A.min ) A.min = B.min;
    if( B.max > A.max ) A.max = B.max;
    return A;
    };

// Increment the interval by another interval.
inline Interval &operator+=( Interval &A, const Interval &B ) 
    {
    A.min += B.min;
    A.max += B.max;
    return A;
    }

// Decrement the interval by another interval.
inline Interval &operator-=( Interval &A, const Interval &B )
    {
    A.min -= B.max;
    A.max -= B.min;
    return A;
    }

// Return the sum of two intervals, which is another interval consisting
// of all possible sums.
inline Interval operator+( const Interval &A, const Interval &B )
    {
    return Interval( A.min + B.min, A.max + B.max );
    }

// Return the difference of two intervals, which is another interval
// consisting of all possible differences.
inline Interval operator-( const Interval &A, const Interval &B )
    {
    return Interval( A.min - B.max, A.max - B.min );
    }

// Determine whether two intervals overlap.
inline bool operator&( const Interval &A, const Interval &B ) 
    {
    return ( A.max >= B.min && A.min <= B.max );
    }

// Standard interval multiplication.
inline Interval operator*( const Interval &A, const Interval &B )
    {
    const double ab( A.min * B.min );
    const double aB( A.min * B.max );
    const double Ab( A.max * B.min );
    const double AB( A.max * B.max );
    if( ab <= aB && Ab <= AB ) return Interval( ab < Ab ? ab : Ab, aB > AB ? aB : AB );
    if( ab <= aB && AB <= Ab ) return Interval( ab < AB ? ab : AB, aB > Ab ? aB : Ab );
    if( aB <= ab && Ab <= AB ) return Interval( aB < Ab ? aB : Ab, ab > AB ? ab : AB );
                               return Interval( aB < AB ? aB : AB, ab > Ab ? ab : Ab );
    }

// Scale all elements of the interval I by 1/c.
inline Interval operator/( const Interval &I, double c )
    {
    const double a( I.min / c );
    const double b( I.max / c );
    return c >= 0.0 ? Interval( a, b ) : Interval( b, a );
    }

// An output operator, useful for debugging.
inline std::ostream &operator<<( std::ostream &out, const Interval &I )
    {
    out << "[" << I.min << "," << I.max << "]";
    return out;
    }

#endif


