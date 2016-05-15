/***************************************************************************
* quartic.cpp                                                              *
*                                                                          *
* Functions for finding the roots of polynomials up through degree four    *
* (quartics).  All root finding is done analytically, and only real roots  *
* are reported.  This software was derived from a C program by             *
* Raoul Rausch, who converted it from FORTRAN in October of 2004.  The     *
* algorithm is based on a paper by J. E. Hacke, published in the American  *
* Mathematical Monthly, Vol. 48, 327-328, 1941.                            *
*                                                                          *
* History:                                                                 *
*   10/11/2005  Rewritten, starting from the 2004 C-code by Raoul Rausch.  *
*                                                                          *
***************************************************************************/
#include "quartic.h"
#include "util.h"
#include "constants.h"

static const double
    one_third( 1.0 / 3.0 ),
    one_sixth( 1.0 / 6.0 ),
    Pi23     ( 2.0 * Pi / 3.0 ),
    Pi43     ( 4.0 * Pi / 3.0 );

static inline double CubeRoot( double Z )
    {
    if( Z > 0.0 ) return  pow(  Z, one_third );
    if( Z < 0.0 ) return -pow( -Z, one_third );
    return 0.0;
    }

bool roots::MaxRoot( double &r ) const
    {
    if( num == 0 ) return false;
    r = r0;
    if( num >= 2 && r < r1 ) r = r1;  
    if( num >= 3 && r < r2 ) r = r2; 
    if( num >= 4 && r < r3 ) r = r3;
    return true;
    }

bool roots::MinPositiveRoot( double &r ) const
    {
    bool okay = false;
    if( num >= 1 && 0.0 <= r0 && r0 < r ) { okay = true; r = r0; } 
    if( num >= 2 && 0.0 <= r1 && r1 < r ) { okay = true; r = r1; }
    if( num >= 3 && 0.0 <= r2 && r2 < r ) { okay = true; r = r2; }      
    if( num >= 4 && 0.0 <= r3 && r3 < r ) { okay = true; r = r3; }
    return okay;
    }

// Find the real roots of the cubic polynomial with the given coefficients,
// where the parameter a0 denotes the constant (zeroth-order) coefficient.
// The return value is the number of real roots found. 
int SolveCubic( double a0, double a1, double a2, double a3, roots &R )
    {
    double r0 = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    double p, q, dis;
    R.num = 0;

    if( a3 != 0.0 ) // Solve the cubic equation.
        { 
        const double over_a3( 1.0 / a3 );
        const double s( -a2 * one_third * over_a3 );
        const double ss( s * s );
        p = a1 * one_third * over_a3 - ss;
        p = p * p * p;
        q = 0.5 * ( 2.0 * ss * s - ( a1 * s + a0 ) * over_a3 );
        dis = q * q + p;
        if( dis < 0.0 )
            {
            R.num = 3; // three real roots.
            double arg = q / sqrt( -p );
            double phi;
            // This can be optimized to remove the trig functions!
            if( arg < -1.0 ) phi = Pi / 3.0;
            else if( arg >  1.0 ) phi = 0.0;
            else phi = acos( arg ) * one_third;
            p = 2.0 * pow( -p, one_sixth );
            r0 = p * cos( phi        ) + s;
            r1 = p * cos( phi + Pi23 ) + s;
            r2 = p * cos( phi + Pi43 ) + s;
            } 
        else
            {
            R.num = 1; // one real root.
            dis = sqrt( dis );
            r0 = CubeRoot( q + dis ) + CubeRoot( q - dis ) + s;
            } 
        } 
    else if( a2 != 0.0 ) // Solve the quadratic equation. 
        {
        const double over_a2( 1.0 / a2 );
        p = 0.5 * a1 * over_a2;
        dis = p * p - a0 * over_a2;
        if( dis > 0.0 )
            {
            R.num = 2;  // Two real roots.
            dis = sqrt( dis );
            r0 = -dis - p;
            r1 =  dis - p;
            } 
        else return 0;
        } 
    else if( a1 != 0.0 ) // Solve the linear equation.
        { 
        R.num = 1; // One real root.
        r0 = a0 / a1;
        } 

    // Clean up the roots with one iteration of Newton's method.
    switch( R.num )
        {
        case 3: R.r2 = r2 - Horner( r2, a0, a1, a2, a3 ) / Horner( r2, a1, 2.0 * a2, 3.0 * a3 );
        case 2: R.r1 = r1 - Horner( r1, a0, a1, a2, a3 ) / Horner( r1, a1, 2.0 * a2, 3.0 * a3 );
        case 1: R.r0 = r0 - Horner( r0, a0, a1, a2, a3 ) / Horner( r0, a1, 2.0 * a2, 3.0 * a3 );
        case 0: break;
        }

    return R.num;
    }

// Find the real roots of the quartic polynomial with the given coefficients,
// where the parameter a0 denotes the constant (zeroth-order) coefficient.
// The return value is the number of real roots found.
int SolveQuartic( double a0, double a1, double a2, double a3, double a4, roots &R )
    {
    roots  rts;
    double maxroot;
    R.num = 0;

    // Handle degenerate cases (i.e. those that are not actual quartics).
    if( a4 == 0.0 ) return SolveCubic( a0, a1, a2, a3, R );

    // Set up and compute the largest root of the cubic resolvent polynomial.
    const double a3_2( a3 * a3 );
    const double a4_2( a4 * a4 );
    const double a4_3( a4 * a4_2 );
    const double a2a4( a2 * a4 );
    const double p( ( 8.0 * a2a4 - 3.0 * a3_2 ) / ( 8.0 * a4_2 ) );
    const double q( ( a3 * ( a3_2 - 4.0 * a2a4 ) + 8.0 * a1 * a4_2 ) / ( 8.0 * a4_3 ) );
    const double r( ( a3 * ( a3 * ( 16.0 * a2a4 - 3.0 * a3_2 ) - 64.0 * a4_2 * a1 ) +
                       256.0 * a4_3 * a0 ) / ( 256.0 * a4_3 * a4 ) );
    SolveCubic( 4.0 * p * r - q * q, -8.0 * r, -4.0 * p, 8.0, rts );
    rts.MaxRoot( maxroot );

    // Use the largest root of the resolvent cubic to find the roots of the quartic.
    const double sh( a3 / ( -4.0 * a4 ) ); // Shift.
    const double c1( 2.0 * maxroot - p );
    const double c2( sqrt( c1 ) );
    const double c3( q / ( 2.0 * c2 ) );
    double       c4( c1 - 4.0 * (maxroot + c3) );
    double       c5( c1 - 4.0 * (maxroot - c3) );

    if( c4 >= 0.0 )
        {
        if( c5 >= 0.0 )
            {
            R.num = 4; // Four real roots.
            c4 = sqrt( c4 );
            c5 = sqrt( c5 );
            R.r0  = sh + 0.5 * ( c2 - c4 );
            R.r1  = sh + 0.5 * (-c2 - c5 );
            R.r2  = sh + 0.5 * ( c2 + c4 );
            R.r3  = sh + 0.5 * (-c2 + c5 );
            }  
        else
            {
            R.num = 2; // Two real roots.
            c4 = sqrt( c4 );
            // Automatically in ascending order.
            R.r0  = sh + 0.5 * ( c2 - c4 );
            R.r1  = sh + 0.5 * ( c2 + c4 );
            }
        }
    else if( c5 >= 0.0 )
        {
        R.num = 2; // Two real roots.
        c5 = sqrt( c5 );
        // Automatically in ascending order.
        R.r0  = sh - 0.5 * ( c2 + c5 );
        R.r1  = sh - 0.5 * ( c2 - c5 );
        }

    return R.num;
    }














		
		


