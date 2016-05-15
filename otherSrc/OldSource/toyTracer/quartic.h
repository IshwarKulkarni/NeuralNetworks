/***************************************************************************
* quartic.h                                                                *
*                                                                          *
* Functions for evaluating and finding the roots of polynomials up through *
* degree four (quartics).  All root finding is done analytically, and only *
* real roots are reported.  This software was derived from a C program     *
* by Raoul Rausch, who converted it from FORTRAN in October of 2004.  The  *
* algorithm is based on a paper by J. E. Hacke, published in the American  *
* Mathematical Monthly, Vol. 48, 327-328, 1941.                            *
*                                                                          *
* History:                                                                 *
*   10/11/2005  Rewritten, starting from the 2004 C-code by Raoul Rausch.  *
*                                                                          *
***************************************************************************/
#ifndef __QUARTIC_INCLUDED__
#define __QUARTIC_INCLUDED__

// The "roots" structure is used for reporting the real roots that were
// found by any of the root-finders.  The "num" field indicates how many
// real roots were found. 
struct roots
    {
    int num;
    double r0;
    double r1;
    double r2;
    double r3;
    bool MaxRoot( double &x ) const;
    bool MinPositiveRoot( double &x ) const;
    };

// Find the real roots of the cubic polynomial with the given coefficients.
// The return value is the number of real roots found.
extern int SolveCubic(
    double a0, // Constant coefficient.
    double a1, // Linear coefficient.
    double a2, // Quadratic coefficient.
    double a3, // Cubic coefficient.
    roots &R
    );

// Find the real roots of the quartic polynomial with the given coefficients.
// The return value is the number of real roots found.
extern int SolveQuartic(
    double a0, // Constant coefficient.
    double a1, // Linear coefficient.
    double a2, // Quadratic coefficient.
    double a3, // Cubic coefficient.
    double a4, // Quartic coefficient.
    roots &R
    );

// Evaluate a quadratic polynomial at x using Horner's rule.
inline double Horner( double x, double a, double b, double c )
    {
    return a + x * ( b + x * c );
    }

// Evaluate a cubic polynomial at x using Horner's rule.
inline double Horner( double x, double a, double b, double c, double d )
    {
    return a + x * ( b + x * ( c + x * d ) );
    }

// Evaluate a quartic polynomial at x using Horner's rule.
inline double Horner( double x, double a, double b, double c, double d, double e )
    {
    return a + x * ( b + x * ( c + x * ( d + x * e ) ) );
    }

#endif

