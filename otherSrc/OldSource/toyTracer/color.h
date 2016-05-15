/***************************************************************************
* color.h                                                                  *
*                                                                          *
* Color is a trivial encapsulation of floating-point RGB colors.  It has   *
* the obvious operators defined as inline functions.  Note that there is   *
* no subtraction operator for colors, and that multiplication is performed *
* component-wise and results in another color.                             *
*                                                                          *
* Color components are stored as floats rather than doubles because they   *
* will typically undergo a mapping down to 256 values, and floats provide  *
* more than adequate precision.  Also, colors may be written to a file as  *
* part of a high-dynamic-range image, and double precision would require   *
* excessive storage.                                                       *
*                                                                          *
* History:                                                                 *
*   04/04/2010  Moved Raster struct to toytracer.h file.                   *
*   04/12/2008  Added Raster for high dynamic range.                       *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __COLOR_INCLUDED__
#define __COLOR_INCLUDED__

#include <cmath>
#include <iostream>

struct Color {
    inline Color() : red(0.0f), green(0.0f), blue(0.0f) {}
    inline Color( double r, double g, double b ) : red(float(r)), green(float(g)), blue(float(b)) {}
    inline ~Color() {}
    inline Color &Clamp( double r_max, double g_max, double b_max );
    float red;
    float green;
    float blue;
    // Define a few basic colors.
    static const Color
        White, Gray, Black,
        Red, Green, Blue,
        Yellow, Magenta, Cyan;
    };

// Clamp each color channel to the given upper bound.
inline Color &Color::Clamp( double r_max, double g_max, double b_max )
    {
    if( red   > r_max ) red   = float(r_max);
    if( green > g_max ) green = float(g_max);
    if( blue  > b_max ) blue  = float(b_max);
    return *this;
    }

inline Color operator+( const Color &A, const Color &B )
    {
    return Color( A.red + B.red, A.green + B.green, A.blue + B.blue );
    }

inline Color operator-( const Color &A, const Color &B )
    {
    return Color( A.red - B.red, A.green - B.green, A.blue - B.blue );
    }

inline Color operator*( double c, const Color &A )
    {
    return Color( c * A.red, c * A.green, c * A.blue );
    }

inline Color operator*( const Color &A, double c )
    {
    return Color( c * A.red, c * A.green, c * A.blue );
    }

// Colors are multiplied component-wise, and result in another color, not
// a scalar.  This is the most significant difference between the Vec3 class
// (where multiplication means dot product) and the Color class.
inline Color operator*( const Color &A, const Color &B )
    {
    return Color( A.red * B.red, A.green * B.green, A.blue * B.blue );
    }

inline Color operator/( const Color &A, double c )
    {
    return Color( A.red / c, A.green / c, A.blue / c );
    }

inline Color& operator+=( Color &A, const Color &B )
    {
    A.red   += B.red;
    A.green += B.green;
    A.blue  += B.blue;
    return A;
    }

inline Color& operator*=( Color &A, double c )
    {
    A.red   *= float(c);
    A.green *= float(c);
    A.blue  *= float(c);
    return A;
    }

inline Color& operator/=( Color &A, double c )
    {
    A.red   /= float(c);
    A.green /= float(c);
    A.blue  /= float(c);
    return A;
    }

inline Color fabs( const Color &C )
    {
    return Color( fabs(C.red), fabs(C.green), fabs(C.blue) );
    }

inline bool operator==( const Color &A, const Color &B )
    {
    return (A.red == B.red) && (A.green == B.green) && (A.blue == B.blue);
    }

inline bool operator!=( const Color &A, const Color &B )
    {
    return (A.red != B.red) || (A.green != B.green) || (A.blue != B.blue);
    }

inline bool operator==( const Color &A, double c )
    {
    return (A.red == c) && (A.green == c) && (A.blue == c);
    }

inline bool operator!=( const Color &A, double c )
    {
    return !( A == c );
    }

inline std::ostream &operator<<( std::ostream &out, const Color &C )
    {
    out << "[" << C.red << ", " << C.green << ", " << C.blue << "]";
    return out;
    }

#endif

