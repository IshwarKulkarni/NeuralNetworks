#ifndef _IMAGERGB_INCLUDED_
#define _IMAGERGB_INCLUDED_

#include "utils/Utils.hxx"
#include "utils/Logging.hxx"

#define COLORCLAMP(V)  (uchar)( (V)> 255 ? 255 : ( (V) < 0 ?  0 : (V) ) ) 

#ifdef __CUDA_ARCH__
#define DEVICE_AND_HOST __forceinline__ __host__ __device__
#else 
#define DEVICE_AND_HOST inline
#endif

typedef unsigned char uchar;
namespace __IM_Color__ {
struct Color
{
    uchar R,G,B;
    DEVICE_AND_HOST Color() {};
    DEVICE_AND_HOST Color(uchar r, uchar g, uchar b): R(r), G(g), B(b) {}    
    
    DEVICE_AND_HOST Color(uint pixel)  // ARGB
    {
        R = ((uchar*)(&pixel))[2];
        G = ((uchar*)(&pixel))[1];
        B = ((uchar*)(&pixel))[0];
    }

    DEVICE_AND_HOST
    Color& operator=(const Color& RHS)
    {
        R = RHS.R, G = RHS.G, B = RHS.B;
        return *this;
    }

    //template<typename Arr>
    //Color& operator=(const Arr& RHS)
    //{
    //    R = (uchar)RHS[0], G = (uchar)RHS[1], B = (uchar)RHS[2];
    //    return *this;
    //}


    DEVICE_AND_HOST
    operator uint() const {  // RGB
        return( (uint(R)<<16) | 
                (uint(G)<<8)  | 
                (uint(B))     );
    }

};    

DEVICE_AND_HOST bool operator<(const Color& color1, const Color& color2)
{
    // for compilation alone.
    return false;
}

DEVICE_AND_HOST bool operator>(const Color& color1, const Color& color2)
{
    // for compilation alone.
    return false;
}


DEVICE_AND_HOST  Color operator+( const Color &A, const Color &B )
{
    return Color( A.R + B.R, A.G + B.G, A.B + B.B );
}

DEVICE_AND_HOST  Color operator-( const Color &A, const Color &B )
{
    return Color( A.R - B.R, A.G - B.G, A.B - B.B );
}

DEVICE_AND_HOST  Color operator*( float c, const Color &A )
{
    return Color((uchar) (c * A.R), (uchar)(c * A.G), (uchar)(c * A.B) );
}

DEVICE_AND_HOST  Color operator*( const Color &A, float c )
{
    return Color((uchar) (c * A.R), (uchar)(c * A.G), (uchar)(c * A.B) );
}

DEVICE_AND_HOST  Color operator*( const Color &A, const Color &B )
{
    return Color( A.R * B.R, A.G * B.G, A.B * B.B );
}

DEVICE_AND_HOST  Color operator/( const Color &A, float c )
{
    return Color((uchar) (A.R/c), (uchar)(A.G/c), (uchar)(A.B/c) );
}

DEVICE_AND_HOST  Color& operator+=( Color &A, const Color &B )
{
    A.R += B.R;
    A.G += B.G;
    A.B += B.B;
    return A;
}

DEVICE_AND_HOST  Color& operator*=( Color &A, float c )
{
    A.R =  COLORCLAMP(A.R*c);
    A.G =  COLORCLAMP(A.G*c);
    A.B =  COLORCLAMP(A.B*c);
    return A;
}

DEVICE_AND_HOST  Color& operator/=( Color &A, float c )
{
    A.R =  COLORCLAMP(A.R/c);
    A.G =  COLORCLAMP(A.G/c);
    A.B =  COLORCLAMP(A.B/c);
    return A;
}

DEVICE_AND_HOST  bool operator==( const Color &A, const Color &B )
{
    return (A.R == B.R) && (A.G == B.G) && (A.B == B.B);
}

DEVICE_AND_HOST  bool operator!=( const Color &A, const Color &B )
{
    return (A.R != B.R) || (A.G != B.G) || (A.B != B.B);
}

DEVICE_AND_HOST  bool operator==( const Color &A, float c )
{
    return (A.R == c) && (A.G == c) && (A.B == c);
}

DEVICE_AND_HOST  bool operator!=( const Color &A, float c )
{
    return !( A == c );
}

static const Color Black  (0,0,0);
static const Color Red    (255,0,0);
static const Color Green  (0,255,0);
static const Color Blue   (0,0,255);
static const Color Yellow (255,255,0);
static const Color Magenta(255,0,255);
static const Color Cyan   (0,255,255);
static const Color White  (255,255,255);

static const Color Gray    (128,128,128);
static const Color Orange  (255,128,0);
static const Color Puple   (128,0,255);

typedef std::vector<Color> ColorVector;
}
#endif