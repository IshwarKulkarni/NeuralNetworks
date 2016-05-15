/***************************************************************************
* supertoroid.cpp   (container plugin)                                     *
*                                                                          *
* Supertoroids are a class of shapes that are generalizations of the       *
* torus.  By adjusting three different exponents we can obtain toroids     *
* with varying degrees of roundedness.  Since the exponents can be         *
* arbitrary, ray intersections cannot be handled analytically.  Therefore, *
* the entire class of shapes is handled by tessellation into triangles,    *
* making the supertoroid class a sub-class of Container, not Object        *
* (similar to bezier).                                                     *
*                                                                          *
* See http://en.wikipedia.org/wiki/Supertoroid for more detail on          *
* supertoroids.                                                            *
*                                                                          *
* This object requires the "triangle" plug-in.                             *
*                                                                          *
* History:                                                                 *
*   05/11/2010  Moved triangle generation into Tessellation base class.    *
*   05/08/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "params.h"
#include "mat3x4.h"
#include "tessellation.h"

using std::string;

namespace __supertoroid_container__ {

struct supertoroid : Tessellation {
    supertoroid() {}
    virtual ~supertoroid() {}
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "supertoroid"; }
    virtual void Eval( double u, double v, Vec3 &P, Vec3 &N ) const;
    double a1, b1;
    double a2, b2;
    double s, t;
    };

REGISTER_PLUGIN( supertoroid );

// Look for a text line that specifies an instance of a supertoroid,
// which has the form
//
// supertoroid  (a1,b1) (a2,b2)  s  t  [with-normals]  [without-normals]  [steps nu nv]
//
Plugin *supertoroid::ReadString( const string &params )
    {
    Vec2 A, B;
    ParamReader get( params );
    if( get[MyName()] && get[A] && get[B] && get[s] && get[t] )
        {
        // Set the required parameters.
        a1 = A.x;
        b1 = A.y;
        a2 = B.x;
        b2 = B.y;
        // Set the defaults, which may be reset by parameters.
        bool with_normals = true;
        unsigned nu = 16;
        unsigned nv = 16;
        
        // Loop through the parameters, processing what we recognize.
        // Complain if there is anything left that is not recognized.
        while( !get.isEmpty() )
            {
            if( get["with-normals"] ) { with_normals = true; continue; }
            if( get["without-normals"] ) { with_normals = false; continue; }
            if( get["steps"] && get[nu] && get[nv] ) { continue; }
            get.Warning( "unprocessed parameters to " + MyName() );
            break;
            }
        // Make a new instance of the supertoroid by copying the parameters
        // of this one.
        supertoroid *superq( new supertoroid( *this ) );
        superq->EnableSymmetry( xyz_symmetry );
        superq->EnableNormals ( with_normals );
        superq->Tessellate(  Interval( 0, Pi ), nu, Interval( 0, Pi/2 ), nv );
        return superq;
        }
    return NULL;
    }

// Create a more robust version of the "pow" function that looks for
// zero raised to a negative power, which happens frequently in evaluating
// supertoroids.  Interpret this to be a very large number.
inline double power( double x, double p )
    {
    if( x == 0 && p <= 0.0 ) return 1.0E10;
    return pow( x, p );
    }

inline double sign( double x )
    { 
    return x > 0.0 ? 1.0 : ( x < 0.0 ? -1.0 : 0.0 );
    }

// Evaluate the supertoroid function at the parameters (u,v) to produce the
// point P(u,v).  Also compute the normal N(u,v) at that point.  Due to 3-way
// symmetry, this function is only intended for the positive octant.  All other
// points and normals can be obtained by appropriate reflections.
void supertoroid::Eval( double u, double v, Vec3 &P, Vec3 &N ) const
    {
    Vec3 U, V;
    const double s2( 2.0 / s );
    const double t2( 2.0 / t );
    const double cos_u( cos(u) );
    const double sin_u( sin(u) );
    const double cos_v( cos(v) );
    const double sin_v( sin(v) );
    
    // Compute the point on the surface.
    const double cos_u_s2( sign(cos_u) * power( fabs(cos_u), s2 ) );
    P.x = ( a1 + b1 * cos_u_s2 ) * power( cos_v, t2 );
    P.y = ( a2 + b2 * cos_u_s2 ) * power( sin_v, t2 );
    P.z = power( sin_u, s2 );

    // Compute the partial with respect to u.
    const double sin_u_pow( s2 * sin_u * power( fabs(cos_u), s2 - 1.0 ) );
    U.x = -b1 * sin_u_pow * power( cos_v, t2 );
    U.y = -b2 * sin_u_pow * power( sin_v, t2 );
    U.z =  s2 * cos_u * power( sin_u, s2 - 1.0 );

    // Compute the partial with respect to v.
    V.x = -( a1 + b1 * cos_u_s2 ) * t2 * sin_v * power( cos_v, t2 - 1.0 );
    V.y =  ( a2 + b2 * cos_u_s2 ) * t2 * cos_v * power( sin_v, t2 - 1.0 );
    V.z =  0.0;

    // The normal is the cross product of the partials.
    N = Unit( U ^ V );
    }

} // namespace __supertoroid_container__





