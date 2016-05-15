/***************************************************************************
* superquadric.cpp   (container plugin)                                    *
*                                                                          *
* Superquadrics are a class of shapes that are generalizations of the      *
* sphere.  By adjusting three different exponents we can obtain shapes     *
* resemble ellipoids, cylinders, and cubes, but with varying degrees       *
* of roundedness.  Since the exponents can be arbitrary, ray intersections *
* cannot be handled analytically.  Therefore, the entire class of shapes   *
* is handled by tessellation into triangles, making the superquadric       *
* class a sub-class of Container, not Object (similar to bezier).          *
*                                                                          *
* See http://en.wikipedia.org/wiki/Superquadrics for more detail on        *
* superquadrics.                                                           *
*                                                                          *
* This object requires the "triangle" plug-in.                             *
*                                                                          *
* History:                                                                 *
*   05/11/2010  Moved triangle generation into Tessellation base class.    *
*   05/07/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "params.h"
#include "mat3x4.h"
#include "tessellation.h"

using std::string;

namespace __superquadric_container__ {

struct superquadric : Tessellation {
    superquadric() {}
    virtual ~superquadric() {}
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "superquadric"; }
    virtual void Eval( double u, double v, Vec3 &P, Vec3 &N ) const;
    double r, s, t;
    unsigned nu, nv;
    };

REGISTER_PLUGIN( superquadric );

// Look for a text line that specifies an instance of a superquadric,
// which has the form
//
// superquadric  r  s  t  [with-normals]  [without-normals]  [steps nu nv]
//
Plugin *superquadric::ReadString( const string &params )
    {
    static Interval interval( 0.0, Pi/2.0 );
    ParamReader get( params );
    if( get[MyName()] && get[r] && get[s] && get[t] )
        {
        // Set the defaults, which may be reset by parameters.
        bool with_normals = true;
        nu = 8;
        nv = 12;
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
        // Make a new instance of the superquadric by copying the parameters
        // of this one.
        superquadric *superq( new superquadric( *this ) );
        // Generate all the triangles, using 3-way symmetry.
        superq->EnableSymmetry( xyz_symmetry );
        superq->EnableNormals ( with_normals );
        superq->Tessellate( interval, nu, interval, nv );
        return superq;
        }
    return NULL;
    }

// Create a more robust version of the "pow" function that looks for
// zero raised to a negative power, which happens frequently in evaluating
// superquadrics.  Interpret this to be a very large number.
inline double power( double x, double p )
    {
    if( x == 0 && p <= 0.0 ) return 1.0E10;
    return pow( x, p );
    }

// Evaluate the superquadric function at the parameters (u,v) to produce the
// point P(u,v).  Also compute the normal N(u,v) at that point.  Due to 3-way
// symmetry, this function is only intended for the positive octant.  All other
// points and normals can be obtained by appropriate reflections.
void superquadric::Eval( double u, double v, Vec3 &P, Vec3 &N ) const
    {
    Vec3 U, V;
    const double r2( 2.0 / r );
    const double s2( 2.0 / s );
    const double t2( 2.0 / t );
    const double cos_u( cos(u) );
    const double sin_u( sin(u) );
    const double cos_v( cos(v) );
    const double sin_v( sin(v) );
    
    // Compute the point on the surface.
    P.x = power( cos_v, r2 ) * power( cos_u, r2 );
    P.y = power( cos_v, s2 ) * power( sin_u, s2 );
    P.z = power( sin_v, t2 );

    // Compute the partial with respect to u.
    U.x = power( cos_v, r2 ) * ( -r2 * sin_u * power( cos_u, r2 - 1.0 ) );
    U.y = power( cos_v, s2 ) * (  s2 * cos_u * power( sin_u, s2 - 1.0 ) );
    U.z = 0.0;

    // Compute the partial with respect to v.
    V.x = ( -r2 * sin_v * power( cos_v, r2 - 1.0 ) ) * power( cos_u, r2 );
    V.y = ( -s2 * sin_v * power( cos_v, s2 - 1.0 ) ) * power( sin_u, s2 );
    V.z = (  t2 * cos_v * power( sin_v, t2 - 1.0 ) );

    // The normal is the cross product of the partials.
    N = Unit( U ^ V );
    }


} // namespace __superquadric_container__





