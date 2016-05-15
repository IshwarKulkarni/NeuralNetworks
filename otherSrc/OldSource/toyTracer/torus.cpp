/***************************************************************************
* torus.cpp   (primitive object plugin)                                    *
*                                                                          *
* The "torus" primitive object is situated on the x-y plane.  It has two   *
* parameters: the major radius (denoted by "a") and the minor radius       *
* (denoted by "b").  The ray-torus intersection test is performed          *
* analytically using a closed-form quartic polynomial root finder.         *
*                                                                          *
* History:                                                                 *
*   04/30/2010  Fixed bug in slab computation (when v is unnormalized).    *
*   04/28/2010  Added annulus test for visibility rays.                    *
*   10/12/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <cmath>
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "quartic.h"
#include "constants.h"

using std::string;

namespace __torus_primitive__ {  // Ensure that there are no name collisions.

struct torus : Primitive {
    torus() {}
    torus( double major_radius, double minor_radius );
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 &P ) const; 
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "torus"; }
    virtual double Cost() const { return 4.0; }
    double a;    // Major radius.
    double b;    // Minor radius.
    double a2b2; // a * a + b * b
    double rad;  // a + b
    double rad2; // rad * rad 
    AABB   bbox;
    };

REGISTER_PLUGIN( torus );


torus::torus( double major_radius, double minor_radius )
    {
    // Pre-compute the often-used values that depend only on the parameters.
    a    = major_radius;
    b    = minor_radius;
    a2b2 = a * a + b * b;
    rad  = a + b;
    rad2 = rad * rad;
    bbox = AABB( Interval( -rad, rad ), Interval( -rad, rad ), Interval( -b, b ) );
    }

Plugin *torus::ReadString( const string &params ) // Reads params from a string.
    {
    double ra, rb;
    ParamReader get( params );
    if( get[MyName()] && get[ra] && get[rb] ) return new torus( ra, rb );
    return NULL;
    }

// Compute bounding interval of the torus along the v axis, where v may
// not be normalized.  Since the torus is always centered at the origin, the
// computed interval will always be symmetric about the origin.  Here theta
// is the angle between v and the z-axis.
Interval torus::GetSlab( const Vec3 &v ) const
    {
    const double vv( v * v );
    const double cos_theta_squared( v.z * v.z / vv );
    const double sin_theta( sqrt( 1.0 - cos_theta_squared ) );
    const double len( ( a * sin_theta + b ) / sqrt( vv ) );
    return Interval( -len, len );
    }

bool torus::Inside( const Vec3 &P ) const
    {
    // Quick up-down reject test.
    if( P.z > b || P.z < -b ) return false;
    const double r2( P.x * P.x + P.y * P.y );
    // Quick radial reject test.
    if( r2 > rad2 || r2 < ( a - b ) * ( a - b ) ) return false;
    const double scale( a / sqrt(r2) );
    const Vec3 Q( scale * P.x, scale * P.y, 0.0 );
    return dist( P, Q ) <= b;
    }

bool torus::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // THINGS TO DO: Should reposition the ray origin to the intersection
    // with the bounding box.  This will increase the accuracy of the root
    // finder.
    roots rts;
    const Vec3 &R( ray.direction );
    const Vec3 &Q( ray.origin    );

    // First check the easy reject cases using the bounding planes on
    // the top and bottom: z = b and z = -b.
    if( R.z >= 0.0 )
        {
        // Above the torus looking up.
        if( Q.z >=  b ) return false;
        }
    else // R.z < 0
        {
        // Below the torus looking down.
        if( Q.z <= -b ) return false;
        }

    // If we are testing for visibility only, see if the ray hits the annulus on
    // the z=0 plane.  If so, we can return early.  Rule out the case where the
    // ray origin is inside the torus by requiring the distance to the z=0 plane
    // to be at least b, the minor radius.
    if( ray.type == visibility_ray && R.z != 0.0 && fabs(Q.z) > b )
        {
        const double s( -Q.z / R.z ); // Distance to z=0 plane.
        if( 0.0 < s && s < hitinfo.distance )
            {
            // See if the point on the z=0 plane falls within the annulus.
            const double r2( LengthSquared( Q + s * R ) );
            if( rad2 > r2 && r2 > ( a - b ) * ( a - b ) )
                {
                // The visibility ray hits the annulus, so we're done.
                hitinfo.object = this;
                return true;
                }
            }
        }

    // See if the ray trivially misses the infinite bounding cylinder.
    // If not, the results of this computation will be used for root finding.
    const double qq( Q.x * Q.x + Q.y * Q.y );
    const double qr( Q.x * R.x + Q.y * R.y );
    if( qr > 0.0 && qq >= rad2 ) return false;

    // Perform a bounding-box test.
    if( !bbox.Hit( ray, hitinfo.distance ) ) return false;

    const double QQ( qq + Q.z * Q.z );
    const double QR( qr + Q.z * R.z );
    const double a2( a * a );

    // Set up and solve the quartic polynomial that results from plugging
    // the ray equation into the implicit equation of the torus.  That is,
    //
    //   || P  -  a Q(P) || = b    (implicit equation of torus.)
    //
    //    P  =  Q + R s            (parametric form of a ray.)
    //
    // where Q(P) is the closest point to P that is on the unit-radius circle,
    // centered at the origin, in the x-y plane.  The function Q is given by
    //
    //    Q(P)  =  ( Px, Py, 0 ) / sqrt( Px Px  +  Py Py )
    //
    // When the radicals are removed via squaring, the result is a quartic
    // equation in s, the distance along the ray to the point of intersection.
    //
    SolveQuartic(
        sqr( QQ - a2b2 ) - 4.0 * a2 * ( b * b - Q.z * Q.z ),      // Constant
        4.0 * QR * ( QQ - a2b2 ) + 8.0 * a2 * Q.z * R.z,          // Linear
        2.0 * ( QQ - a2b2 ) + 4.0 * ( QR * QR + a2 * R.z * R.z ), // Quadratic
        4.0 * QR,                                                 // Cubic
        1.0,                                                      // Quartic
        rts // The real roots returned by the solver.
        );

    // If the smallest positive root is larger than the distance already
    // in hitinfo, then we do not consider it a hit.
    double s( hitinfo.distance );
    if( !rts.MinPositiveRoot( s ) ) return false;

    // If we are simply testing for visibility, there is no need to compute
    // the point P or the normal.
    hitinfo.object = this;
    if( ray.type == visibility_ray ) return true;

    // Compute the actual point of intersection using the distance.
    const Vec3 P( ray.origin + s * R );

    hitinfo.distance = s;
    hitinfo.point    = P;
    hitinfo.normal   = Unit( P - a * Unit( P.x, P.y, 0.0 ) ); 
    return true;
    }

} // namespace __torus_primitive__



