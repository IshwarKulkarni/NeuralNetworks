/***************************************************************************
* cone.cpp   (primitive object plugin)                                     *
*                                                                          *        
* Canonical cone along the Z-axis with unit radius, running from z = 0 to  *
* z = 1.  The cone can be "hollow" or not.  A hollow cone has no end cap   *
* and no interior.                                                         *
*                                                                          *
* History:                                                                 *
*   10/11/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"

namespace __cone_primitive__ {  // Ensure that there are no name collisions.

struct cone : Primitive {
    cone() {}
    cone( bool hollow_ ) { hollow = hollow_; }
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 &P ) const; 
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual int GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "cone"; }
	virtual double Cost() const { return 3.0;}
    bool hollow;
    };

REGISTER_PLUGIN( cone );

Plugin *cone::ReadString( const std::string &params )
    {
    ParamReader get( params );
    if( get[MyName()] ) return new cone( get["hollow"] );
    return NULL;
    }

Interval cone::GetSlab( const Vec3 &v ) const
    {
    const double vv = v * v;
    const double z  = v.z;
    const double r  = sqrt( vv - z * z );
    const double s_max = max( z, r, -r );
    const double s_min = min( z, r, -r );
    return Interval( s_min, s_max ) / vv;
    }

bool cone::Inside( const Vec3 &P ) const
    {
    if( hollow || P.z < 0.0 || P.z > 1.0 ) return false;
    return P.x * P.x + P.y * P.y <= sqr( 1.0 - P.z ); 
    }

bool cone::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // The intersection is computed with a cone whose vertex is at the origin
    // as this simplifies the computation somewhat.  To do this we need only
    // translate the ray origin down by one at the start, then translate the
    // point of intersection up at the end.
    const Vec3 Q( ray.origin - Vec3(0,0,1) );  // Shift the cone down by 1.
    const Vec3 R( ray.direction );
    bool hit = false;
    double z = 0.0;
    double s = Infinity;
    double s1, s2;
    Vec3 P; // The point of intersection with the cone.
    Vec3 N; // The normal vector at the point of intersection.

    // Test for easy (vertical) reject cases.  The shifted cone has its peak
    // at the origin, thus running from 0 to -1 along the Z-axis.
    if( R.z >= 0.0 )
        {
        // Above the cone looking up.
        if( Q.z >= 0.0 ) return false;
        }
    else 
        {
        // Below the cone looking down.
        if( Q.z <= -1.0 ) return false;
        }

    const double a = R.x * R.x + R.y * R.y - R.z * R.z;
    const double b = R.x * Q.x + R.y * Q.y - R.z * Q.z;
    const double c = Q.x * Q.x + Q.y * Q.y - Q.z * Q.z;

    if( fabs(a) < MachEps )
        {
        if( fabs(b) < MachEps ) return false;
        // The equation is actually linear, not quadratic.
        s1 = -c / b;
        s2 = s1;
        }
    else
        {
        const double discr = b * b - a * c;
        if( discr < 0.0 ) return false;
        const double radical = sqrt( discr );
        s1 = ( -b - radical ) / a; // When a > 0, this is the min root.
        s2 = ( -b + radical ) / a; // When a > 0, this is the max root.
        if( a < 0.0 ) { double x = s1; s1 = s2; s2 = x; } // Sort.
        }

    if( s1 >= 0.0 ) // Min root is a potential hit.
        {
        s = s1;
        z = Q.z + s * R.z;
        hit = ( -1.0 <= z && z <= 0.0 );
        }

    if( !hit && s2 >= 0.0 ) // Max root is a potential hit.
        {
        s = s2;
        z = Q.z + s * R.z;
        hit = ( -1.0 <= z && z <= 0.0 );
        }

    if( hit )
        {
        // Fill in the point of intersection and the normal.
        P.x = Q.x + s * R.x;
        P.y = Q.y + s * R.y;
        P.z = z;
        N = Unit( P.x, P.y, sqrt( sqr(P.x) + sqr(P.y) ) ); 
        }

    // Now check the end cap, provided the cone is not hollow and the ray 
    // is not parallel to the cap.

    if( !hollow && R.z != 0.0 ) 
        {
        const double s_cap = -( 1.0 + Q.z ) / R.z;  // Distance to z = -1 plane.
        // Only consider a cap hit if it would be closer than the current hit.
        if( s_cap < s && s_cap > 0.0 )
            {
            const Vec3 C( Q + s_cap * R );
            // If C is in the unit disk, the ray hits an end cap.
            if( C.x * C.x + C.y * C.y <= 1.0 )
                {
                s = s_cap;
                P = C;
                N = Vec3( 0, 0, -1.0 );
                hit = true;
                }
            } 
        }

    if( !hit || s >= hitinfo.distance ) return false;

    P.z = P.z + 1.0;  // Undo the earlier translation.

    hitinfo.distance = s;
    hitinfo.point    = P;
    hitinfo.normal   = N;
    hitinfo.object   = this;
    return true;
    }

int cone::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const
    {
    // To be supplied...
    return 0;
    }

} // namespace __cone_primitive__







