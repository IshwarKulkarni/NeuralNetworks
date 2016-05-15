/***************************************************************************
* cylinder.cpp   (primitive object plugin)                                 *
*                                                                          *        
* Canonical cylinder along the Z-axis with unit radius, running from       *
* z = -1 to z = 1.  The cylinder can be "hollow" or not.  A hollow         *
* cylinder has no end caps and no interior.                                *
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

namespace __cylinder_primitive__ {  // Ensure that there are no name collisions.

struct cylinder : Primitive {
    cylinder() {}
    cylinder( bool hollow_ ) { hollow = hollow_; }
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 &P ) const; 
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual int GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "cylinder"; }
	virtual double Cost() const { return 3.0;}
    bool hollow;
    };

REGISTER_PLUGIN( cylinder );

Plugin *cylinder::ReadString( const std::string &params )
    {
    ParamReader get( params );
    if( get[MyName()] ) return new cylinder( get[ "hollow" ] );
    return NULL;
    }

Interval cylinder::GetSlab( const Vec3 &v ) const
    {
    double dot = fabs( v.z ) + sqrt( v.x * v.x + v.y * v.y );
    return Interval( -dot, dot ) / ( v * v );
    }

bool cylinder::Inside( const Vec3 &P ) const
    {
    if( hollow ) return false;
    if( P.z < -1.0 || P.z > 1.0 ) return false;
    return P.x * P.x + P.y * P.y <= 1.0; 
    }

bool cylinder::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    double s = Infinity;
    const Vec3 Q( ray.origin );
    const Vec3 R( ray.direction );
    Vec3 P; // The point of intersection with the cylinder.
    Vec3 N; // The normal vector at the point of intersection.

    // Test for easy (vertical) reject cases.
    if( R.z >= 0.0 )
        {
        // Above the cylinder looking up.
        if( Q.z >= 1.0 ) return false;
        }
    else 
        {
        // Below the cylinder looking down.
        if( Q.z <= -1.0 ) return false;
        }

    // Compute some values that will be used for both fast reject tests and
    // for computing intersections with the infinite cylinder.

    const double b = R.x * Q.x + R.y * Q.y;  // The 2 is factored out.
    const double c = Q.x * Q.x + Q.y * Q.y;

    // Test another easy (horizontal) reject case.
    if( c >= 1.0 && b > 0 ) return false; 

    const double a = R.x * R.x + R.y * R.y;

    // See if the ray hits the infinite unit-radius cylinder along the Z-axis.
    // The ray can only hit the walls of the cylinder if it is not parallel
    // to the Z-axis.
    if( a != 0.0 )  
        {
        double discr = b * b - a * ( c - 1.0 );
        if( discr < 0.0 ) return false;            // Both roots are complex; the ray misses.
        discr = sqrt( discr );
        const double s_max = ( -b + discr ) / a;   // The biggest root.
        if( s_max <= 0.0 ) return false;           // Neither root is positive.
        const double s_min = ( -b - discr ) / a;   // The smallest root.

        if( s_min > 0.0 ) // Both roots are positive.
            {
            const Vec3 Near( Q + s_min * R );
            const Vec3 Far ( Q + s_max * R );  
            if( -1.0 <= Near.z && Near.z <= 1.0 )
                {
                // The close hit with the infinite cylinder is valid.
                s = s_min;
                P = Near;
                N = Unit( P.x, P.y, 0.0 );
                }             
            else if( -1.0 <= Far.z && Far.z <= 1.0 )
                {
                // The far hit with the infinite cylinder is valid.
                s = s_max;
                P = Far;
                N = Unit( P.x, P.y, 0.0 );
                }
            else if( Near.z * Far.z > 0.0 )
                {
                // Both hits are out of range, and on the SAME side of the cylinder
                // (i.e. both above or both below), therefore, the ray cannot hit
                // the finite cylinder or the caps.
                return false;
                }
            }
        else // Only the s_max root is positive.
            {
            const Vec3 Far( Q + s_max * R );
            if( -1.0 <= Far.z && Far.z <= 1.0 )
                {
                // The far hit with the infinite cylinder is valid.
                s = s_max;
                P = Far;
                N = Unit( P.x, P.y, 0.0 );
                }
            }
        }
       
    // Now check the end caps, provided the cylinder is not hollow and the ray 
    // is not parallel to the caps.

    if( !hollow && R.z != 0.0 ) 
        {
        const double t1    = (  1.0 - Q.z ) / R.z;  // Distance to z =  1 plane.
        const double t2    = ( -1.0 - Q.z ) / R.z;  // Distance to z = -1 plane.
        const double s1    = min( t1, t2 );
        const double s2    = max( t1, t2 );
        const double s_cap = ( s1 > 0.0 ) ? s1 : s2; 

        // Only consider a cap hit if it would be closer than the current hit.
        if( s_cap < s && s_cap > 0.0 )
            {
            Vec3 C = Q + s_cap * R;
            // If C is in the unit disk, the ray hits an end cap.
            if( C.x * C.x + C.y * C.y <= 1.0 )
                {
                s = s_cap;
                P = C;
                N = Vec3( 0, 0, C.z > 0.0 ? 1.0 : -1.0 );
                }
            // No need to consider another potential cap hit.  If the closest one
            // is outside the unit disk, and the other is inside the unit disk,
            // then they ray must hit the cylinder wall, which we've already
            // accounted for.
            } 
        } 

    if( s >= hitinfo.distance ) return false;

    // We have an actual hit.  Fill in all the geometric information so
    // that the shader can shade this point.

    hitinfo.distance = s;
    hitinfo.point    = P;
    hitinfo.normal   = N;
    hitinfo.object   = this;
    return true;
    }

int cylinder::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const
    {
    // To be supplied...
    return 0;
    }


} // namespace __cylinder_primitive__

