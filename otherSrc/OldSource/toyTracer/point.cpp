/***************************************************************************
* point.cpp   (primitive object plugin)                                    *
*                                                                          *
* The point object is not a real object in that a ray can never hit it, by *
* definition.  It is merely used as a way to encode point light sources.   *
* Hence, a point object with non-zero emission is a point light source.    *
*                                                                          *
* History:                                                                 *
*   10/10/2004  Broken out of objects.C file.                              *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "params.h"

using std::string;

namespace __point_primitive__ {  // Ensure that there are no name collisions.

struct Point : Primitive {     // This is used for encoding point light sources.
    Point() {}
    Point( const Vec3 &p ) { position = p; }
    virtual bool Intersect( const Ray &ray, HitInfo & ) const { return false; }
    virtual bool Inside( const Vec3 & ) const { return false; }
    virtual Object *Transform( const Mat3x4 & ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual unsigned GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "point"; }
    Vec3 position;
    };

REGISTER_PLUGIN( Point );

Plugin *Point::ReadString( const string &params ) // Reads params from a string.
    {
    Vec3 pos;
    ParamReader get( params );
    if( get[MyName()] && get[pos] ) return new Point( pos );
    return NULL;
    }

Interval Point::GetSlab( const Vec3 &v ) const
    {
    double dot = v * position;
    return Interval( dot, dot );
    }

unsigned Point::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const
    {
    // Return exactly one point, with unit weight.
    samples[0].P = position;
    samples[0].w = 1.0;
    return 1;
    }

// Return a clone of this object that is transformed by M.
Object *Point::Transform( const Mat3x4 &M ) const
    {
    return new Point( M * position );
    }

} // namespace __point_primitive__






