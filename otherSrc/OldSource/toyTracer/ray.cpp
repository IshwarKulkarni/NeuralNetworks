/***************************************************************************
* ray.cpp                                                                  *
*                                                                          *
* The "Ray" structure defines a ray in 3-space, consisting of an origin    *
* (denoted by "Q") and a direction (denoted by "R").  Several additional   *
* fields are also included (e.g. type and generation) as a convenience for *
* some ray tracing algorithms.                                             *
*                                                                          *                                                                        
* History:                                                                 *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "ray.h"
#include "util.h"

using std::string;
using std::ostream;

Ray::Ray()
    {
    ignore = NULL;
    generation = 1;
    type = undefined_ray;
    from = NULL;
    }

Ray::Ray( const Ray &r )
    {
    *this  = r;
    ignore = NULL;
    }

// Compute the reflected ray given the incident ray (i.e. directed
// toward the surface), and the normal to the surface.  The normal
// may be directed away from or into the surface.  Both the surface
// normal and the ray direction vector are assumed to be normalized.
Vec3 Reflect( const Ray &r, const Vec3 &N )
    {
    Vec3 U( r.direction );
    return U - ( 2.0 * ( U * N ) ) * N;
    }

Ray Clone(Ray r){

	Ray ret;
	ret.direction = r.direction;
	ret.from = ret.from;
	ret.generation = r.generation;
	ret.ignore = r.ignore;
	ret.origin = r.origin;
	ret.type = r.type;
	return ret;


	}

// Compute the refracted ray given the incident ray (i.e. directed
// toward the surface), the normal to the surface, and the ratio of the
// refractive indices of the material containing the ray and the
// material into which the ray is refracted.  The normal
// may be directed away from or into the surface.  Both the surface
// normal and the ray direction vector are assumed to be normalized.
Vec3 Refract( const Ray &r, const Vec3 &N, double eta1_over_eta2 )
    {
    Vec3 U( r.direction );
    // This is just a placeholder...
    return U;
    }

// An output method, useful for debugging.
ostream &operator<<( ostream &out, const Ray &r )
    {
    out << "<" 
        << ToString( r.type ) << ":"
        << r.generation       << ", "
        << r.origin           << ", "
        << r.direction
        << ">";
    return out;
    };
