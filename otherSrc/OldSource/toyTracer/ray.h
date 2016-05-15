/***************************************************************************
* ray.h                                                                    *
*                                                                          *
* The "Ray" structure defines a ray in 3-space, consisting of an origin,   *
* a direction, a "type", and a generation number.  The ray also records    *
* the object it originated from (if there was one), and a single object to *
* ignore (such as a light source), which is convenient for some ray        *
* tracing algorithms.                                                      *
*                                                                          *
* History:                                                                 *
*   04/28/2010  Added "visibility_ray" for detecting occlusions.           *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __RAY_INCLUDED__
#define __RAY_INCLUDED__

#include <iostream>
#include "vec3.h"
#include "graphics/geometry/Vec23.hxx"

struct Object; // Declared elsewhere.

// Possible ray "types" that may be set and used by a shader and/or an
// acceleration method.
enum ray_type {         // A flag that may affect the processing of a ray.
	undefined_ray = 0,  // Used to indicate unitialized rays.
	generic_ray = 1,  // Any ray; No special meaning.
	visibility_ray = 2,  // A ray cast to detect occlusion, not closest hit.
	indirect_ray = 3,  // A ray cast from a surface to sample illumination.
	light_ray = 4,  // A ray cast from a light source (e.g. for photon mapping).
	refracted_ray = 5,  // A ray that has passed through a translucent interface.
	special_ray = 6,   // A ray used for some other special purpose.
	reflected_ray = 7,	//Let me not be redundant by adding what this ray is.
	sparse_rast_ray = 8
};

struct Ray { // A ray in R3.
	Ray();
	Ray(const Ray &r);
	inline Vec3 operator[](double s) const { return origin + s * direction; }
	Vec3 origin;          // The ray originates from this point.
	Vec3 direction;       // Unit vector indicating direction of ray.
    Vec::Loc  loc;        // Gen 1 rays have a location
	ray_type type;        // Different rays may be processed differently.
	unsigned generation;  // How deep in the ray tree.  1 ==> generated from eye.
	const Object *from;   // The object from which the ray was cast.
	const Object *ignore; // Optionally ignore one object when intersecting.
};

// Compute the reflected ray given the incident ray (i.e. directed
// toward the surface), and the normal to the surface.  The normal
// may be directed away from or into the surface.  Both the surface
// normal and the ray direction vector are assumed to be normalized.
extern Vec3 Reflect(
	const Ray &r,
	const Vec3 &N
	);

// Compute the refracted ray given the incident ray (i.e. directed
// toward the surface), the normal to the surface, and the ratio of the
// refractive indices of the material containing the ray and the
// material into which the ray is refracted.  The normal
// may be directed away from or into the surface.  Both the surface
// normal and the ray direction vector are assumed to be normalized.
extern Vec3 Refract(
	const Ray &r,
	const Vec3 &N,
	double eta1_over_eta2
	);

// An output method for rays, useful for debugging.
extern std::ostream &operator<<(
	std::ostream &out,
	const Ray &r
	);

#endif

