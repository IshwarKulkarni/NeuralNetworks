/***************************************************************************
* util.h                                                                   *
*                                                                          *
* Miscellaneous utilities, such as predicates on materials & objects.      *
*                                                                          *
* History:                                                                 *
*   04/16/2010  Added functions for generating permutations.               *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __UTIL_INCLUDED__
#define __UTIL_INCLUDED__

#include <string>
#include "toytracer.h"

inline bool almostEqual(double a, double b)
{
    return (fabs(a - b) < MachEps_d);
}

inline double sqr(double x)
{
    return x * x;
}

inline double min(double x, double y)
{
    return x <= y ? x : y;
}

inline double max(double x, double y)
{
    return x >= y ? x : y;
}

inline double min(double x, double y, double z)
{
    return (x <= y && x <= z) ? x : (y <= z ? y : z);
}

inline double max(double x, double y, double z)
{
    return (x >= y && x >= z) ? x : (y >= z ? y : z);
}

inline bool isEmitter(const Material *mat)
{
    return (mat != NULL) &&
        (mat->emission != 0.0);
}

inline bool isEmitter(const Object *obj)
{
    return (obj != NULL) &&
        (obj->material != NULL) &&
        (obj->material->emission != 0.0);
}

inline bool isDiffuse(const Object *obj)
{
    return (obj != NULL) &&
        (obj->material != NULL) &&
        (obj->material->diffuse != 0.0);
}

inline bool isReflective(const Object *obj)
{
    return (obj != NULL) &&
        (obj->material != NULL) &&
        (obj->material->reflectivity != 0.0);
}

inline bool isTranslucent(const Object *obj)
{
    return (obj != NULL) &&
        (obj->material != NULL) &&
        (obj->material->translucency != 0.0);
}

// Construct an axis-aligned bounding box (AABB) for an object
// by requesting three orthogonal slabs.
inline AABB GetBox(const Object &obj)
{
    return AABB(
        obj.GetSlab(Vec3::X_axis()),
        obj.GetSlab(Vec3::Y_axis()),
        obj.GetSlab(Vec3::Z_axis())
        );
}

// Return enough backspaces to rub out the integer n.
extern const std::string &rubout(
    int n
    );

// Return enough backspaces to rub out the string str.
extern const std::string &rubout(
    const std::string &str
    );

extern bool operator==(
    const Material &a,
    const Material &b
    );

// Return a random number uniformly distributed in [a,b].
extern double rand(
    double a,
    double b
    );

// Fill the integer array with a random permutation of
// the integers 0, 1, ... n-1.  This is based on the "rand"
// function, so it is dependent upon the state of the random
// number generator.
extern void random_permutation(
    int n,
    int perm[],
    unsigned long seed = 0
    );

// Fill the integer array with a non-random but arbitrary
// permutation of the integers 0, 1, ... n-1.  This provides
// some of the benefits of randomness while being completely
// repeatable.  Different seed values will produce different
// permutations.
extern void non_random_permutation(
    int n,
    int perm[],
    unsigned long seed = 0
    );

// Return the smallest divisor of n if it's composite, or
// zero if it's prime.
extern unsigned long smallest_divisor(
    unsigned long n
    );

// Convert a plugin type to a string.
extern std::string ToString(
    plugin_type ptype
    );

// Convert a ray type to a string.
extern std::string ToString(
    ray_type rtype
    );

// Print information about a material.
extern std::ostream &operator<<(
    std::ostream &out,
    const Material &m
    );

// Print information about an object.
extern std::ostream &operator<<(
    std::ostream &out,
    const Object &obj
    );

// Copy pointer fields from one object to another.
extern void CopyAttributes(
    Object *,       // The object to copy to.
    const Object *  // The object to copy from.
    );

// Count the number of objects contained in a scene graph.
// This function will count either just the primitive objects,
// or the primitives and the aggregates.
int CountObjects(
    const Object *root,
    bool just_primitives = true
    );

extern Vec3 getRefractedDirection(Vec3 V, Vec3 N, double n);


#endif

