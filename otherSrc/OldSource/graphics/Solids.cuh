// Definitions and ray intersection of solids
#ifndef _SOLIDS_HXX_
#define _SOLIDS_HXX_

#include "graphics/geometry/Primitives.hxx"
#include "graphics/RayTracer.hxx"
#include "utils/Readable.hxx"

#ifdef __CUDACC__
#define DEVHOST __device__ __host__
#include "cuda_runtime.h"
#else
#define DEVHOST 
#endif 

struct Solid 
{
    // This method will fill the hitinfo object except .color member
    DEVHOST virtual bool RayIntersect(struct Material** mat, const struct Ray& ray, struct HitInfo& hitInfo) const { return false; }
    void AddToContainer();
    unsigned MaterialIdx;
};

struct DTriangle : Solid, Readable // deformed triangle, deformed with normals
{
    DEVHOST virtual bool RayIntersect(Material** mat, const struct Ray& ray, struct HitInfo& hitInfo) const;
    virtual inline std::string GetPrefix() const { return "DTriangle"; };
    virtual DTriangle* MakeFromStream(std::istream& strm);
    
    TrianglePrimitive Primitive;
    Vec3<Vec3f> Normals;
};
Register(DTriangle)


struct Triangle : Solid, Readable, TrianglePrimitive
{
    DEVHOST virtual bool RayIntersect(Material** mat, const Ray& ray, HitInfo& hitInfo) const;

    virtual inline std::string GetPrefix() const { return "Triangle"; };
    virtual Triangle* MakeFromStream(std::istream& strm);
};
Register(Triangle)

struct Sphere : Solid, Readable
{
    DEVHOST virtual bool RayIntersect(struct Material** mat, const Ray& ray, HitInfo& hitInfo) const;

    virtual inline std::string GetPrefix() const { return "Sphere"; };
    virtual Sphere* MakeFromStream(std::istream& strm);

    SpherePrimitive Primitive;

    Sphere() {}
};
Register(Sphere)



#endif