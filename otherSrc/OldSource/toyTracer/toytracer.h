/***************************************************************************
* toytracer.h                                                              *
*                                                                          *
* This is the main header file for a "Toy" ray tracer, or "toytracer".     *
* Every module of the system includes this header as it defines all of     *
* the fundamental structures needed by the ray tracer.                     *
*                                                                          *
* History:                                                                 *
*   04/26/2010  Added Preprocess plugin.                                   *
*   04/21/2010  Added ObjectSet as base class for Aggregate and Container. *
*   04/17/2010  Added Hit method to Object for keeping statistics.         *
*   04/15/2010  Added Container class & added Transform method to Object.  *
*   04/01/2010  Added Raster class & changed return value of Writer.       *
*   04/12/2008  Added Writer plugin base class.                            *
*   10/18/2005  Added primitive base classes.                              *
*   09/29/2005  Now supports more plugins, including shaders.              *
*   10/10/2004  Added Aggregate sub-class & REGISTER_OBJECT macro.         *
*   10/06/2004  Added type to ray & ray to HitInfo.  Removed HitGeom.      *
*   09/29/2004  Updated for Fall 2004 class.                               *
*   04/10/2003  Added GetSamples to Object class.                          *
*   04/06/2003  Added Object class.                                        *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __TOYTRACER_INCLUDED__  // Include this file only once.
#define __TOYTRACER_INCLUDED__

#define TOYTRACER_MAJOR_VERSION 8  // Indicates major revisions.
#define TOYTRACER_MINOR_VERSION 3  // Indicates minor fixes & extensions.

// Declare all the main structures here so that we can define pointers
// to these objects before their full definitions are provided.

struct Plugin;      // Things that can be added to the toytracer.
struct Object;      // Anything that can be ray traced.
struct ObjectSet;   // Base class for anything maintaining a set of objects.
struct Container;   // A plugin that is a set of objects, for tessellations etc.
struct Primitive;   // Simple geometrical objects: triangle, sphere, etc.
struct Aggregate;   // Collections of primitive and/or other aggregate objects.
struct Shader;      // Determines the appearance of a surface.
struct Envmap;      // An environment map, or the light "from infinity".
struct Material;    // Surface parameters that are passed to shaders.
struct Camera;      // The parameters of the camera, image resolution, etc.
struct Builder;     // Builds the scene, usually by reading a file (e.g. sdf).
struct Scene;       // All geometry, lights, environment map, rasterizer, etc.
struct Preprocess;  // A process run after scene is built, but before rendering.
struct Rasterizer;  // The function that casts primary rays & creates a Raster.
struct Raster;      // A 2D array of (floating point) colors corresponding to pixels.
struct Writer;      // Writes the Raster to a file.

#include <string>
#include <vector>
#include "vec3.h"         // Defines the Vec3 class, which are points in R3.
#include "vec2.h"         // Defines the Vec2 class, which are points in R2.
#include "mat3x3.h"       // Defines 3x3 matrices.
#include "mat3x4.h"       // Defines 3x4 matrices; i.e. affine transforms.
#include "color.h"        // Defines the Color class; real RGB values.
#include "interval.h"     // Defines a (min,max) interval of the real line.
#include "aabb.h"         // Defines a 3D axis-aligned bounding box.
#include "ray.h"          // Defines rays in 3-space: origin, direction, etc.
#include "plugins.h"      // Defines functions for accessing plugins.
#include "constants.h"    // Mathematical constants & default values.

// The Material class defines all the surface attributes used for shading.
struct Material {
    Material();
    ~Material() {}
    Color  diffuse;       // Diffuse color.
    Color  specular;      // Color of highlights.
    Color  emission;      // Emitted light.
    Color  ambient;       // Ambient light (from all directions).
    Color  reflectivity;  // Weights for refleced light, between 0 and 1.
    Color  translucency;  // Weights for refracted light, between 0 and 1.
    double Phong_exp;     // Phong exponent for specular highlights.
    double ref_index;     // Refractive index.
    long   type;          // A flag that can be used by shaders.
};

// The HitInfo class records all info at a ray-object intersection.  It also
// communicates maximum distance to an intersection method. 
struct HitInfo {
    HitInfo();
    ~HitInfo() {}
    const Object *object; // The object that was hit (set by Intersect).
    double  distance;     // Distance to hit (used & reset by Intersect).
    Vec3    point;        // ray-object intersection point (set by Intersect).
    Vec3    normal;       // Surface normal (set by Intersect).
    Vec2    uv;           // Texture coordinates (set by intersect).
    Ray     ray;          // The ray that hit the surface (set by Cast).
};

// The Sample class is used for Monte Carlo sampling.  It consists of a point and
// weight returned from a sampling algorithm.
struct Sample {
    Vec3   P;  // A point on or in the object being sampled.
    double w;  // The weight, which should be the approximate solid angle.
};

// The Camera class defines all the camera parameters and the image resolution.
struct Camera {
    Camera() {}
    ~Camera() {}
    Vec3     eye;     // Position of eye.
    Vec3     lookat;  // The point we are looking toward.
    Vec3     up;      // A vector not parallel to the direction of gaze.
    double   vpdist;  // Distance to the view plane.
    Interval x_win;   // Horizontal extent of view window (typically [-1,1]).
    Interval y_win;   // Vertical extent of view window (typically [-1,1]).
    unsigned x_res;   // Horizontal image resolution in pixels.
    unsigned y_res;   // Vertical image resolution in pixels.
};

// The Scene class contains all the information needed to trace rays through 
// the simulated environment as well as pointers to the plugins that will
// produce the image.
struct Scene {
    Scene() { init(); }
    ~Scene() { lights.clear(); }
    void  init();
    Color Trace(const Ray &ray) const;
    bool  Cast(const Ray &ray, HitInfo &hitinfo) const;
    bool  Visible(const Vec3 &P, const Vec3 &Q, const Object *ignore = NULL, double eps = 0.0) const;
    virtual const Object *GetLight(unsigned i) const { return lights[i]; }
    virtual unsigned NumLights() const { return lights.size(); }
    static double NumRaysCast() { return num_rays_cast; }
    Envmap      *envmap;         // Global environment map, if ray hits nothing.
    Shader      *shader;         // Default shader for objects that don't specify one.
    Object      *object;         // A single primitve or an aggregate object.
    Rasterizer  *rasterize;      // Cast all primary rays & create a Raster.
    Writer      *writer;         // Writes a Raster to a file--tone mapping optional.
    std::vector<Object*> lights; // All objects that are emitters.  
    unsigned max_tree_depth;     // Limit on depth of the ray tree.
private: // Keep the following private to ensure that statistics are kept properly.
    static double num_rays_cast;
};

// The Raster class contains the raw color information produced by the rasterizer.  There is
// no clamping or tone mapping at this satage.
struct Raster {
    Raster(int width = 0, int height = 0);
    ~Raster();
    inline Color const &pixel(unsigned r, unsigned c) const { return pixels[r * width + c]; }
    inline Color       &pixel(unsigned r, unsigned c) { return pixels[r * width + c]; }
    int width;
    int height;
    Color *pixels;
};

// This is the base class for classes that contain collections of objects, which includes both
// Aggregates and Containers.  All the virtual methods have simple default definitions that create
// and access a simple vector of objects.  For many object collections, these defaults are
// completely sufficient.  For those requiring something more sophisticated, each of these
// virtual methods can be redefined appropriately.
struct ObjectSet {
    ObjectSet() { index = 0; }
    virtual ~ObjectSet() { children.clear(); }
    virtual void AddChild(Object *obj) { children.push_back(obj); }
    virtual void Close() {} // Called after all children have been added.
    virtual int  Size() const { return children.size(); }
    virtual void Begin() const { index = 0; } // Start at front of list.
    virtual Object *GetChild() const { return index >= children.size() ? NULL : children[index++]; }
    mutable unsigned index;
    std::vector<Object*> children;
};

/***************************************************************************
* Below are all the plugin types.  Each plugin must have one of these as   *
* its base class and must supply all of the pure virtual methods.  Plugins *
* are added to the toytracer by means of the REGISTER_PLUGIN macro and     *
* by linking in the object code.   (See the plugins.h header file.)        *
***************************************************************************/

// Base class for all shaders, one of which is associated with each object.
struct Shader : Plugin {
    Shader() {}
    virtual ~Shader() {}
    virtual Color Shade(const Scene &scene, const HitInfo &hitinfo) const = 0;
    virtual plugin_type PluginType() const { return shader_plugin; }
};

// Base class for environment maps, one of which is associated with each object.
struct Envmap : Plugin {
    Envmap() {}
    virtual ~Envmap() {}
    virtual Color Shade(const Ray &ray) const = 0;
    virtual plugin_type PluginType() const { return envmap_plugin; }
};

// Base class for object containers, which are plugins that manage collections of objects.
struct Container : Plugin, ObjectSet {
    Container() { index = 0; num_containers++; }
    virtual ~Container() {}
    virtual plugin_type PluginType() const { return container_plugin; }
    static double num_containers; // Count how many containers were created.
};

// Base class for all objects that can be ray traced.  This is the most fundamental class
// of the toytracer, as it encapsulates most of the operations needed for ray tracing.
// This is an abstract class that cannot be instantiated directly.
struct Object : Plugin {
    Object();
    virtual ~Object() {}
    inline  bool Hit(const Ray &, HitInfo &) const; // Calls Intersect.
    virtual Interval GetSlab(const Vec3 &) const = 0;
    virtual double Cost() const { return 1.0; }
    virtual Object *Transform(const Mat3x4 &) const { return NULL; }
    virtual bool Inside(const Vec3 &) const { return false; }
    virtual Ray getRefracetedRay(Ray r, Vec3 P, Vec3 N)const { return r; }
    virtual Ray getReflectedRay(Ray r, Vec3 p, Vec3 N)const { return r; }
    virtual unsigned GetSamples(const Vec3 &P, const Vec3 &N, Sample *, unsigned n) const { return 0; }
    static  double NumIntersectionTests() { return num_intersection_tests; }
    static  double NumVisibilityTests() { return num_visibility_tests; }

    Shader   *shader;
    Envmap   *envmap;
    Material *material;
private: // Keep the following private to ensure that statistics are properly computed.
    virtual bool Intersect(const Ray &ray, HitInfo &) const = 0;
    static double num_visibility_tests;
    static double num_intersection_tests;
};

// A public wrapper that maintains statistics for the object's Intersect method.
inline bool Object::Hit(const Ray &ray, HitInfo &info) const
{
    if (ray.type == visibility_ray) num_visibility_tests++;
    num_intersection_tests++;
    return Intersect(ray, info);
}

// Base class for all primitive objects, such as spheres, triangles, and quads.
struct Primitive : Object {
    Primitive() { num_primitives++; }
    virtual ~Primitive() {}
    virtual plugin_type PluginType() const { return primitive_plugin; }
    static double num_primitives; // Count how many primitives were created.
};

// Base class for all aggregate objects, such as bounding volume hierarchies and
// spatial subdivision methods (i.e. acceleration techniques).
struct Aggregate : Object, ObjectSet {
    Aggregate() { num_aggregates++; }
    virtual ~Aggregate() {}
    virtual plugin_type PluginType() const { return aggregate_plugin; }
    static double num_aggregates; // Count how many aggregates were created.
};

// Base class for scene builder, which creates the camera and the scene to be rendered.
// Typically, a builder will read a scene description from a file, but it could also
// create a scene algorithmically.
struct Builder : Plugin {
    Builder() {}
    virtual ~Builder() {}
    virtual bool BuildScene(std::string command, Camera &, Scene &) = 0;
    virtual plugin_type PluginType() const { return builder_plugin; }
};

// Base class for "preprocesses" that are run after builder, but before the rasterizer.
// A preprocess can build photon maps or perform other operations on the scene graph
// before rendering begins.  The order in which preprocesses are run is determined by
// their priorities.
struct Preprocess : Plugin {
    Preprocess() {}
    virtual ~Preprocess() {}
    virtual bool Run(std::string command, Camera &, Scene &) = 0;
    virtual double Priority() const { return 0.0; }
    virtual plugin_type PluginType() const { return preprocess_plugin; }
};

// Base class for all raserization algorithms.  A raserizer casts all the primary
// rays and creates a Raster of the resulting colors.
struct Rasterizer : Plugin {
    Rasterizer() {}
    virtual ~Rasterizer() {}
    virtual Raster *Rasterize(const Camera &camera, const Scene &scene) = 0;
    virtual plugin_type PluginType() const { return rasterizer_plugin; }
    virtual bool GetHitInfo(const Ray &, HitInfo &) { return false; }
};

// Base class for all writers, which handle the Raster produced by the raserizer.
// Typically, the writer will save the raster to a file, possibly after tone mapping.
struct Writer : Plugin {
    Writer() {}
    virtual ~Writer() {}
    virtual std::string Write(std::string file_name, const Raster *) = 0;
    virtual plugin_type PluginType() const { return writer_plugin; }
};

#endif

