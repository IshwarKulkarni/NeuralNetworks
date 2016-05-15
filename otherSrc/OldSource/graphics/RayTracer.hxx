#ifndef RAY_TRACING_HXX
#define RAY_TRACING_HXX

#include "graphics/geometry/primitives.hxx"
#include "graphics/Solids.cuh"
#include "imageprocessing/color.hxx"
#include "utils/Readable.hxx"

#include <string>
#include <istream>
#include <vector>
#include <map>
#include <ostream>

struct Ray // a photon
{
    Vec3f    origin,
            direction,
            intersectionPt;
};

std::ostream& operator <<(std::ostream& out, const Ray& ray);

struct LocalCordinateSystem
{
    Vec3f Origin, Right, Up; // right and up are direction vec
};

struct Solid;

struct HitInfo
{
    Vec3f    hitPoint, hitNormal;
    Colorf    color;
    float    hitDistance;
    const Solid* hitObject;
};

struct Camera : Readable{
    Vec3f   eye,     // Position of eye.
            lookat,  // The point we are looking toward.
            up;      // A vector not parallel to the direction of gaze.
    Intervalf   x_win,   // Horizontal extent of view window (typically [-1,1]).
                y_win;   // Vertical extent of view window (typically [-1,1]).
    unsigned    x_res,   // Horizontal image resolution in pixels.
                y_res;   // Vertical image resolution in pixels.
    float       vpdist;  // Distance to the view plane.

    // Derived types:
    Vec3f O, dR, dU; // Origin of raster, step right, step Up.
    Vec3f xPlaneN, yPlaneN, zPlaneN; // normals to three planes
    virtual void Build();
    virtual inline std::string GetPrefix() const { return "Camera"; };
    virtual Camera* MakeFromStream(std::istream& strm);
};
Register(Camera)


struct ObjectContainer
{
    std::vector < Solid* >* ObjectVec;
    
    // fancy intersection acceleration implementation (alliteration) goes here
    virtual void Cast(const Ray& ray, HitInfo& hit) const;
    virtual inline void AddSolid(Solid* solid){ ObjectVec->push_back(solid); }
    
    unsigned NumLights;
    virtual inline Solid* GetLight(unsigned i){ return ObjectVec->at(i); }
    
    bool built;
    virtual void Build(); // this will construct the fancy acceleration structure
};

/*
Rasterizer ( produces rays)
Scene Traces Rays and returns color(produces hitinfo)
    Trace Casts ray
        Cast populates HitInfo and calls intersect
    Trace Shades returns a color
        Shade shoots light rays etc
*/
struct Rasterizer
{
    Color* Raster;
    unsigned width, height, pitch;
    virtual void Build();
    virtual void Rasterize();
};

struct Shader : public Readable
{
    Colorf Trace(const Ray& ray) const;
    void DrawAxes(const Ray& ray, HitInfo& hit) const;

    virtual inline std::string GetPrefix() const { return "Shader"; };
    virtual Shader* MakeFromStream(std::istream& strm);

    virtual inline void Build() {};

    bool    MarkAxes;   // false by default
    Colorf  AxesColorX, AxesColorY, AxesColorZ,
            VoidColor;  // black by default
};
Register(Shader)

struct Material : Readable {

    enum MaterialType
    {
        MaterialType_Generic = 0,
        MaterialType_Light,
        MaterialType_Refractive, 
        MaterialType_Translucent
    }type;          // A flag that can be used by shaders.

    Colorf  diffuse;       // Diffuse color.
    Colorf  specular;      // Color of highlights.
    Colorf  emission;      // Emitted light.
    Colorf  ambient;       // Ambient light (from all directions).
    Colorf  reflectivity;  // Weights for reflected light, between 0 and 1.
    Colorf  translucency;  // Weights for refracted light, between 0 and 1.
    float   Phong_exp;     // Phong exponent for specular highlights.
    float   ref_index;     // Refractive index.
    
    virtual inline std::string GetPrefix() const { return "Material"; };
    virtual Material* MakeFromStream(std::istream& strm);
    static bool ReadMaterial(std::istream& strm, Material* mat);
};
Register(Material)

struct MaterialContainer
{
    std::vector<Material*>* MaterialVec;
    std::map<std::string, unsigned> MaterialMap;

    inline unsigned GetTopOfStack() { return (unsigned)MaterialVec->size() - 1; }
    inline unsigned GetMaterial(std::string name){
        auto found = this->MaterialMap.find(name);
        if (found == this->MaterialMap.end())
            return unsigned(-1);
        else
            return (unsigned)found->second;
    }
    inline virtual Material* GetMaterial(unsigned idx){ return MaterialVec->at(idx); }
    unsigned AddMaterial(std::string name, Material* mat);
    void Build();
};

struct RayTracer
{
    Camera            m_Camera;
    ObjectContainer   m_Objects;
    MaterialContainer m_Materials;
    Shader            m_Shader;
    Rasterizer        m_Rasterizer;

    virtual void Build(std::istream& strm);
    virtual void BuildAllElse();

    static bool RegisterReadbleObject(Readable*readable);
    virtual void Fire(std::string filename);

    static inline RayTracer& Inst() // singleton instance
    {
        static RayTracer instance;
        return instance;
    }

    ~RayTracer() { Die(); }

protected:
    RayTracer();
    RayTracer(RayTracer&){};
    RayTracer& operator=(const RayTracer&){ return Inst(); };
    virtual void Die();
};

inline Camera&            CameraInst()            { return RayTracer::Inst().m_Camera;     }
inline ObjectContainer&   ObjectContainerInst()   { return RayTracer::Inst().m_Objects;    }
inline MaterialContainer& MaterialContainerInst() { return RayTracer::Inst().m_Materials;  }
inline Shader&            ShaderInst()            { return RayTracer::Inst().m_Shader;     }
inline Rasterizer&        RasterizerInst()        { return RayTracer::Inst().m_Rasterizer; }

#endif
