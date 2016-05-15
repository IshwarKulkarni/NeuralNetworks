#ifndef CURAYTRACER_CUH
#define CURAYTRACER_CUH

#include "graphics/RayTracer.hxx"
#ifdef __CUDACC__
#define DEVHOST __device__ __host__
#include "cuda_runtime.h"
#else
#define DEVHOST 
#endif 

struct CuRasterizer : public Rasterizer
{
    CuRasterizer() :Rasterizer(RasterizerInst()){}
    Color* cuRaster;
    virtual void Rasterize();
    virtual void Build();
};

struct CuObjectContainer : public  ObjectContainer
{
    CuObjectContainer() :ObjectContainer(ObjectContainerInst()){}
    virtual void Build(); // to be called after CuMaterialContainer::Build() is called
    virtual Solid* GetLight(unsigned i){ return CuSolids[i]; }
    DEVHOST void CuCast(const Ray& ray, HitInfo& hit) const;
    bool built;
    Solid** CuSolids; // copy of Solid pointer on the device;
    Material** CuMaterials;

    unsigned NumSolids;
};
struct CuMaterialContainer : public  MaterialContainer
{
    CuMaterialContainer() :MaterialContainer(MaterialContainerInst()){}
    virtual void Build();
    inline virtual Material* GetMaterial(unsigned idx){ return CuMaterials[idx]; }

    bool built;
    Material** CuMaterials; // copy of Material pointer on the device;
    unsigned NumMaterials;
};

struct CuShader : public  Shader
{
    CuShader() :Shader(ShaderInst()){}
    virtual void Build();
    DEVHOST Colorf CuTrace(const CuObjectContainer* ObjectContainer, const Ray& ray) const;
};

struct CuRayTracer : public RayTracer
{
    CuObjectContainer   m_Objects;
    CuMaterialContainer m_Materials;
    CuShader            m_Shader;
    CuRasterizer        m_Rasterizer;

    virtual void Build(std::istream& strm);

    virtual void Fire(std::string filename);

    static inline CuRayTracer& Inst() // singleton instance
    {
        static  CuRayTracer instance;
        return instance;
    }

    ~CuRayTracer(){ Die(); }

private:
    CuRayTracer() {}        
    CuRayTracer(RayTracer&){};
    CuRayTracer& operator=(const CuRayTracer&){ return Inst(); };
    virtual void Die();
};


inline CuObjectContainer&   CuObjectContainerInst()   { return CuRayTracer::Inst().m_Objects; }
inline CuMaterialContainer& CuMaterialContainerInst() { return CuRayTracer::Inst().m_Materials; }
inline CuShader&            CuShaderInst()            { return CuRayTracer::Inst().m_Shader; }
inline CuRasterizer&        CuRasterizerInst()        { return CuRayTracer::Inst().m_Rasterizer; }


#endif
