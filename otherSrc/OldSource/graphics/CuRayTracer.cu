#include "graphics/CuRayTracer.cuh"
#include "utils/CudaUtils.hxx"
#include "device_functions.h"
#include "utils/Logging.hxx"

void CuObjectContainer::Build()
{
    if (built) return;

    ObjectContainer::Build();

    CuSolids = cudaCopyPtrVector(*(this->ObjectVec), NumSolids);
    CuMaterials = CuMaterialContainerInst().CuMaterials;
    built = true;
}

DEVHOST inline void CuObjectContainer::CuCast(const Ray& ray, HitInfo& hit) const
{
    for (unsigned i = 0; i < NumSolids; ++i)
        CuSolids[i]->RayIntersect(CuMaterials, ray, hit);
}

void CuMaterialContainer::Build()
{
    if (built) return;
    CuMaterials = cudaCopyPtrVector(*(this->MaterialVec), NumMaterials);
    built = true;
}

void CuShader::Build()
{
    
}

void CuRasterizer::Build()
{
    Rasterizer::Build();
    cuRaster = cudaPitchedAllocCopy((Color*)(0), width, height, pitch);
}
DEVHOST Colorf CuShader::CuTrace(const CuObjectContainer* ObjectContainer, const Ray& ray) const
{
    HitInfo hit;
    hit.color = VoidColor;
    unsigned inf = 0x7f800000;
    hit.hitDistance = *((float*)(&inf));

    ObjectContainer->CuCast(ray, hit);

    return hit.color;
}

inline void CuRayTracer::Build(std::istream& strm) 
{
    RayTracer::Build(strm);
    CuMaterialContainerInst().Build();
    CuObjectContainerInst().Build();
    CuRasterizerInst().Build();
    CameraInst().Build();
}
void CuRayTracer::Die()
{
    cudaClearPtrArrays(CuObjectContainerInst().CuSolids, 
        CuObjectContainerInst().ObjectVec->size());

    cudaClearPtrArrays(CuMaterialContainerInst().CuMaterials,
        CuMaterialContainerInst().NumMaterials);

}

void CuRayTracer::Fire(std::string filename)
{
    CuRasterizerInst().Rasterize(); 

    ImageIO::WriteImage(filename,
        make_pair(CameraInst().x_res, CameraInst().y_res),
        3, (const byte *)(CuRasterizerInst().Raster), true, ImageIO::PPM);
}

static __global__ void LaunchRasterizer(
    const CuShader* shader, 
    const CuObjectContainer* container, 
    const Camera* camera, 
    const CuRasterizer* rasterizer)
{
    //IMXYCHECK(camera->x_res,camera->y_res);
    unsigned x = threadIdx.x,
        y = threadIdx.y;

    Ray ray;

    ray.origin = camera->eye;
    ray.direction = normalize(
        camera->O
        + (x + 0.5f) * camera->dR
        - (y + 0.5f) * camera->dU);
    printf("trying : (%d, %d) \n", x, y);
    Colorf ret = shader->CuTrace(container, ray);
    Color& c = offsetToPitchedArray(rasterizer->cuRaster, rasterizer->pitch, x, y);

    c.R = ret.x;
    c.G = ret.y;
    c.B = ret.z;
}

void CuRasterizer::Rasterize()
{
    Logging::Timer RasTime("Rasterizer timer");
    Camera& camera = CameraInst();

    LaunchConfig(camera.x_res, camera.y_res, .8f);

    CuShader*           shader    = cudaAllocCopy(&CuShaderInst());
    CuObjectContainer*  container = cudaAllocCopy(&CuObjectContainerInst());
    Camera*             cuCamera  = cudaAllocCopy(&CameraInst());

    //LaunchRasterizer <<< gridSize,blockSize >>> (shader, container, cuCamera, this);
    LaunchRasterizer <<< dim3(CuRasterizerInst().width, CuRasterizerInst().height), dim3(1,1,1)>>> (shader, container, cuCamera, this);
    CUDA_CHECK(); 
    cudaCopyOut(cuRaster, pitch, width, height, &(Rasterizer::Raster));

    cudaFree(shader   );
    cudaFree(container);
    cudaFree(cuCamera );
}

