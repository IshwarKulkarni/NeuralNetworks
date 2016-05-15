#include "cuda_runtime.h"
#include "utils/CudaUtils.hxx"
#include "utils/Logging.hxx"
#include "utils/StringUtils.hxx"
#include "graphics/CuRayTracer.cuh"
#include "utils/Commandline.hxx"
#include "utils/StringUtils.hxx"

using namespace std;
using namespace StringUtils;

void triIntersectionTest();

int main(int argc, char** args)
{
    SetLastDevice();

    bool test = false;
    if (!test)
    {
        NameValuePairParser cmdLineParser(argc, args);

        string filename = "scene";
        cmdLineParser.Get("scene", filename);

        StringUtils::establishEndsWith(filename, ".scn");
        fstream sceneFile(filename.c_str(), ios_base::in | std::ios::binary);

        CuRayTracer::Inst().Build(sceneFile);

        StripAfter(filename, ".");
        cmdLineParser.Get("raster", filename);
        CuRayTracer::Inst().Fire(filename);
    }
    else
    {
        triIntersectionTest();
    }
    return 0;
}


void triIntersectionTest()
{
    TrianglePrimitive prix1y1z0 = make_triangle(XAxisf, YAxisf, NullVec3f);
    Vec3f zDown(0, 0, -1);
    bool hit = false;
    Vec3f out;
    
    out = prix1y1z0.ray_intersect(Vec3f(1, 0, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath,"did not hit 1");

    out = prix1y1z0.ray_intersect(Vec3f(0, 1, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 2");

    out = prix1y1z0.ray_intersect(Vec3f(0, 1-Epsilon, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 3");

    out = prix1y1z0.ray_intersect(Vec3f(1 - Epsilon, 0, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 4");

    out = prix1y1z0.ray_intersect(Vec3f(Epsilon, 0, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 5");

    out = prix1y1z0.ray_intersect(Vec3f(0, Epsilon, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 6");

    out = prix1y1z0.ray_intersect(Vec3f(Epsilon, Epsilon, 1), zDown, hit);
    THROW_IF(!hit || !almostEqual(out.x,1), LogicalOrMath, "did not hit 7");

    srand((unsigned)time(NULL));
    Vec3f O(Rand(10, -10), Rand(10, -10), Rand(10, -10));
    Vec3f C = (prix1y1z0.x + prix1y1z0.y + prix1y1z0.z) / 3; //centroid
    out = prix1y1z0.ray_intersect(O, normalize(C-O), hit);
    THROW_IF(!hit, LogicalOrMath, "did not hit 7");
}