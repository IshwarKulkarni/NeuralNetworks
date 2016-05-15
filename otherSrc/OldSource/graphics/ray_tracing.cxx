#include <iostream>
#include <vector>
#include "graphics/geometry/primitives.hxx"
#include "graphics/RayTracer.hxx"
#include "imageprocessing/ImageRGB.hxx"


using namespace std;
using namespace Logging;

struct Primitive 
{

};

struct ObjectContainer : vector< Solid*> {
    bool ray_intersect(Ray& ray)
    {
        HitInfo hit;
        hit.hitDistance = std::numeric_limits<float>::infinity();
        for (auto& o : Objects)
            if (o->RayIntersect(ray, hit))
                return true;
        return false;
    }
} Objects;


int main()
{
    Camera camera;
    fstream cameraFile("E:\\Projects\\MyCudaProject\\MyCudaProject\\cameraFile.txt");

    unsigned ThreadsPerBlock;
    cameraFile >> std::skipws >> ThreadsPerBlock >> camera;

    TrianglePrimitive tri = make_triangle(
        { 1.f, 0.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f, 0.f },
        { 0.f, 0.f, 1.f, 0.f });

    SpherePrimitive sph( 0.f, 0.f, 0.f, .5f );

    Triangle    t; t. (tri, { 255, 255, 255 });
    Solid       s; t. (sph, { 255, 255, 255 })

    Objects.push_back();
    Objects.push_back();

    const Vec3f    G = normalize(camera.lookat - camera.eye),
        U = normalize(camera.up / G),
        R = normalize(G ^ U),
        O = camera.vpdist * G +
            camera.x_win.min * R +
            camera.y_win.max * U,  // Raster bottom left corner
        dR = (camera.x_win.width() * R / camera.x_res), // Right increments.
        dU = (camera.y_win.width() * U / camera.y_res); // Up increments.

    Matrix2D<Colorf> raster(camera.x_res, camera.y_res, Colorf(0));
    
    
    for (unsigned row = 0; row < camera.y_res; row++)
        for (unsigned col = 0; col < camera.x_res; col++)
        {
            Ray ray;

            ray.origin = camera.eye,
                ray.direction = normalize(O + (col + 0.5f) * dR - (row + 0.5f) * dU),
                ray.intersectionPt = NullVec3f,
                ray.intersectionDist = numeric_limits<float>::infinity(),
                ray.color = { 0.f, 0.f, 0.f };
            
            Log << ray << LogEndl;
            if (Objects.ray_intersect(ray))
            {
                raster[row][col] = ray.color;
                Log << " Intersected: " << ray.intersectionPt <<  " , " << ray.intersectionDist <<  "\n";
            }
        }

    ImageRGB image(camera.x_res, camera.y_res, Black);
    
    for (unsigned row = 0; row < camera.y_res; row++)
        for (unsigned col = 0; col < camera.x_res; col++)
        {
            image[row][col].R = char(raster[row][col].x);
            image[row][col].G = char(raster[row][col].y);
            image[row][col].B = char(raster[row][col].z);
        }

    image.WriteAsImage("raster");
    
    return 0;
}



void hosttrace(unsigned x, unsigned y, const Vec3f eye, const Vec3f O, const Vec3f dR, const Vec3f dU,
    Color* raster, size_t ElemPitch)
{
    Ray ray;
        ray.origin = eye,
        ray.direction = normalize(O + (x + 0.5f) * dR - (y + 0.5f) * dU),
        ray.intersectionPt = { finf, finf, finf },
        ray.intersectionDist = finf;
        ray.color.x = ray.color.y = ray.color.z = 0;

    Triangle t = make_triangle({ 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 });
    TraceableObject o(t, Colorf(255, 255, 255));

    Color c;
    if (o.intersect(ray))
    {
        //printf("\nx = %d, y = %d");
        c.R = char(ray.color.x);
        c.G = char(ray.color.y);
        c.B = char(ray.color.z);
        raster[x + y*ElemPitch] = c;
    }
}

#define iDivUp(a, b) (uint)( (a % b) ? (a / b + 1) : (a / b) )
typedef Vec<unsigned> dim3;

int main2()
{
    fstream cameraFile("E:\\Projects\\MyCudaProject\\MyCudaProject\\cameraFile.txt");

    unsigned ThreadsPerBlock;
    Camera camera;
    cameraFile >> std::skipws >> ThreadsPerBlock >> camera;

    const Vec3f
        G = normalize(camera.lookat - camera.eye),
        U = normalize(camera.up / G),
        R = normalize(G ^ U),
        O = camera.vpdist * G +
        camera.x_win.min * R +
        camera.y_win.max * U,  // Raster bottom left corner 
        dR = (camera.x_win.width() * R / camera.x_res), // Right increments.
        dU = (camera.y_win.width() * U / camera.y_res); // Up increments.


    dim3 gridSize(iDivUp(camera.x_res, ThreadsPerBlock),
        iDivUp(camera.y_res, ThreadsPerBlock), 1);
    dim3 blockSize(ThreadsPerBlock, ThreadsPerBlock, 1);

    THROW_IF(camera.x_res != gridSize.x * blockSize.x || camera.y_res != gridSize.y * blockSize.y,
        DimensionException, "Camera raster size is different from number of threads");
    Matrix2D<Color> raster(camera.x_res, camera.y_res, Black);

    for (size_t gx = 0; gx < gridSize.x; gx++)
    for (size_t gy = 0; gy < gridSize.y; gy++)
        for (size_t bx = 0; bx < blockSize.x; bx++)
        for (size_t by = 0; by < blockSize.y; by++)
        {
            unsigned x = gx*blockSize.x + bx,
                y = gy*blockSize.y + by;
            hosttrace(x, y, camera.eye, O, dR, dU, raster.GetData(), raster.Width());
        }
        
    raster.Write("raster.host.csv");
    ImageIO::WriteImage("raster.host.ppm", make_pair(camera.x_res, camera.y_res), 3,
            (byte*)raster.GetData(), true, ImageIO::PPM);

    return 0;
}
