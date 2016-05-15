#include "graphics/RayTracer.hxx"
#include "utils/StringUtils.hxx"
#include "utils/Exceptions.hxx"
#include "utils/Logging.hxx"
#include "utils/Readable.hxx"
#include "graphics/geometry/vectors_io.hxx"
#include "imageprocessing/ImageRGB.hxx"

using namespace std;
using namespace StringUtils;
using namespace Logging;

static CommentHogger PCommentHogger = { '#' };

void RayTracer::Fire(std::string filename)
{
    m_Rasterizer.Rasterize();
    ImageIO::WriteImage(filename, 
        make_pair(m_Camera.x_res, m_Camera.y_res),
        3, (const byte *) m_Rasterizer.Raster, true, ImageIO::PPM);
    
  // Logging::Log << Matrix2D<Color>(CameraInst().x_res, CameraInst().y_res, m_Rasterizer.Raster);
}

RayTracer::RayTracer()
{
    ObjectContainerInst().ObjectVec = new std::vector < Solid* >();
    MaterialContainerInst().MaterialVec = new std::vector<Material*>();
}

void RayTracer::Die()
{
    auto& objs = ObjectContainerInst().ObjectVec;
    if (objs)
    {
        for (auto p : *objs)
            delete p;
        delete objs;
        ObjectContainerInst().ObjectVec = NULL;
    }

    auto& mats = MaterialContainerInst().MaterialVec;
    if (mats)
    {
        for (auto p : *mats)
            delete p;
        delete mats;
        MaterialContainerInst().MaterialVec = NULL;
    }
}

void Camera::Build()
{
    Vec3f
        G = normalize(lookat - eye),
        U = normalize(up / G),
        R = normalize(G ^ U);
        
    O = vpdist * G + x_win.min * R + y_win.max * U; // Raster bottom left corner 
    dR = (x_win.width() * R / x_res);                // Right increments.
    dU = (y_win.width() * U / y_res);                // Up increments.

    xPlaneN = normalize(eye ^ XAxisf);
    yPlaneN = normalize(eye ^ YAxisf);
    zPlaneN = normalize(eye ^ ZAxisf);
}

unsigned MaterialContainer::AddMaterial(std::string name, Material* mat){
    
    bool newMaterial = (this->MaterialMap.find(name) == this->MaterialMap.end());

    THROW_IF(!newMaterial, FileParseException, "Material with name \"%s\" has already been defined\n",
        name.c_str());

    MaterialVec->push_back(mat);
    this->MaterialMap[name] = (unsigned)this->MaterialVec->size() - 1;

    return (unsigned)this->MaterialVec->size() - 1;
}

void MaterialContainer::Build()
{
    // this is the first, default material, a pale White non-emmissive surface
    Material*  defaultMat = new Material;

    defaultMat->diffuse = { 225, 225, 225 };
    defaultMat->specular = { 127, 127, 127 };
    defaultMat->emission = { 0, 0, 0 };
    defaultMat->ambient = { 127, 127, 127 };
    defaultMat->reflectivity = { 0, 0, 0 };
    defaultMat->translucency = { 0, 0, 0 };
    defaultMat->Phong_exp = 2.5;
    defaultMat->ref_index = 1;
    defaultMat->type = Material::MaterialType_Generic;
    AddMaterial("\"default\"", defaultMat);
}


void ObjectContainer::Cast(const Ray& ray, HitInfo& hit) const
{
    bool didHit = false;
    for (unsigned i = 0; i < ObjectVec->size(); ++i)
        if (ObjectVec->at(i)->RayIntersect(&(MaterialContainerInst().MaterialVec->at(0)), ray, hit))
            didHit = true;

}

void ObjectContainer::Build() // this will construct the fancy acceleration structure
{
    if (!built)
    {
        std::partition(
            ObjectVec->begin(), ObjectVec->end(), [&](Solid* s){
                return MaterialContainerInst().GetMaterial(s->MaterialIdx)->type ==
                    Material::MaterialType_Light; }
            );
        built = true;
    }
}

void Rasterizer::Rasterize()
{
    Timer RasTime("Rasterizer timer");
    Camera& camera = CameraInst();
    
    for (unsigned y = 0; y < camera.y_res; y++)
    {
        cout << Rubout((int)(100 * (y / (double)camera.y_res)))<< (int)(100 * (y / (double)camera.y_res));
        for (unsigned x = 0; x < camera.x_res; x++)
        {
            Ray ray;

            ray.origin = camera.eye;
            ray.direction = normalize(
                camera.O
                + (x + 0.5f) * camera.dR
                - (y + 0.5f) * camera.dU);

            Raster[x + y*pitch] = ShaderInst().Trace(ray);
        }
    }
}

void Rasterizer::Build()
{
    if (Raster) delete[] Raster;
    
    Raster = new Color[CameraInst().x_res * CameraInst().y_res];
    
    memset(Raster, 0, sizeof(Color)*CameraInst().x_res * CameraInst().y_res);

    width = pitch = CameraInst().x_res, height = CameraInst().y_res;
}

Colorf Shader::Trace(const Ray& ray) const
{
    HitInfo hit;
    memset(&hit, 0, sizeof(HitInfo));
    hit.color = VoidColor;
    hit.hitDistance = std::numeric_limits<float>::infinity();
    
    if (MarkAxes)
        DrawAxes(ray, hit);

    ObjectContainerInst().Cast(ray, hit);
    return hit.color; // return hit color for now
}

void Shader::DrawAxes(const Ray& ray, HitInfo& hit) const
{
    Camera& c = CameraInst();
    if (scaledAlmostEqual(c.xPlaneN * ray.direction, 0, 100))
        hit.hitDistance = -c.eye.x / ray.direction.x, hit.color = AxesColorX;
    else if (scaledAlmostEqual(c.yPlaneN * ray.direction, 0, 100))
        hit.hitDistance = -c.eye.y / ray.direction.y, hit.color = AxesColorY;
    else if (scaledAlmostEqual(c.zPlaneN * ray.direction, 0, 100))
        hit.hitDistance = -c.eye.z / ray.direction.z, hit.color = AxesColorZ;
}

inline ostream& operator <<(ostream& out, const Ray& ray)
{
    out << "|Ray, origin = " << ray.origin
        << " direction = " << ray.direction
        << " intersectionPoint = " << ray.intersectionPt << "|";        
    return out;
}
