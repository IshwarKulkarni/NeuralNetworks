// Reading the objects

#include "graphics/Solids.cuh"
#include "utils/StringUtils.hxx"
#include "utils/Exceptions.hxx"
#include "utils/Logging.hxx"
#include "geometry/vectors_io.hxx"

#ifdef __CUDACC__
#define DEVHOST __device__ __host__
#include "cuda_runtime.h"
#else
#define DEVHOST 
#endif 

static CommentHogger PCommentHogger = { '#' };

using namespace std;
using namespace StringUtils; 
using namespace Logging;

static inline unsigned getMaterial(istream& strm)
{
    string temp; char sep;
    strm >> PCommentHogger >> sep; // this sep is "

    getline(strm, temp, '\"');
    temp = "\"" + temp + "\""; // this is double quoted name

    unsigned mat = MaterialContainerInst().GetMaterial(temp);
    if (mat != unsigned(-1))
        return mat;

    PrintLocation(strm, Log);
    Log << "Unkonw material above, this solid will be assigned last defined material\n";
    return MaterialContainerInst().GetTopOfStack();
}

void Solid::AddToContainer(){
    ObjectContainerInst().ObjectVec->push_back(this); // make a copy in the array
}

DTriangle* DTriangle::MakeFromStream(std::istream& strm)
{
    DTriangle* t = new DTriangle;
    char sep;
    strm
        >> SComment(t->Primitive.x) >> SComment(sep)
        >> SComment(t->Primitive.y) >> SComment(sep)
        >> SComment(t->Primitive.z) >> SComment(sep);

    make_triangle(t->Primitive);

    string temp;
    bool normalsDefined = false, matDefined = false;

    while (sep != StatementDelimiter && strm.peek() != StatementDelimiter)
    {
        sep = 0;
        strm >> SComment(temp);
        if (CaseInsensitiveMatch()(temp, "normals:"))
        {
            strm
                >> SComment(t->Normals.x) >> SComment(sep)
                >> SComment(t->Normals.y) >> SComment(sep)
                >> SComment(t->Normals.z);
            normalsDefined = true;
        }
        else if (CaseInsensitiveMatch()(temp, "material:"))
            t->MaterialIdx = getMaterial(strm), matDefined = true;
        else
            THROW(UnexpectedLiteralException, "Unknown attribute \"%s\" for DTriangle\n", temp.c_str());
        
        strm >> SComment(sep) >> PCommentHogger;
    }

    if (sep != StatementDelimiter)
        CheckStatementEnd(strm);
    
    if (!matDefined)
        t->MaterialIdx = MaterialContainerInst().GetTopOfStack();
    
    if (!normalsDefined)
        t->Primitive.w = t->Normals.x = t->Normals.y = t->Normals.z = (t->Primitive.e1 ^ t->Primitive.e2);

    t->AddToContainer();
    return t;
}

Triangle* Triangle::MakeFromStream(std::istream& strm)
{
    Triangle* t = new Triangle;
    char sep;
    strm
        >> SComment(t->x) >> SComment(sep)
        >> SComment(t->y) >> SComment(sep)
        >> SComment(t->z) >> SComment(sep);

    make_triangle(*t);

    string temp;
    bool normalsDefined = false, matDefined;

    while (sep != StatementDelimiter && strm.peek() != StatementDelimiter)
    {
        sep = 0;
        strm >> SComment(temp);
        if (CaseInsensitiveMatch()(temp, "material:"))
            t->MaterialIdx = getMaterial(strm), matDefined = true;
        else
            THROW(UnexpectedLiteralException, "Unknown attribute \"%s\" for Triangle\n", temp.c_str());

        strm >> SComment(sep) >> PCommentHogger;
    }

    if (sep != StatementDelimiter)
        CheckStatementEnd(strm);

    if (!matDefined)
        t->MaterialIdx = MaterialContainerInst().GetTopOfStack();
    if (!normalsDefined)
        t->w = (t->e1 ^ t->e2);

    t->AddToContainer();
    return t;
}

Sphere* Sphere::MakeFromStream(std::istream& strm)
{
    Sphere* s = new Sphere;
    char sep = 0;
    string temp;
    bool materialDefined = false;


    strm >> SComment(s->Primitive) >> SComment(sep) >> SComment(s->Primitive.w) >> SComment(sep);
    if (sep != StatementDelimiter && strm.peek() != StatementDelimiter)
    {        
        strm >> SComment(temp) >> PCommentHogger;
        if (CaseInsensitiveMatch()(temp, "material:"))
            s->MaterialIdx = getMaterial(strm), materialDefined = true;
    }

    if (!materialDefined)
        s->MaterialIdx = MaterialContainerInst().GetTopOfStack();

    if ( sep != StatementDelimiter)
        CheckStatementEnd(strm);

    s->AddToContainer();
    return s;
}

DEVHOST bool DTriangle::RayIntersect(Material** mat, const struct Ray& ray, struct HitInfo& hitInfo) const
{
    bool didHit = false;
    Vec3f intVec = Primitive.ray_intersect(ray.origin, ray.direction, didHit);

    if (!didHit || intVec[0] > hitInfo.hitDistance) return false;

    float hitDist = intVec[0];
    
    hitInfo.hitDistance = hitDist;
    hitInfo.color = mat[this->MaterialIdx]->ambient;
    hitInfo.hitObject = this;
    hitInfo.hitPoint = ray.origin + hitDist*ray.direction;
    hitInfo.hitNormal = Normals.x*(1 - intVec[0] - intVec[1]) + 
                        Normals.y*(intVec[0]) + Normals.z*(intVec[1]);

    return true;
}

DEVHOST bool Triangle::RayIntersect(Material** mat, const struct Ray& ray, struct HitInfo& hitInfo) const
{
    bool didHit = false;
    Vec3f intVec = ray_intersect(ray.origin, ray.direction, didHit);

    if (!didHit || intVec[0] > hitInfo.hitDistance) return false;

    float hitDist = intVec[0];

    hitInfo.hitDistance = hitDist;
    hitInfo.color = mat[this->MaterialIdx]->ambient;
    hitInfo.hitObject = this;
    hitInfo.hitPoint = ray.origin + hitDist*ray.direction;
    hitInfo.hitNormal = w;

    return true;
}

DEVHOST bool Sphere::RayIntersect(Material** mat, const struct Ray& ray, struct HitInfo& hitInfo) const
{
    bool didHit = false;
    float dist= Primitive.ray_intersect(ray.origin, ray.direction, didHit);

    if (!didHit) return false;

    float hitDist = dist;

    if (hitDist > hitInfo.hitDistance) return false;

    hitInfo.hitDistance = hitDist;
    hitInfo.color = mat[this->MaterialIdx]->ambient;
    hitInfo.hitObject = this;
    hitInfo.hitPoint = ray.origin + hitDist*ray.direction;
    hitInfo.hitNormal = normalize(hitInfo.hitPoint - *(Vec3f*)(this));

    return true;
}