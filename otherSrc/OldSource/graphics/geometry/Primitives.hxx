#ifndef _PRIMITIVES_GEOMETRY_HXX_
#define _PRIMITIVES_GEOMETRY_HXX_

#include "vectors.hxx"
#include "cmath"
#ifdef __CUDACC__

#define finf __int_as_float(0x7f800000)
#define fnan __int_as_float(0x7fc00000)
#define DEVICE  __device__
#else

union uintToFloat{
    unsigned u;
    float f;
};
#define DEVICE

#endif

struct SpherePrimitive :  Vec4f
{
    float& radius(){ return w; }
    const float& radius() const { return w; }
    SpherePrimitive(float a, float b, float c, float r){ x = a; y = b; z = c; w = r; }
    SpherePrimitive(float v[4]) { x = v[0]; y = v[1]; z = v[2]; w = v[3]; }
    SpherePrimitive() { x = 0; y = 0; z = 0; w = 0; }
    FINLINE float ray_intersect(const Vec3f&  O, const Vec3f& D, bool& intersects) const
    {
        intersects = false;
        auto A = (O - *this);
        auto b = 2.0 * (A * D);
        auto radius2 = w*w;
        auto discr = b * b - 4.0 * (A * A - radius2); 
                

        if (discr > 0.0)
        {
            float radical = float(sqrt(discr));

            float s = float(0.5f * (-b - radical));
            if (s > 0.0)
            {
                intersects = true;
                return s;
            }

            s = float(0.5f * (-b + radical));

            if (s > 0.0)
            {
                intersects = true;
                return s;
            }

            intersects = false;
            return s;
        }
        else
            return -1;
    }
};

struct TrianglePrimitive : Vec < Vec3f >
{
    Vec3f e1, e2;

    FINLINE Vec3f ray_intersect(const Vec3f&  O, const Vec3f& D, bool &intersects) const
    {
        intersects = false;
        auto P = D ^ e2;
        auto det = e1*P;
        
        if (det <= 0)
            return P; // garbage

        Vec3f T = O - x;

        auto u = T*P ;

        if (u < 0.f || u > det)
            return T; //garbage

        Vec3f Q = T ^ e1;

        auto v = D*Q;

        if (v < Epsilon || u + v > det)
            return T; // garbage

        
        intersects = true;
        det = 1 / det;
        return Vec3f( (e2* Q), u, v) * det;
    }
};

FINLINE void make_triangle(TrianglePrimitive& p) { p.e1 = p.y - p.x, p.e2 = p.z - p.x; }

FINLINE TrianglePrimitive make_triangle(const Vec3f& a, const Vec3f& b, const Vec3f& c)
{
    TrianglePrimitive t;
    t.x = a;
    t.y = b;
    t.z = c;

    t.e1 = b - a, t.e2 = c - a;
    return t;
}

FINLINE TrianglePrimitive make_triangle(const Vec3f* v) { return make_triangle(v[0], v[1], v[2]); }

#endif // !_PRIMITIVES_GEOMETRY_HXX_
