/***************************************************************************
* triangle.cpp    (primitive object plugin)                                *
*                                                                          *
* The triangle object is defined by three vertices in R3.  This is a       *
* simple flat triangle with no normal vector interpolation.  The           *
* triangle structure is defined to accommodate the barycentric coord       *
* method of intersecting a ray with a triangle.                            *
*                                                                          *
* History:                                                                 *
*   04/16/2010  Added vertex normals & external funcs for making triangles.*
*   10/03/2005  Removed bounding box computation.                          *
*   10/10/2004  Broken out of objects.C file.                              *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"
#include "triangle.h"

namespace __triangle_primitive__ {  // Ensure that there are no name collisions.

struct Triangle : Primitive {
    Triangle() { V = NULL; }
    Triangle( const Vec3 &A, const Vec3 &B, const Vec3 &C );
    Triangle( const Vec3 P[], const Vec3 N[] );
    virtual ~Triangle() { delete[] V; }
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 & ) const { return false; }
    virtual Object *Transform( const Mat3x4 & ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual int GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "triangle"; }
    bool WithNormals() const { return V != NULL; }
	virtual double Cost() const { return .5;}
    private:  // Private data and methods... 

    void init( const Vec3 &A, const Vec3 &B, const Vec3 &C );
    Mat3x3 M;    // Inverse of barycentric coord transform.
    Vec3   N;    // Normal to plane of the triangle;
    double d;    // Distance from origin to plane of triangle.
    int    axis; // The dominant axis;
    Vec3   A;    // Vertex.
    Vec3   B;    // Vertex.
    Vec3   C;    // Vertex.
    Vec3  *V;    // Vertex normals, if supplied.
    };

Plugin *Triangle::ReadString( const std::string &params ) // Read params from string.
    {
    ParamReader get( params );
    Vec3 p[3], n[3];
    if( get[MyName()] && get[p[0]] && get[p[1]] && get[p[2]] )
        {
        // See if normal vectors have also been supplied.
        if( get[n[0]] && get[n[1]] && get[n[2]] )
             return new Triangle( p, n ); // Triangle with normals.
        else return new Triangle( p[0], p[1], p[2] );
        }
    return NULL;
    }

// Create a flat triangle without vertex normals.
Triangle::Triangle( const Vec3 &A_, const Vec3 &B_, const Vec3 &C_ )
    {
    init( A_, B_, C_ );
    V = NULL; // This signals that there are no vertex normals.
    }

// Create a triangle with associated vertex normals.
Triangle::Triangle( const Vec3 Pnt[], const Vec3 Nrm[] )
    {
    init( Pnt[0], Pnt[1], Pnt[2] );
    V = new Vec3[3];
    V[0] = Unit( Nrm[0] );
    V[1] = Unit( Nrm[1] );
    V[2] = Unit( Nrm[2] );
    }

void Triangle::init( const Vec3 &A_, const Vec3 &B_, const Vec3 &C_ )
    {
    A = A_; // Store the vertices.
    B = B_;
    C = C_;
    V = NULL;

    N = Unit( (A - B) ^ (C - B) ); // Compute the normal vector.
    d = A * N;  // Compute the distance from the origin to the plane.

    // Determine which axis is the "dominant" axis of the triangle.  That is, which
    // coordinate axis is closest to being orthogonal to the triangle.

    if( fabs(N.x) >= fabs(N.y) && fabs(N.x) >= fabs(N.z) ) axis = 0; else
    if( fabs(N.y) >= fabs(N.x) && fabs(N.y) >= fabs(N.z) ) axis = 1; else
    if( fabs(N.z) >= fabs(N.x) && fabs(N.z) >= fabs(N.y) ) axis = 2;

   // Set up the matrix for finding the barycentric coordinates.

    Mat3x3 W;
    W(0,0) = A.x; W(0,1) = B.x; W(0,2) = C.x;
    W(1,0) = A.y; W(1,1) = B.y; W(1,2) = C.y;
    W(2,0) = A.z; W(2,1) = B.z; W(2,2) = C.z;
    W(axis,0) = 1.0;
    W(axis,1) = 1.0;
    W(axis,2) = 1.0;

    // Store the inverse of this matrix.  It will provide an efficient means of
    // finding the barycentric coordinates of the point of intersection with the
    // plane containing the triangle.

    M = Inverse( W );
    }

Interval Triangle::GetSlab( const Vec3 &v ) const
    {
    const double a( v * A );
    const double b( v * B );
    const double c( v * C );
    return Interval(
        OneMinusEps * min( a, b, c ),
        OnePlusEps  * max( a, b, c )
        ) / ( v * v );
    }

bool Triangle::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // Compute the point of intersection with the plane containing the triangle.
    // Report a miss if the ray does not hit this plane.

    const double denom( ray.direction * N );
    if( fabs(denom) < 1.0E-4 ) return false;
    const double s( ( d - ray.origin * N ) / denom );
    if( s <= 0.0 || s > hitinfo.distance ) return false;
    const Vec3 P( ray.origin + s * ray.direction );

    // Create a new vector that is a copy of P, but with one coordinate (corresponding
    // to the dominant exis) set to 1.  This is the right-hand-side of the equation for
    // the barycentric coordinates.

    Vec3 Q( P );
    switch( axis )
        {
        case 0: Q.x = 1.0; break;
        case 1: Q.y = 1.0; break;
        case 2: Q.z = 1.0; break;
        }

    // Solve for the barycentric coordinates and check their sign.  The ray hits
    // the triangle if and only of all the barycentric coordinates are non-negative.

    const double a( M(0,0) * Q.x + M(0,1) * Q.y + M(0,2) * Q.z );
    if( a < 0.0 ) return false;

    const double b( M(1,0) * Q.x + M(1,1) * Q.y + M(1,2) * Q.z );
    if( b < 0.0 ) return false;

    const double c( M(2,0) * Q.x + M(2,1) * Q.y + M(2,2) * Q.z );
    if( c < 0.0 ) return false;

    // We have an actual hit.  Fill in all the geometric information so that
    // the shader can shade this point.  If normal vectors have been provided,
    // interpolate them using the barycentric coords (a,b,c) to compute the
    // normal at P.  Otherwise, just use the normal to the triangle.

    hitinfo.distance = s;
    hitinfo.point    = P; 
    hitinfo.normal   = ( V == NULL ) ? N : Unit( a * V[0] + b * V[1] + c * V[2] );
    hitinfo.object   = this;
    return true;
    }

int Triangle::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const
    {
    int count = 0;
    Vec3    W = (A - B) ^ (C - B);
    double dA = Length( W ) / ( 2.0 * n * n );
    W = Unit( W );
    for( int i = 0; i < n; i++ )
    for( int j = 0; j < n; j++ )
        {
        double s = sqrt( ( i + rand(0,1) ) / n );
        double t =       ( j + rand(0,1) ) / n;
        Vec3 Q = (1.0 - s) * A + (s - s * t) * B + ( s * t ) * C;
        Vec3 R = Q - P;
        double d = Length( R );
        double c = fabs( R * W / d );
        samples[ count ].P = Q;
        samples[ count ].w = dA * c / ( d * d );
        count++;
        }
    return n * n;
    }

// Return a clone of this object that is transformed by Mat.
Object *Triangle::Transform( const Mat3x4 &Mat ) const
    {
    const Vec3 Q[] = { Mat * A, Mat * B, Mat * C };
    if( WithNormals() )
        {
        Mat3x3 W( Inverse( Mat.mat ) );
        Vec3 U[] = { W ^ A, W ^ B, W ^ C }; 
        return new Triangle( Q, U );
        }
    return new Triangle( Q[0], Q[1], Q[2] );
    }
    
// Register the new object with the toytracer.  When this module is linked in, the 
// toytracer will automatically recognize the new objects and read them from sdf files.

REGISTER_PLUGIN( Triangle );

} // namespace __triangle_primitive__


// Provide a convenient way for other objects to create flat triangles and triangles
// with normals.

Object *MakeTriangle( const Vec3 P[] )
    {
    return new __triangle_primitive__::Triangle( P[0], P[1], P[2] );
    }

Object *MakeTriangle( const Vec3 P[], const Vec3 N[] )
    {
    // Create a triangle with vertex normals.
    return new __triangle_primitive__::Triangle( P, N );
    }

Object *MakeTriangle(const Vec3 &v1,const Vec3 &v2,const Vec3 &v3 )
    {
		
		return new __triangle_primitive__::Triangle(v1,v2,v3);
    }

