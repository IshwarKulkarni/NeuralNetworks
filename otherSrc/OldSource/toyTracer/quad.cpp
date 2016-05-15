/***************************************************************************
* quad.cpp    (primitive object plugin)                                    *
*                                                                          *
* The quad object is defined by four (planar) vertices in R3.  This is a   *
* simple flat quad with no normal vector interpolation.                    *
*                                                                          *
* History:                                                                 *
*   04/18/2010  Added Transform method.                                    *
*   04/16/2010  Added external funcion for creating quads.                 *
*   10/23/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"
#include "quad.h"

namespace __quad_primitive__ {  // Ensure that there are no name collisions.

struct Quad : Primitive {
    Quad() {}
    Quad( const Vec3 &A, const Vec3 &B, const Vec3 &C, const Vec3 &D );
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 & ) const { return false; }
    virtual Object *Transform( const Mat3x4 & ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual unsigned GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "quad"; }
	virtual double Cost() const { return .5;}

    private: // Private data and methods...

    Vec3   N;    // Normal to plane of quad;
    double d;    // Distance from origin to plane of quad.
    double area; // Area of the quad.
    Vec3   Eab, Ebc, Ecd, Eda;
    Vec3   A, B, C, D;  
    };

REGISTER_PLUGIN( Quad );

Plugin *Quad::ReadString( const std::string &params ) // Read params from string.
    {
    Vec3 v1, v2, v3, v4;
    ParamReader get( params );
    if( get[MyName()] && get[v1] && get[v2] && get[v3] && get[v4] )
        return new Quad( v1, v2, v3, v4 );
    return NULL;
    }

Quad::Quad( const Vec3 &A_, const Vec3 &B_, const Vec3 &C_, const Vec3 &D_ )
    {
    A = A_; // Store the vertices.
    B = B_;
    C = C_;
    D = D_;

    // Compute the quad normal by summing the triangle normals.  The area of the quad
    // is also computed along the way; it's just 1/2 the length of the resulting vector.
   
    const Vec3 O ( 0.25 * (A + B + C + D) );
    const Vec3 AO( A - O );
    const Vec3 BO( B - O );
    const Vec3 CO( C - O );
    const Vec3 DO( D - O );
    const Vec3 V ((AO ^ BO) + (BO ^ CO) + (CO ^ DO) + (DO ^ AO));
    area = 0.5 * Length( V );
    N = -Unit( V );
    d = O * N;

    // Compute unit vectors in the direction of each edge.  These will be used for the
    // in-out test, to see if a point on the plane of the quad is inside the quad or not.

    Eab = Unit( B - A );
    Ebc = Unit( C - B );
    Ecd = Unit( D - C );
    Eda = Unit( A - D ); 
    }

Interval Quad::GetSlab( const Vec3 &v ) const
    {
    const double a = v * A;
    const double b = v * B;
    const double c = v * C;
    const double d = v * D;
    if( c <= d )
         return Interval( min( a, b, c ), max( a, b, d ) ) / ( v * v );
    else return Interval( min( a, b, d ), max( a, b, c ) ) / ( v * v );
    }

inline bool Quad::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // Compute the point of intersection with the plane containing the quad.
    // Report a miss if the ray does not hit this plane.

    const double denom = ray.direction * N;
    if( fabs(denom) < MachEps ) return false;
    const double s = ( d - ray.origin * N ) / denom;

    if( s <= 0.0 || s > hitinfo.distance ) return false;
    const Vec3 P( ray.origin + s * ray.direction );

    // Compute a sequence of cross products using the quad edges.  The point P is inside
    // the quad if and only if each vector dotted with the quad normal is positive.

    if( (( P - A ) ^ Eab) * N < 0.0 ) return false; 
    if( (( P - B ) ^ Ebc) * N < 0.0 ) return false;
    if( (( P - C ) ^ Ecd) * N < 0.0 ) return false;
    if( (( P - D ) ^ Eda) * N < 0.0 ) return false;

    // We have an actual hit.  Fill in all the geometric information so
    // that the shader can shade this point.

    hitinfo.distance = s;
    hitinfo.point    = P; 
    hitinfo.normal   = N;
    hitinfo.object   = this;
    return true;
    }

// This function generates nxn stratified samples over the surface of the quad.
// The weight of each sample is the quad area / n^2, times the area-to-solid-angle
// conversion factor of cos(theta)/r^2, where theta is the incident angle on the
// quad, and r is the distance between the point O and the given sample P.
// (NOTE: currently this is only correct for parallelograms.)

unsigned Quad::GetSamples( const Vec3 &O, const Vec3 &N1, Sample *samples, unsigned n ) const
    {
    int k = 0;
    double darea = area / ( n * n );  // differential area.
    double delta = 1.0 / n;

    // Compute an n by n grid of stratified (i.e. jittered) samples over the quad.
    // Use bilinear interpolation to parametrize the quad.  Fill in the array of
    // samples as we loop over them, weighting each sample by the correct conversion factor.

    for( unsigned i = 0; i < n; i++ )
    for( unsigned j = 0; j < n; j++ )
        {
        double s = ( i + rand( 0.0, 0.999 ) ) * delta;
        double t = ( j + rand( 0.0, 0.999 ) ) * delta;
        Vec3 P = (1.0 - s) * ( (1.0 - t) * A + t * B ) + s * ( (1.0 - t) * D + t * C );
        Vec3 U = Unit( P - O );
        samples[k].P = P;
        samples[k].w = darea * fabs( N * U ) / LengthSquared( P - O );
        k++;
        }

    // Return the number of samples generated.  For quads, this will always be the
    // square of the number requested.

    return n * n;
    }


// Return a clone of this object that is transformed by Mat.
Object *Quad::Transform( const Mat3x4 &Mat ) const
    {
    return new Quad( Mat * A, Mat * B, Mat * C, Mat * D );
    }

} // namespace __quad_primitive__



Object *MakeQuad( const Vec3 P[] )
    {
    return new __quad_primitive__::Quad( P[0], P[1], P[2], P[3] );
    }

Object *MakeQuad( const Vec3 A,const Vec3 B,const Vec3 C,const Vec3 D)
    {
    return new __quad_primitive__::Quad( A,B,C,D );
    }



