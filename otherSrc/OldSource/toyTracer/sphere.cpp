/***************************************************************************
* sphere.cpp   (primitive object plugin)                                   *
*                                                                          *
* A sphere of given radius and center.  This is one of the most basic      *
* objects to ray trace.                                                    *
*                                                                          *
* To intersect a ray with a sphere with the given center and radius, we    *
* solve the following equation for s: || Q + sR - C || = radius, where Q   *
* is the ray origin, R is the ray direction, and C is the center of the    *
* sphere.  This is equivalent to  ( A + sR )^T ( A + sR ) = radius^2,      *
* where A = Q - C.  Expanding, A.A + 2 s A.R + s^2 R.R = radius^2,         *
* where "." denotes the dot product.  Since R is a unit vercor, R.R = 1.   *
* Rearranging, we have s^2 + (2 A.R) s + (A.A - radius^2) = 0, which is    *
* a quadratic equation in s, the distance along the ray to the point of    *
* intersection.  If this equation has complex roots, then the ray misses   *
* the sphere.  Otherwise, we must determine whether either of the roots    *
* falls on the positive part of the ray, and if so, which is closer.       *
*                                                                          *
* History:                                                                 *
*   10/10/2004  Broken out of objects.C file.                              *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"
#include <cstdlib>
#include <time.h>
#include <math.h>
#include "util.h"

namespace __sphere_primitive__ {  // Ensure that there are no name collisions.

struct Sphere : Primitive {
    Sphere() {}
    Sphere( const Vec3 &center, double radius );
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 &P ) const { return dist( P, center ) <= radius; } 
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual unsigned GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "sphere"; }
	virtual double Cost() const { return 2.0;}
	virtual Ray getRefracetedRay(Ray r,Vec3 P,Vec3 N )const;
	virtual Ray getReflectedRay(Ray r,Vec3 P, Vec3 N )const;
	 HitInfo getFarthestPoint(Vec3, Vec3)const;
    Vec3   center;
    double radius;
    double radius2;
	bool stratified;
    };

// Register the new object with the toytracer.  When this module is linked in, 
// the toytracer will automatically recognize the sphere object.

REGISTER_PLUGIN( Sphere );


Sphere::Sphere( const Vec3 &cent, double rad )
    {
    center  = cent;
    radius  = rad;
    radius2 = rad * rad;
    }

Plugin *Sphere::ReadString( const std::string &params ) // Reads params from a string.
    {
    Vec3 cent;
    double r;
    ParamReader get( params );
    if( get[MyName()] && get[cent] && get[r] ) return new Sphere( cent, r );
    return NULL;
    }

Interval Sphere::GetSlab( const Vec3 &v ) const
    {
    const double len = Length(v);
    const double dot = ( v * center ) / len;
    return Interval( dot - radius, dot + radius ) / len;
    }

bool Sphere::Intersect( const Ray &ray, HitInfo &hitinfo ) const  {
    
	const Vec3 A( ray.origin - center );
    const Vec3 R( ray.direction );
    const double b = 2.0 * ( A * R );
    const double discr = b * b - 4.0 * ( A * A - radius2 );  // The discriminant.

    // If the discriminant if negative, the quadratic equation had negative
    // roots, and the ray misses the sphere.

    if( discr < 0.0 ) return false;

    const double radical = sqrt( discr );

    // First try the smaller of the two roots.  If this is positive, it is the
    // closest intersection.

    double s = 0.5 * ( -b - radical );
    if( s > 0.0 ){
        // If the closest intersection is too far away, report a miss.
        if( s > hitinfo.distance ) return false;
	}
	else{
    
        // Now try the other root, since the smallest root puts the
        // point of intersection "behind" us.  If the larger root is now
        // positive, it means we are inside the sphere.
        s = 0.5 * ( -b + radical );
        if( s <= 0 ) return false; // Both roots are behind.
        if( s > hitinfo.distance ) return false;
	}

    // We have an actual hit.  Fill in all the geometric information so
    // that the shader can shade this point.

    hitinfo.distance = s;
    hitinfo.point    = ray.origin + s * R;
    hitinfo.normal   = Unit( hitinfo.point - center );
    hitinfo.object   = this;
    return true;
}

inline  unsigned Sphere::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const{
   
	Vec3 W = Vec3(center - P);
	Vec3 U = Unit(W);
	double d = Length(W);
	double alpha = asin(radius/d);
	
	double h = cos(alpha);
	double thickness = 1 - h; 
	double weight = 2*Pi*(1-h)/(double)(n*n);
	Vec3 refNormal = Unit(U-Vec3(0,0,1)); //reflecting the samples to move them away from z-axis

	unsigned k = 0;
	double r,theta;


	bool strat = true; ////////////////// stratified always set to true!
	for( unsigned i=0;i<n;i++){
		for(unsigned j=0;j<n;j++){
			if(strat){
				samples[k].P.z  = ((i+rand(0,1))/n)*thickness + h ;
				theta = ((j+rand(0,1))/n)*2*Pi;
			}
			else{
				samples[k].P.z = rand(h,1.0);
				theta = rand(0,2*Pi);
			}

			r = sqrt(1-samples[k].P.z*samples[k].P.z);			
						
			samples[k].P.x = r*cos(theta);
			samples[k].P.y = r*sin(theta);
			
			samples[k].P = samples[k].P  - ( 2.0 * ( samples[k].P  * refNormal ) ) * refNormal;
			//reflect about the normal to plane of rflection


			samples[k].P += P; //translate the point
			samples[k++].w = weight;

	
		}
	}
	return n*n;
}


Ray Sphere::getRefracetedRay(Ray r,Vec3 P,Vec3 Normal )const{

	Ray iRay; //internal Ray
	Vec3 N = Normal;
	iRay.origin = P;
	iRay.direction = getRefractedDirection(r.direction,N,this->material->ref_index);
	if(iRay.direction == Vec3(0,0,0))
		return getReflectedRay(r,P,N);
	
	Vec3 iHit = P;
	Vec3 D ;
	int i=0;
	do{
		D = iRay.direction;
		iHit += getFarthestPoint(iHit,D).distance *D;
		N = Unit(center - iHit);
		D = getRefractedDirection(iRay.direction,N,1.0/this->material->ref_index);
		if(D == Vec3(0,0,0))
			iRay.direction = iRay.direction- ( 2.0 * ( iRay.direction* N) ) * N;
		i++;
		}while(i<1 || D == Vec3(0,0,0));

	iRay.from = r.from;
	iRay.generation = r.generation;
	iRay.ignore = this;
	iRay.origin = iHit+iRay.direction*10E-4;
	iRay.type = refracted_ray;

	return iRay;
}

inline HitInfo Sphere::getFarthestPoint(Vec3 P, Vec3 R)const{
	
	const Vec3 A( P - center );
    const double b = 2.0 * ( A * R );
    const double discr = b * b - 4.0 * ( A * A - radius2 );  // The discriminant.

    const double radical = sqrt( discr );

    
	double s = max( 0.5 * ( -b - radical ), 0.5 * ( -b + radical ));
	HitInfo h;
	h.distance = s;
	return h;
}
//gives the refracted ray, behaviour undefined if the original ray "misses" this sphere.
//call Intersect method and if the ray intersects the block, then call this method 
//witht the same ray.

inline Ray Sphere::getReflectedRay(Ray r, Vec3 P, Vec3 N) const{

	Ray rr(r);
	rr.direction = r.direction- ( 2.0 * ( r.direction* N) ) * N;
	rr.origin = P + rr.direction*10E-4;
	rr.ignore = this;
	rr.type = reflected_ray;

	return rr;
}


} //namespace __sphere_primitive__

