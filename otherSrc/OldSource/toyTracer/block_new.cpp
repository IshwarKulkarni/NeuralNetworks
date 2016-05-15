
/***************************************************************************
* blockNew.cpp    (primitive object plugin)                                *
*                                                                          *
* The "Block" object defines an axis-aligned box.  Its constructor takes   *
* two vectors, which are taken to be the "min" and "max" corners of the    *
* Block (i.e. the three min coords, and the three max coords).   
*
* This Block_new was written to handle refraction                                                                          *
* History:                                                                 *
*   10/10/2004  Split off from objects.cpp file.                           *
*                                                                          *
***************************************************************************/
#include <string>
#include "quad.h"
#include "toytracer.h"
#include "util.h"
#include "params.h"
//#include "RayPair.h"

#define SWAP(a,b,c) {c t;t=(a);(a)=(b);(b)=t;}

struct blockNew : Primitive {
    blockNew() {}
    blockNew( const Vec3 &Min, const Vec3 &Max );
    virtual bool Intersect( const Ray &ray, HitInfo &hitinfo ) const;
    virtual bool Inside( const Vec3 &P ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
	virtual int GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const{return 0;}
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "block"; }
	double Cost() const { return 2.0;}
	virtual Ray getRefracetedRay(Ray r,Vec3 N )const;
	virtual Ray getReflectedRay(Ray r,Vec3, Vec3 N )const;
	
	Vec3 Min,Max;
};

REGISTER_PLUGIN( blockNew );


blockNew::blockNew( const Vec3 &Min_, const Vec3 &Max_ ){
    Min  = Min_;
    Max  = Max_;

}

Plugin *blockNew::ReadString( const std::string &params ){
    ParamReader get( params );
    Vec3 Vmin, Vmax;
    if( get[MyName()] && get[Vmin] && get[Vmax] ) return new blockNew( Vmin, Vmax );
    return NULL;
    }

bool blockNew::Inside( const Vec3 &P ) const{
    
	if( P.x < Min.x || P.x > Max.x ) return false;
    if( P.y < Min.y || P.y > Max.y ) return false;
    if( P.z < Min.z || P.z > Max.z ) return false;
    return true;
    }

Interval blockNew::GetSlab( const Vec3 &v ) const{
	const double
    ax = v.x * Min.x,
    ay = v.y * Min.y,
    az = v.z * Min.z,
    bx = v.x * Max.x,
    by = v.y * Max.y,
    bz = v.z * Max.z,
    a  = min( ax, bx ) + min( ay, by ) + min( az, bz ), 
    b  = max( ax, bx ) + max( ay, by ) + max( az, bz ); 
	return Interval( a, b ) / ( v * v );
}

inline bool blockNew::Intersect( const Ray &ray, HitInfo &hitinfo ) const{
		
	const Vec3 R( ray.direction );
	const Vec3 Q( ray.origin );

	Vec3 SMinXnorm(-1,0,0), SMaxXnorm(1,0,0); 
	Vec3 SMinYnorm(0,-1,0), SMaxYnorm(0,1,0); 
	Vec3 SMinZnorm(0,0,-1), SMaxZnorm(0,0,1);

    Vec3 NormalMin; //finally this will contain normal to face of intersection
    Vec3 NormalMax; //this will contain normal to face where ray will leave the block

    double SMinX=(Min.x-Q.x)/R.x, SMaxX=(Max.x-Q.x)/R.x;
    if(SMinX>SMaxX){
		SWAP(SMinX,SMaxX,double)
        SWAP(SMinXnorm,SMaxXnorm,Vec3);
    }
	
	double SMinY=(Min.y-Q.y)/R.y, SMaxY=(Max.y-Q.y)/R.y;
    if(SMinY>SMaxY){
        SWAP(SMinY,SMaxY,double)
        SWAP(SMinYnorm,SMaxYnorm,Vec3);
    }
 
	double SMinZ=(Min.z-Q.z)/R.z, SMaxZ=(Max.z-Q.z)/R.z;
    if(SMinZ>SMaxZ){
        SWAP(SMinZ,SMaxZ,double)
        SWAP(SMinZnorm,SMaxZnorm,Vec3);
    }

	
    double shortest = max(SMinX,SMinY,SMinZ);
    double longest = min(SMaxX,SMaxY,SMaxZ);

	if( shortest == SMinX)
		NormalMin = SMinXnorm;

	if( shortest == SMinY)
		NormalMin = SMinYnorm;

	if( shortest == SMinZ)
		NormalMin = SMinZnorm;

	if( longest == SMaxX)
		NormalMax = SMaxXnorm;

	if( longest == SMaxY)
		NormalMax = SMaxYnorm;

	if( longest == SMaxZ)
		NormalMax = SMaxZnorm;

    if(longest<shortest)
		return false;
    
    
    if(shortest<0){
        SWAP(shortest,longest,double);
        SWAP(NormalMin,NormalMax,Vec3);
    }

    if(shortest<=0) return false;

    if(shortest>hitinfo.distance) return false; 

	hitinfo.distance = shortest;
    hitinfo.point    = Q + shortest* R;
    hitinfo.normal   = NormalMin;
    hitinfo.object   = this;
    return true;
}

Vec3 getFarthestPoint(Ray incoming, Vec3 Min, Vec3 Max){

	const Vec3 R( incoming.direction );
	const Vec3 Q( incoming.origin );

	Vec3 SMinXnorm(-1,0,0), SMaxXnorm(1,0,0); 
	Vec3 SMinYnorm(0,-1,0), SMaxYnorm(0,1,0); 
	Vec3 SMinZnorm(0,0,-1), SMaxZnorm(0,0,1);

    Vec3 NormalMin; //finally this will contain normal to face of intersection
    Vec3 NormalMax; //this will contain normal to face where ray will leave the block

    double SMinX=(Min.x-Q.x)/R.x, SMaxX=(Max.x-Q.x)/R.x;
    if(SMinX>SMaxX){
		SWAP(SMinX,SMaxX,double)
        SWAP(SMinXnorm,SMaxXnorm,Vec3);
    }
	
	double SMinY=(Min.y-Q.y)/R.y, SMaxY=(Max.y-Q.y)/R.y;
    if(SMinY>SMaxY){
        SWAP(SMinY,SMaxY,double)
        SWAP(SMinYnorm,SMaxYnorm,Vec3);
    }
 
	double SMinZ=(Min.z-Q.z)/R.z, SMaxZ=(Max.z-Q.z)/R.z;
    if(SMinZ>SMaxZ){
        SWAP(SMinZ,SMaxZ,double)
        SWAP(SMinZnorm,SMaxZnorm,Vec3);
    }

	
    double shortest = max(SMinX,SMinY,SMinZ);
    double longest = min(SMaxX,SMaxY,SMaxZ);

	if( shortest == SMinX)
		NormalMin = SMinXnorm;

	if( shortest == SMinY)
		NormalMin = SMinYnorm;

	if( shortest == SMinZ)
		NormalMin = SMinZnorm;

	if( longest == SMaxX)
		NormalMax = SMaxXnorm;

	if( longest == SMaxY)
		NormalMax = SMaxYnorm;

	if( longest == SMaxZ)
		NormalMax = SMaxZnorm;
    
    if(shortest<0){
        SWAP(shortest,longest,double);
        SWAP(NormalMin,NormalMax,Vec3);
    }

	return Q + longest* R;
}
inline Ray blockNew::getReflectedRay(Ray ray, Vec3 hitpoint,  Vec3 N)const{

	 Ray rr(ray);
	 rr.direction = Unit( ( 2.0 * ( ray.direction * N ) ) * N - ray.direction );
	 rr.origin = hitpoint + N * 10E-4;
	rr.ignore = this;
	 return rr;

}

//gives the refracted ray, behaviour unddefined the original ray "misses" this block.
//call Intersect method and if the ray intersects the block, then call this method 
//witht the same ray.
inline Ray blockNew::getRefracetedRay(Ray ray, Vec3 N)const{
	Ray rr(ray);
	rr.direction = Vec3(0,0,0);
	Vec3 V = ray.direction;
	double c1 = -( N* V );
	Vec3 Rl = V + (2 * N * c1 );

	double refIndex = this->material->ref_index; //assuming incoming ray is coming from air(vaccum)
	double root = 1.0 - refIndex * (1 - c1*c1);
	if(root <0)
		return rr;
	double c2 = sqrt( root );
	rr.direction = (refIndex * V) + (refIndex * c1 - c2) * N;

//	std::cout<<getFarthestPoint(rr, Min, Max) ;

	rr.origin = getFarthestPoint(rr, Min, Max) ;
	rr.direction = ray.direction;

	rr.origin += rr.direction*10E-4;
	rr.ignore = this;
	
	return rr;
}