
/***************************************************************************
* block_s.cpp    (primitive object plugin)                                   *
*                                                                          *
* The "Block" object defines an axis-aligned box.  Its constructor takes   *
* two vectors, which are taken to be the "min" and "max" corners of the    *
* Block (i.e. the three min coords, and the three max coords).             *
*                                                                          *
* History:                                                                 *
*   10/10/2004  Split off from objects.cpp file.                           *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define _max(a, b, c) MAX( MAX(a,b), c)
#define _min(a, b, c) MIN( MIN(a,b), c)
#define SWAP(a,b,c) {c t;t=(a);(a)=(b);(b)=t;}

namespace __block_primitive__ {  // Ensure that there are no name collisions.

struct block_s : Primitive {
    block_s() {}
    block_s( const Vec3 &Min, const Vec3 &Max );
    virtual bool Intersect( const Ray &ray, HitInfo &hitinfo ) const;
    virtual bool Inside( const Vec3 &P ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual int GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "block_s"; }
	virtual double Cost() const { return 2.0;}
	Vec3 Min; // Minimum coordinates along each axis.
    Vec3 Max; // Maximum coordinates along each axis.
    };

// Register the new object with the toytracer.  When this module is linked in, the 
// toytracer will automatically recognize the new objects and read them from sdf files.

REGISTER_PLUGIN( block_s );

block_s::block_s( const Vec3 &Min_, const Vec3 &Max_ )
    {
    Min  = Min_;
    Max  = Max_;
    }

Plugin *block_s::ReadString( const std::string &params ) // Read params from string.
    {
    ParamReader get( params );
    Vec3 Vmin, Vmax;
    if( get[MyName()] && get[Vmin] && get[Vmax] ) 
		return new block_s( Vmin, Vmax );
    return NULL;
    }

// Determine whether the given point is on or in the object.
bool block_s::Inside( const Vec3 &P ) const
    {
    if( P.x < Min.x || P.x > Max.x ) return false;
    if( P.y < Min.y || P.y > Max.y ) return false;
    if( P.z < Min.z || P.z > Max.z ) return false;
    return true;
    }

Interval block_s::GetSlab( const Vec3 &v ) const
    {
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

bool block_s::Intersect( const Ray &ray, HitInfo &hitinfo ) const{
	

	const Vec3 R( ray.direction );

    //sxminnorm,sxmaxnorm etc represents the normals for  different faces.
    double sxmin=(Min.x-ray.origin.x)/ray.direction.x;
    Vec3 sxminnorm(-1,0,0);
    double sxmax=(Max.x-ray.origin.x)/ray.direction.x;
    Vec3 sxmaxnorm(1,0,0);
    double symin=(Min.y-ray.origin.y)/ray.direction.y;
    Vec3 syminnorm(0,-1,0);
    double symax=(Max.y-ray.origin.y)/ray.direction.y;
    Vec3 symaxnorm(0,1,0);
    double szmin=(Min.z-ray.origin.z)/ray.direction.z;
    Vec3 szminnorm(0,0,-1);
    double szmax=(Max.z-ray.origin.z)/ray.direction.z;
    Vec3 szmaxnorm(0,0,1);

    double temp;//temporary variable
    double tmin;//max(sxmin,symin,szmin)
    double tmax;//min(sxmax,symax,szmax)

    Vec3 tnormin;//normal for the face which gives tmin
    Vec3 tnormax;//normal for the face which give tmax
    Vec3 tnorm;//temporary variable

    //sxmin should contain the minimum distance of intersection with x=Min.x and x=Max.x. If it is greater than sxmax
    //they are exchanged. Similar thing is done for y and z directions.
    if(sxmin>sxmax)
    {
            temp=sxmin;
            sxmin=sxmax;
            sxmax=temp;

            tnorm=sxminnorm;
            sxminnorm=sxmaxnorm;
            sxmaxnorm=tnorm;
    }

    if(symin>symax)
    {
            temp=symin;
            symin=symax;
            symax=temp;

            tnorm=syminnorm;
            syminnorm=symaxnorm;
            symaxnorm=tnorm;
    }

    if(szmin>szmax)
    {
            temp=szmin;
            szmin=szmax;
            szmax=temp;

            tnorm=szminnorm;
            szminnorm=szmaxnorm;
            szmaxnorm=tnorm;
    }

    //find tmin=max(sxmin,symin,szmin)
    if(sxmin>symin)
    {
            if(szmin>sxmin)
            {
                    tmin=szmin;
                    tnormin=szminnorm;
            }
            else
            {
                    tmin=sxmin;
                    tnormin=sxminnorm;

            }
    }
    else
    {
            if(szmin>symin)
            {
                    tmin=szmin;
                    tnormin=szminnorm;
            }
            else
            {
                    tmin=symin;
                    tnormin=syminnorm;
            }
    }

    //find tmax=min(sxmax,symax,szmax)
    if(sxmax<symax)
    {
            if(szmax<sxmax)
            {
                    tmax=szmax;
                    tnormax=szmaxnorm;
            }
            else
            {
                    tmax=sxmax;
                    tnormax=sxmaxnorm;
            }
    }
    else
    {
            if(szmax<symax)
            {
                    tmax=szmax;
                    tnormax=szmaxnorm;
            }
            else
            {
                    tmax=symax;
                    tnormax=symaxnorm;
            }
    }

    if(tmax<tmin)
    {
            return false;
    }

    //if the minimum root is -ve then try other root
    if(tmin<0)
    {
            tmin=tmax;
            tnormin=tnormax;
    }

    //if other root is also -ve return false
    if(tmin<=0) return false;

    if(tmin>hitinfo.distance)
    {
            return false;
    }

    //populate the hitinfo struct
    hitinfo.distance = tmin;
    hitinfo.point    = ray.origin + tmin * R;
    hitinfo.normal   = tnormin;
    hitinfo.object   = this;
    return true;
        
    }

int block_s::GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, int n ) const
    {
    // To be filled in later.
    return 0;
    }

} // namespace __block_primitive__

