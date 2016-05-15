/***************************************************************************
* warp.cpp   (aggregate object plugin)                                     *
*                                                                          *
* The "warp" object is an aggregate object that accepts only a single      *
* child object and accepts a 3x4 matrix as a parameter.  This object       *
* allows arbitrary affine transformations to be applied to any object.     * 
*                                                                          *
* History:                                                                 *
*   04/21/2010  Renamed "transform" to "warp" & made a transform container.*
*   10/03/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "warp.h"

using std::string;
using std::cerr;
using std::endl;

namespace __warp_aggregate__ {  

// Define the Warp object as a sub-class of the "Aggregate" class.
// This sub-class must define all the necessary virtual methods as well as
// any special data members that are specific to this type of object.

struct Warp : Aggregate { 
    Warp() {}
    Warp( const Mat3x4 &, Object *obj = NULL );
   ~Warp() {}
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 & ) const;
    virtual Object *Transform( const Mat3x4 & ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "warp"; }
    virtual void AddChild( Object *obj );
    virtual Object *GetChild() const { return index++ == 0 ? object : NULL; }
    virtual unsigned GetSamples( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) const;
    virtual double Cost() const { return object == NULL ? 1.0 : object->Cost(); }
    Mat3x4 matrix;
    Mat3x4 inverse;
    Object *object;
    };

REGISTER_PLUGIN( Warp );

Warp::Warp( const Mat3x4 &mat, Object *obj )
    {
    // Store the original matrix along with its inverse.  We will need both
    // matrices for transforming points and normals from one space to the other.
    matrix  = mat;
    inverse = Inverse( mat );
    object  = obj;
    }

Plugin *Warp::ReadString( const string &params )
    {
    Mat3x4 M;
    ParamReader get( params );
    if( get["begin"] && get[MyName()] && get[M] ) return new Warp( M );
    return NULL;
    }

// Return a clone of this object that is transformed by M.
Object *Warp::Transform( const Mat3x4 &M ) const
    {
    return new Warp( M * matrix, object );
    }

bool Warp::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // See if the one child is the object we should ignore.
    if( object == ray.ignore ) return false;
    // Transform the given ray back into the canonical space and perform
    // the intersection test there.  When the hit point and normal are found
    // in the canonical space, map them back into the transformed space.
    Ray c_ray( ray ); // Ray inverse-transformed into the canonical space.
    HitInfo c_hit;    // Hit information obtained in the canonical space.
    Vec3 c_dir      = inverse.mat * ray.direction;
    double stretch  = Length( c_dir ); 
    c_ray.origin    = inverse * ray.origin;
    c_ray.direction = c_dir / stretch;
    c_ray.ignore    = ray.ignore;  // Still ignore the same object.
    c_hit.distance  = hitinfo.distance * stretch;

    // Intersect the ray in canonical space with the canonical object.
    if( object->Hit( c_ray, c_hit ) )
        {
        // Return early for visibility rays to skip the inverse transformation.
        if( ray.type == visibility_ray )
            {
            hitinfo.object = c_hit.object;
            return true;
            }
        // Transform the hit info from canonical space back to the
        // original space.  We needn't check that the hit was close
        // enough, because the object intersector already did this.
        hitinfo.distance = c_hit.distance / stretch;
        hitinfo.point    = matrix * c_hit.point;
        hitinfo.normal   = Unit( inverse.mat ^ c_hit.normal );
        hitinfo.object   = c_hit.object;
        return true;
        }
    return false;
    }

bool Warp::Inside( const Vec3 &P ) const
    {
    // Map the point P back into the canonical space and call the
    // object's "Inside" function.
    return object->Inside( inverse * P );
    }

Interval Warp::GetSlab( const Vec3 &v ) const
    {
    // The vector v defining the slab is normal to the slab planes.  Hence, v
    // transforms like a normal vector.  The inverse transpose of the inverse
    // matrix is just the transpose, so we transform v by the transpose of the
    // original matrix (without translation).  This gives u the correct direction,
    // but the length will need to be fixed.
    const Vec3 u( matrix.mat ^ v );
    const Vec3 q( inverse.vec    );

    // Stretch the transformed plane normal appropriately so that lengths
    // relative to it in the transformed space correspond exactly to lengths
    // relative to the original plane normal in the untransformed space.
    const double stretch( ( v * v ) / ( u * u ) );
    const Vec3 w( u * stretch );

    // Account for the fact that the bounding planes may be translated,
    // which is not handled by the plane normal transformation.
    const double offset( ( w * q ) / ( w * w ) );
    return object->GetSlab( w ) - offset;
    }

void Warp::AddChild( Object *obj )
    {
    // The Warp object is intended to be applied to a single object.
    // Complain if more than one child is added.
    if( object == NULL )
        {
        object = obj;
        }
    else
        {
        cerr << "Error: Warp object can accept only one child object." << endl;
	    }
    }

unsigned Warp::GetSamples( const Vec3 &, const Vec3 &, Sample *, unsigned ) const
    {
    cerr << "Error: Getting samples from a transformed object is not implemented." << endl;
    return 0;
    }

} // namespace __warp_aggregate__


// Define an externally accessible function for creating warp objects directly, as they
// are useful for other plug-ins to create (such as the transform container).
Object *MakeWarp( const Mat3x4 &M, Object *obj )
    {
    return new __warp_aggregate__::Warp( M, obj );
    }


