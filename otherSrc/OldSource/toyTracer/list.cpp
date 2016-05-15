/***************************************************************************
* list.cpp   (aggregate object plugin)                                     *
*                                                                          *
* The "list" object is a trivial aggregate object.  It simply collects     *
* all children into a list (using the std::vector class, which is supplied *
* by the Aggregate class by default) and tests each ray against each       *
* object in the list.  It places a single bounding box around the entire   *
* collection, but not around each individual object.  This amounts to      *
* brute-force ray tracing.                                                 *
*                                                                          *
* History:                                                                 *
*   10/16/2004  Changed the way the bounding box is computed.              *
*   10/16/2004  Added more documentation, and call to "Inverse" function.  *
*   10/06/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"

namespace __list_aggregate__ {

// Define the actual List object as a sub-class of the "Aggregate" class.
// This sub-class must define all the necessary virtual methods as well as
// any special data members that are specific to this type of object.
// (ALL access to this object will be through these virtual methods.)

struct list : Aggregate { 
    list() { bbox = AABB::Null(); }
    virtual ~list() {}
    virtual bool Intersect( const Ray &ray, HitInfo & ) const;
    virtual bool Inside( const Vec3 & ) const;
    virtual Interval GetSlab( const Vec3 & ) const;
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "List"; }
    virtual void AddChild( Object * );
    virtual double Cost() const;
    AABB bbox;
    };

// Register the new object with the toytracer.  When this module is linked in, the 
// toytracer will automatically recognize the new objects and read them from sdf files.
REGISTER_PLUGIN( list );

// Fill in all the virtual methods for the List object...

Plugin *list::ReadString( const std::string &params )
    {
    ParamReader get( params );
    // Every aggregate object should look for a "begin" preceding its name.
    // An aggregate can also accept additional parameters on this line (e.g. the
    // size of a uniform grid, maximum depth of a BSP tree, etc.).  It is recommended
    // that missing parameters default to something reasonable.
    if( get["begin"] && get[MyName()] ) return new list();
    return NULL;
    }

bool list::Intersect( const Ray &ray, HitInfo &hitinfo ) const
    {
    // If the ray does not hit the bounding box, then the ray
    // misses the contents.
    if( !bbox.Hit( ray, hitinfo.distance ) ) return false;

    bool found_a_hit( false );

    // Each intersector is ONLY allowed to write into the "HitInfo"
    // structure if it has determined that the ray hits the object
    // at a CLOSER distance than currently recorded in HitInfo.distance.
    // When a closer hit is found, the fields of the "HitInfo" structure
    // are updated to hold information about the object that was just hit.
    // Return "true" immediately on ANY hit if we are simply testing for
    // visibility.  In this case, the only field of the HitInfo that is
    // updated will be the object pointer.

    for( int i = 0; i < Size(); i++ )
        {
        const Object *obj( children[i] );
        if( obj != ray.ignore && obj->Hit( ray, hitinfo ) )
            {
            // We've hit an object, but continue looking unless this ray is for
            // testing visibility only, in which case we do not need the closest hit.
            if( ray.type == visibility_ray ) return true;
            found_a_hit = true; 
            }
        }

    // No need to fill in any fields of hitinfo, as the closest object
    // that was hit (if there was one) will have already done so.
    return found_a_hit;
    }

// Determine whether the point P is in the interior of the object.
bool list::Inside( const Vec3 &P ) const
    {
    // If the point is not inside the bounding box of the list, then
    // there is no need to check the child objects.
    if( !bbox.Contains( P ) ) return false;
    for( int i = 0; i < Size(); i++ )
        {
        const Object *obj = children[i];
        if( obj->Inside( P ) ) return true;
        }
    return false;
    }

// Return the min & max extent of the object in the direction v.
Interval list::GetSlab( const Vec3 &v ) const
    {
    Interval I( Interval::Null() );
    for( int i = 0; i < Size(); i++ )
        {
        // Expand the interval for this slab to ensure that it
        // contains all the intervals enclosing the child objects.
        I << children[i]->GetSlab(v);
        }
    return I;
    }

// Since the List object simply tests all its children, in the order they
// were given, the cost of intersecting a ray with this type of object is
// the sum of the costs of all the children.  (Hence, this is a VERY
// inefficient aggregate object for a large number of objects.)
double list::Cost() const
    {
    double cost( 0.0 );
    for( int i = 0; i < Size(); i++ )
        cost += children[i]->Cost();
    return cost;
    }

// The list object supplies its own AddChild method so that it can
// construct a bounding box containing all the child objects as they
// are added.
void list::AddChild( Object *obj )
    {
    children.push_back( obj );
    // Grow the bounding box, if necessary, with each child added.
    bbox << GetBox( *obj );  
    }


} // namespace __list_aggregate__




