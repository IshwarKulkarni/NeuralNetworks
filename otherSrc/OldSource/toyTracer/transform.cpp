/***************************************************************************
* transform.cpp   (container object plugin)                                *
*                                                                          *
* The "transform" object is a container that transforms its chilren with   *
* a 3x4 matrix.  The "Transform" method of each child is first called      *
* to see if the object can transform itself.  If it can, then it is the    *
* transformed clone that is added to the container.  If it cannot, the it  *
* is a "warp" object containing the original object that is added to the   *
* container.                                                               *
*                                                                          *
* History:                                                                 *
*   04/20/2010  Changed transform from an Aggregate to a Container.        *
*   10/03/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "warp.h"

namespace __transform_container__ {  // Ensure that there are no name collisions.

struct transform : Container {
    transform() {}
    transform( const Mat3x4 &M ) { Mat = M; }
    virtual ~transform() {}
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "transform"; }
    virtual void AddChild( Object *obj );
    Mat3x4 Mat;
    };

Plugin *transform::ReadString( const std::string &params )
    {
    Mat3x4 M;
    ParamReader get( params );
    if( get["begin"] && get[MyName()] && get[M] ) return new transform( M );
    return NULL;
    }

void transform::AddChild( Object *obj )
    {
    Object *new_obj = obj->Transform( Mat );
    if( new_obj == NULL )
        {
        // The object was unable to create a transformed clone of itself.
        // Instead, create a warp object that will transform the object.
        new_obj = MakeWarp( Mat, obj );
        }
    else
        {
        // The object cloned itself.  Ensure that the new clone has the
        // same attributes as the original object.
        CopyAttributes( new_obj, obj );
        }
    // Push either a transformed clone of the object, or a warped
    // version of the original object.
    children.push_back( new_obj );
    }

REGISTER_PLUGIN( transform );

} // namespace __transform_container__


