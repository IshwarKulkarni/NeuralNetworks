/***************************************************************************
* basic_envmap.cpp    (environment map plugin)                             *
*                                                                          *
* An "environment map" is a function that associates a color with any      *
* given ray, often by table-lookup (i.e. by indexing colors in a image     *
* or a high-dynamic-range map of an environment.  This "basic" version     *
* simply returns a single color, with no lookup or other computation.      *
* Its fixed color is specified as a parameter to the envmap.               *
*                                                                          *
* History:                                                                 *
*   09/27/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "color.h"
#include "params.h"

namespace __basic_envmap__ {

struct basic_envmap : Envmap {
    inline  basic_envmap() : color( 0.15, 0.25, 0.35 ) {}
    inline  basic_envmap( const Color &c ) : color(c) {}
    inline ~basic_envmap() {}
    virtual Color Shade( const Ray & ) const { return color; }
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "basic_envmap"; }
    virtual bool Default() const { return true; }
    Color color;
    };

Plugin *basic_envmap::ReadString( const std::string &params ) 
    {
    Color c;
    ParamReader get( params );
    if( get["envmap"] && get[MyName()] && get[c] )
        return new basic_envmap( c );
    return NULL;
    }

REGISTER_PLUGIN( basic_envmap );

} // namespace __basic_envmap__



