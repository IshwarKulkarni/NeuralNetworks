/***************************************************************************
* sphere.h                                                                 *
*                                                                          *
* As a plugin, the "sphere" primitive object is accessed almost            *
* exclusively through the plugin mechanism, which requires no globally     *
* accessible types or functions.  However, as other objects may wish to    *
* create spheres directly, this header file defines a convenient function  *
* by which to do so that does not rely upon the ReadString method of the   *
* Plugin base class.                                                       *
*                                                                          *
* History:                                                                 *
*   05/09/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __SPHERE_INCLUDED__
#define __SPHERE_INCLUDED__

#include "toytracer.h"
#include "vec3.h"

// Create a sphere object of the given center and radius, which default
// to (0,0,0) and 1.0, respectively, if not specified.

extern Object *MakeSphere(
    const Vec3 &center = Vec3(0,0,0),
    double rad = 1.0
    );

#endif
