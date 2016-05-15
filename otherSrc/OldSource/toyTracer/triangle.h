/***************************************************************************
* triangle.h                                                               *
*                                                                          *
* As a plugin, the "triangle" primitive object is accessed almost          *
* exclusively through the plugin mechanism, which requires no globally     *
* accessible types or functions.  However, triangles are such a fundamental*
* modeling primitive that other objects may wish to create them directly   *
* (e.g. for surface tessellations).  This header file defines several      *
* convenient functions by which triangle objects can be created directly,  *
* rather than through the ReadString method of the Plugin base class.      *
*                                                                          *
* History:                                                                 *
*   04/16/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __TRIANGLE_INCLUDED__
#define __TRIANGLE_INCLUDED__

#include "toytracer.h"
#include "vec3.h"

// Make a vanilla triatngle object with no normal vectors. 
//
extern Object *MakeTriangle(
    const Vec3 &A,
    const Vec3 &B,
    const Vec3 &C
    );

// Make a triangle with normals associated with the vertices.
// The vertex normals need not be normalized.

extern Object *MakeTriangle(
    const Vec3 P[],
    const Vec3 N[]
    );

#endif
