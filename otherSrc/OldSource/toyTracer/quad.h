/***************************************************************************
* quad.h                                                                   *
*                                                                          *
* As a plugin, the "quad" primitive object is accessed almost              *
* exclusively through the plugin mechanism, which requires no globally     *
* accessible types or functions.  However, quadss are such a fundamental   *
* modeling primitive that other objects may wish to create them directly   *
* (e.g. for surface tessellations).  This header file defines several      *
* convenient functions by which quad objects can be created directly,      *
* rather than through the ReadString method of the Plugin base class.      *
*                                                                          *
* History:                                                                 *
*   04/16/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __QUAD_INCLUDED__
#define __QUAD_INCLUDED__

#include "toytracer.h"
#include "vec3.h"

extern Object *MakeQuad(
    const Vec3 P[]  // Four vertices.
    );

extern Object *MakeQuad(
    const Vec3 P[], // Four vertices.
    const Vec3 N[]  // Four associated vertex normals.
    );
extern Object *MakeQuad(
    const Vec3 A, Vec3 B,Vec3 C,Vec3 D  // Four vertices.
    
    );
//
//extern Object *MakeQuad(
//    const Vec3 A1, const Vec3 A2,const Vec3 A3,const Vec3 A4,  // Four vertices.
//	const Vec3 N1, const Vec3 N2,const Vec3 N3,const Vec3 N4 , // Four vertices.
//    
//    );
#endif
