/***************************************************************************
* warp.h                                                                   *
*                                                                          *
* The "warp" object is an aggregate object that accepts a single child     *
* object.  It can apply an arbitrary affine transformation to any object   *
* by performing ray transformations.  This header file declares an         *
* external function for creating warp objects directly, rather than via    *
* the ReadString method. This allows other objects such as the "transform" *
* container to create transformed objects using warp aggregates.           *
*                                                                          *
* History:                                                                 *
*   04/21/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __WARP_INCLUDED__
#define __WARP_INCLUDED__

#include "toytracer.h"
#include "mat3x4.h"

extern Object *MakeWarp(
    const Mat3x4 &M,  // The transformation to apply.
    Object *obj       // The object to apply it to.
    );

#endif
