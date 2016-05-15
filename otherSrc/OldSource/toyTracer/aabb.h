/***************************************************************************
* aabb.h    (Axis-Aligned Bounding Box)                                    *
*                                                                          *
* AABB is a simple structure that defines a three-dimensional box.         *
* This header defines numerous convenient operations on these 3D boxes     *
* such as those that translate or expand the box.                          *
*                                                                          *
* History:                                                                 *
*   04/17/2010  Made Hit a member function.                                *
*   12/10/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __AABB_INCLUDED__
#define __AABB_INCLUDED__

#include <string>
#include <iostream>
#include "constants.h"
#include "interval.h"
#include "mat3x3.h"
#include "mat3x4.h"
#include "ray.h"

// Axis-Aligned box in R3.  This is very useful as a bounding volume.
struct AABB {
    inline AABB() {}
    inline AABB( const Interval &A, const Interval &B, const Interval &C ) : X(A), Y(B), Z(C) {}
    inline ~AABB() {}
    inline Vec3 MinCorner() const { return Vec3( X.min, Y.min, Z.min ); }
    inline Vec3 MaxCorner() const { return Vec3( X.max, Y.max, Z.max ); }
    bool   Hit( const Ray &r, double max_dist = Infinity ) const;
    bool   Contains( const Vec3 &P ) const;
    bool   Expand( const AABB & );
    static inline AABB Null();
    Interval X;
    Interval Y;
    Interval Z;
    };

// Returns an AABB that is maximally degenerate in all three axes.
inline AABB AABB::Null()
    {
    return AABB( Interval::Null(), Interval::Null(), Interval::Null() );
    }

// An AABB is degenerate (i.e. contains nothing) iff one of its
// intervals is degenerate.
inline bool IsNull( const AABB &A )
    { 
    return IsNull( A.X ) || IsNull( A.Y ) || IsNull( A.Z );
    }

// The centroid is the average of any two opposing corners.
inline Vec3 Center( const AABB &A )
    {
    return 0.5 * ( A.MinCorner() + A.MaxCorner() );
    }

// Expand box A to enclose box B.
inline AABB & operator<<( AABB &A, const AABB &B ) 
    {
    A.X << B.X;
    A.Y << B.Y;
    A.Z << B.Z;
    return A;
    };

// Determine whether two boxes overlap, which happens iff
// the intervals along all three axes overlap.
inline bool operator&( const AABB &A, const AABB &B ) 
    {    
    return ( A.X & B.X ) && ( A.Y & B.Y ) && ( A.Z & B.Z );
    };

// Expand box A to enclose the point p.
inline AABB & operator<<( AABB &A, const Vec3 &p ) 
    {
    A.X << p.x;
    A.Y << p.y;
    A.Z << p.z;
    return A;
    }

// Translate the box A by p.
inline AABB & operator+=( AABB &A, const Vec3 &p ) 
    {
    A.X += p.x;
    A.Y += p.y;
    A.Z += p.z;
    return A;
    }

// Translate the box A by -p.
inline AABB & operator-=( AABB &A, const Vec3 &p ) 
    {
    A.X -= p.x;
    A.Y -= p.y;
    A.Z -= p.z;
    return A;
    }

inline std::ostream &operator<<( std::ostream &out, const AABB &box )
    {
    out << "(" << box.X << "," << box.Y << "," << box.Z << ")";
    return out;
    }

extern double Volume(
    const AABB &
    );

extern double SurfaceArea(
    const AABB &
    );

// Transforms a bounding box by a 3x3 matrix.
extern AABB operator*(  
    const Mat3x3 &M,
    const AABB &box
    ); 

// Transforms a bounding box by a 3x4 matrix.
extern AABB operator*(  
    const Mat3x4 &M,
    const AABB &box
    );

#endif


