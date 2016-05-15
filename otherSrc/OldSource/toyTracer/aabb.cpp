/***************************************************************************
* aabb.cpp   (Axis-Aligned Bounding Box)                                   *
*                                                                          *
* The AABB structure encodes a three-dimensional axis-aligned box as a     *
* collection of three intervals, one for each axis.  These boxes are ideal *
* as bounding boxes as they are equipped with a very efficient ray-box     *
* intersector.  There are also a number of useful methods & functions that *
* expand the box, transform it, compute its surface area, etc.             *
*                                                                          *
* History:                                                                 *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include "aabb.h"
#include "util.h"

// This operator allows a 3x3 matrix to transform a bounding box.
// The result is another axis-aligned box guaranteed to contain all
// of the the original vertices transformed by the given matrix.
AABB operator*( const Mat3x3 &M, const AABB &box )
    {
    double new_min[] = { 0, 0, 0 };
    double new_max[] = { 0, 0, 0 };
    double old_min[] = { box.X.min, box.Y.min, box.Z.min };
    double old_max[] = { box.X.max, box.Y.max, box.Z.max }; 

    // Find the extreme points by considering the product
    // of the min and max with each component of M.
                     
    for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
        {
        const double a( M(i,j) * old_min[j] );
        const double b( M(i,j) * old_max[j] );
        new_min[i] += min( a, b ); 
        new_max[i] += max( a, b );
        }

    // Return the smallest bounding box enclosing the extremal points.
    return AABB(
        Interval( new_min[0], new_max[0] ),
        Interval( new_min[1], new_max[1] ),
        Interval( new_min[2], new_max[2] )
        );
    }

// Determine whether the given ray intersects the box.  If there is an intersection,
// but it's farther than "max_dist", then regard it as a miss.
bool AABB::Hit( const Ray &ray, double max_dist ) const
    {
    const Vec3 &R( ray.direction );
    const Vec3 &Q( ray.origin );
    double r, s, t;
    double min( 0.0 );
    double max( max_dist );

    if( R.x > 0 ) // Looking in positive X direction.
        {
        if( Q.x > X.max ) return false;
        r = 1.0 / R.x;
        s = ( X.min - Q.x ) * r; if( s > min ) min = s;
        t = ( X.max - Q.x ) * r; if( t < max ) max = t;
        }
    else if( R.x < 0.0 ) // Looking in negative X direction.
        {
        if( Q.x < X.min ) return false;
        r = 1.0 / R.x;
        s = ( X.max - Q.x ) * r; if( s > min ) min = s;
        t = ( X.min - Q.x ) * r; if( t < max ) max = t;
        }
    else if( Q.x < X.min || Q.x > X.max ) return false;

    if( min > max ) return false; // Degenerate interval.

    if( R.y > 0 ) // Looking in positive Y direction.
        {
        if( Q.y > Y.max ) return false;
        r = 1.0 / R.y;
        s = ( Y.min - Q.y ) * r; if( s > min ) min = s;
        t = ( Y.max - Q.y ) * r; if( t < max ) max = t;
        }
    else if( R.y < 0 ) // Looking in negative Y direction.
        {
        if( Q.y < Y.min ) return false;
        r = 1.0 / R.y;
        s = ( Y.max - Q.y ) * r; if( s > min ) min = s;
        t = ( Y.min - Q.y ) * r; if( t < max ) max = t;
        }
    else if( Q.y < Y.min || Q.y > Y.max ) return false;

    if( min > max ) return false; // Degenerate interval.

    if( R.z > 0 ) // Looking in positive Z direction.
        {
        if( Q.z > Z.max ) return false;
        r = 1.0 / R.z;
        s = ( Z.min - Q.z ) * r; if( s > min ) min = s;
        t = ( Z.max - Q.z ) * r; if( t < max ) max = t;
        }
    else if( R.z < 0 ) // Looking in negative Z direction.
        {
        if( Q.z < Z.min ) return false;
        r = 1.0 / R.z;
        s = ( Z.max - Q.z ) * r; if( s > min ) min = s;
        t = ( Z.min - Q.z ) * r; if( t < max ) max = t;
        }
    else if( Q.z < Z.min || Q.z > Z.max ) return false;

    // There is a hit if and only if the intersection interval [min,max]
    // is not degenerate.
    return min <= max;
    }

bool AABB::Contains( const Vec3 &p ) const
    {
    return X.Contains( p.x ) && Y.Contains( p.y ) && Z.Contains( p.z );
    }

double SurfaceArea( const AABB &A )
    {
    if( IsNull( A ) ) return 0.0;
    const double x( A.X.Length() );
    const double y( A.Y.Length() );
    const double z( A.Z.Length() );
    return 2.0 * ( x * ( y + z ) + y * z );
    }

double Volume( const AABB &A )
    {
    return A.X.Length() * A.Y.Length() * A.Z.Length();
    }

// Expand this box to enclose box "b" if it does not already do so.
// Return true iff the box actually had to be expanded.
bool AABB::Expand( const AABB &b )
    {
    bool expand( false );
    if( b.X.min < X.min ) { expand = true; X.min = b.X.min; } 
    if( b.X.max > X.max ) { expand = true; X.max = b.X.max; } 
    if( b.Y.min < Y.min ) { expand = true; Y.min = b.Y.min; } 
    if( b.Y.max > Y.max ) { expand = true; Y.max = b.Y.max; } 
    if( b.Z.min < Z.min ) { expand = true; Z.min = b.Z.min; } 
    if( b.Z.max > Z.max ) { expand = true; Z.max = b.Z.max; } 
    return expand;
    }

// Applies an affine transformation (encoded as a 3x4 matrix) to a
// 3D box and returns the tightest-fitting axis-aligned enclosing
// the result.
AABB operator*( const Mat3x4 &M, const AABB &box ) 
    {
    AABB new_box( M.mat * box );
    return new_box += M.vec;
    }


