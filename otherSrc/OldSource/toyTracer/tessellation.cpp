/***************************************************************************
* tessellation.cpp                                                         *
*                                                                          *
* The "Tessellation" class serves as a base class for Containers that      *
* generate tessellations of parametric surfaces into triangles or quads,   *
* with or without normals.  Examples: bezier, superquadric, supertoroid.   *
*                                                                          *
* If the tessellated surface has certain basic symmetries, this class can  *
* automatically generate all the reflections.  For example, superquadrics  *
* have 3-way symmetry, so triangles need only be generated for one octant, *
* and the other seven reflections can be filled in automatically.          *
*                                                                          *
* Tessellations require both the "triangle" and the "quad" plug-ins.       *
*                                                                          *
* To do:                                                                   *
*    Eliminate redundant evaluations.                                      *
*    Accommodate quads, or a mixture of quads and triangles.               *
*    Look for degenerate normals and "repair" them using geometric normal. *
*                                                                          *
* History:                                                                 *
*   05/11/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include "toytracer.h"
#include "mat3x4.h"
#include "triangle.h"  // Needed for generating triangle objects directly.
#include "quad.h"      // Needed for generating quad objects directly.
#include "tessellation.h"

Tessellation::Tessellation()
    {
    with_normals = false;
    normals_set  = false;
    symtype      = no_symmetry;
    elemtype     = triangles_only;  // Not yet implemented.
    }

// Tessellate the given portion of the uv plane into triangles or quads and add
// them to the list of objects in this container.  Symmetries will be applied to the
// triangles as needed.
void Tessellation::Tessellate( Interval uint, unsigned nu, Interval vint, unsigned nv )
    {
    Vec3 v0, v1, v2, v3;
    Vec3 n0, n1, n2, n3;
    const double ustep( uint.Length() / nu );
    const double vstep( vint.Length() / nv );

    for( unsigned u = 0; u < nu; u++ )
        {
        const double umin( uint.min + ( u     ) * ustep );
        const double umax( uint.min + ( u + 1 ) * ustep );

        for( unsigned v = 0; v < nv; v++ )
            {
            const double vmin( vint.min + ( v     ) * vstep );
            const double vmax( vint.min + ( v + 1 ) * vstep );

            // This is very inefficient, with many redundant evaluations.
            Eval( umax, vmin, v0, n0 );
            Eval( umin, vmin, v1, n1 );
            Eval( umin, vmax, v2, n2 );
            Eval( umax, vmax, v3, n3 );

            // Partition the quad into two triangles, with or without normals.
            // Construct the two triangles so that they have the same clock-sense.
            // Skip degenerate triangles.  
            if( v0 != v2 && v0 != v1 && v2 != v1 )
                {
                if( with_normals ) SetNormals( n0, n1, n2 );
                AddTriangle( v0, v1, v2 );
                }
            if( v0 != v2 && v0 != v3 && v2 != v3 ) 
                {
                if( with_normals ) SetNormals( n0, n2, n3 );
                AddTriangle( v0, v2, v3 );
                }
            }
        }
    }

void Tessellation::SetNormals( Vec3 &n1, Vec3 &n2, Vec3 &n3 )
    {
    // Save the normals for subsequent calls.
    N1 = n1;
    N2 = n2;
    N3 = n3;
    // Set a flag that will cause the next call to AddTriangle to use these normals.
    normals_set = true;
    }

// Add the given triangle to the list of children as well as other reflections of this
// triangle if one of the symmetry types was specified.  This way we need only generate
// triangles for one octant, quadrant, or half-space of a symmetrical object.  The
// triangle is supplied with normal vectors if this call is preceded by a call to
// SetNormals, and normals are enabled.
void Tessellation::AddTriangle( Vec3 &v1, Vec3 &v2, Vec3 &v3 )
    {
    // The default loop limits are such that no symmetries are created.
    int a_limit(1);
    int b_limit(1);
    int c_limit(1);
    // Reset the loop limits based on the type of symmetry desired.
    switch( symtype )
        {
        case xyz_symmetry: c_limit = -1; // Fall through.
        case xy_symmetry : b_limit = -1; // Fall through.
        case x_symmetry  : a_limit = -1; // Fall through.
        case no_symmetry : break;
        }
    // Create 1, 2, 4, or 8 triangles, depending on the type of symmetry.
    for( int a = 1; a >= a_limit; a -= 2 )
    for( int b = 1; b >= b_limit; b -= 2 )
    for( int c = 1; c >= c_limit; c -= 2 )
        {
        // Create a scaling matrix that reflects through zero or more of
        // the planes x=0, y=0, and z=0.  We can use this same matrix to
        // transform both points and normals since it is orthogonal.
        const Mat3x3 M( Scale( a, b, c ) ); 
        if( with_normals && normals_set )
            {
            // Check the sign of the determinant.
            if( a * b * c < 0 )
                {
                // Change the clock-sense of the triangle if the transfomation
                // has negative determinant.  This will keep the clock-sense
                // of outward-facing triangles consistent, which may be important
                // to subsequent algorithms.
                const Vec3 Q[] = { M * v1, M * v3, M * v2 };
                const Vec3 R[] = { M * N1, M * N3, M * N2 };
                children.push_back( MakeTriangle( Q, R ) );
                }
            else
                {
                const Vec3 Q[] = { M * v1, M * v2, M * v3 };
                const Vec3 R[] = { M * N1, M * N2, M * N3 };
                children.push_back( MakeTriangle( Q, R ) );
                }
            }
        else
            {
            children.push_back( MakeTriangle( v1, v2, v3 ) );
            }
        }
    // The normals must be explicitly set before each call to AddTriangle.
    normals_set = false;
    }
