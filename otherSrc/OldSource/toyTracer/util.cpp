/***************************************************************************
* util.cpp                                                                 *
*                                                                          *
* Miscellaneous utilities, such as predicates on materials & objects.      *
*                                                                          *
* History:                                                                 *
*   04/23/2010  Added output operators for Material and Object.            *
*   04/16/2010  Added functions for generating permutations.               *
*   10/16/2005  Added ToString function for plugin_type.                   *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <cmath>
#include <string>
#include <iostream>
#include "toytracer.h"
#include "constants.h"
#include "util.h"

using std::string;
using std::endl;
using std::ostream;

bool operator==( const Material &a, const Material &b )
    {
    return (
        a.diffuse      == b.diffuse      &&
        a.specular     == b.specular     &&
        a.emission     == b.emission     &&
        a.ambient      == b.ambient      &&
        a.reflectivity == b.reflectivity &&
        a.translucency == b.translucency &&
        a.Phong_exp    == b.Phong_exp    &&
        a.ref_index    == b.ref_index    &&
        a.type         == b.type
        );
    }

// Return a string version of a plugin type.
string ToString( plugin_type ptype )
    {
    switch( ptype )
        { 
        case primitive_plugin  : return "primitive";
        case aggregate_plugin  : return "aggregate";
        case shader_plugin     : return "shader";
        case envmap_plugin     : return "environment map";
        case rasterizer_plugin : return "rasterizer";
        case builder_plugin    : return "builder";
        case writer_plugin     : return "writer";
        case container_plugin  : return "container";
        case preprocess_plugin : return "preprocess";
        default                : break;
        }
    return "unknown";
    }

// Return a string version of a ray type.
string ToString( ray_type rtype )
    {
    switch( rtype )
        { 
        case undefined_ray : return "undefined";
        case generic_ray   : return "generic";
        case visibility_ray: return "visibility";
        case indirect_ray  : return "indirect";
        case light_ray     : return "light";
        case refracted_ray : return "refracted";
        case special_ray   : return "special";
        default            : break;
        }
    return "unknown";
    }

// Return a string containing a rubout character for 
// each (base ten) digit of the integer passed in.
const string &rubout( int n )
    {
    static string buff;
    buff = "\b";
    if( n < 0 ) { buff.append( "\b" ); n = -n; }
    while( n >= 10 ) { buff.append( "\b" ); n /= 10; }
    return buff;
    }

// Return a string containing a rubout character for 
// each character of the string passed in.
const string &rubout( const string &str )
    {
    static string buff;
    buff = str;
    for( unsigned i = 0; i < buff.size(); i++ ) buff[i] = '\b';
    return buff;
    }

double rand( double a, double b )
    {
    double x = float(rand()) / RAND_MAX;
    if( x < 0.0 ) x = -x;
    return a + x * ( b - a );
    }

// Return the smallest divisor of the integer n.  If n is prime, return 0.
unsigned long smallest_divisor( unsigned long n )
    {
    // Handle even numbers first.
    if( (n & 1) == 0 ) return 2;
    // Now check all odd divisors from 3 up to the square root of x.
    unsigned long limit = 2 + unsigned long( floor( sqrt( double(n) ) ) );
    for( unsigned long div = 3; div <= limit; div += 2 )
        {
        if( n % div == 0 ) return div;
        }
    return 0;
    }

// Fill the integer array with a random permutaion of the
// integers 0, 1, ... n-1.
void random_permutation( int n, int perm[], unsigned long seed )
    {
    // Optionally reseed the random number generator.
    if( seed != 0 ) srand( seed );

    // Initialize the permutation to the identity.
    for( int i = 0; i < n; i++ ) perm[i] = i;

    // Step through the permutation, swapping successive
    // elements with a random element chosen from the
    // remainder of the array.
    for( int j = 0; j < n - 1; j++ )
        {
        int k = int( floor( rand( j + 1, n ) ) );
        if( k <= j ) k = j + 1;
        if( k >= n ) k = n - 1;
        // Swap elements j and k.
        int tmp = perm[j];
        perm[j] = perm[k];
        perm[k] = tmp;
        }
    }

// Fill the integer array with an arbitrary permutaion of the
// integers 0, 1, ... n-1.  This is a purely deterministic version.
void non_random_permutation( int n, int perm[], unsigned long seed )
    {
    static const unsigned long prime[] = {
        99761231, 7219841, 1126513, 518709589, 10208743, 7772393, 8756771
        };
    static const int plen( sizeof( prime ) / sizeof( unsigned long ) );

    // Initialize the permutation to the identity.
    for( int i = 0; i < n; i++ ) perm[i] = i;
    if( seed == -1 ) return;

    // Step through the permutation, swapping successive
    // elements with an element chosen from the remainder of the
    // array using deterministic (but chaotic) mod operations.
    int p = seed % plen;
    int step = 3 + n / 11;
    int offset = seed;
    for( int j = 0; j < n - 2; j++ )
        {
        offset += step;
        if( offset > n ) offset = offset % 61;
        if( ++p > plen ) p = 0;
        int k = j + 1 + int( ( prime[p] + offset ) % ( n - j - 2 ) );
        if( k <= j ) k = j + 1;
        if( k >= n ) k = n - 1;
        // Swap elements j and k.
        int tmp = perm[j];
        perm[j] = perm[k];
        perm[k] = tmp;
        }
    }

// Print information about a material.
extern ostream &operator<<( ostream &out, const Material &m )
    {
    out << "\nMaterial " << long(&m)
        << "\n  diffuse     : " << m.diffuse
        << "\n  specular    : " << m.specular
        << "\n  emission    : " << m.emission
        << "\n  reflectivity: " << m.reflectivity
        << "\n  translucency: " << m.translucency
        << "\n  Phong exp   : " << m.Phong_exp
        << "\n  ref index   : " << m.ref_index
        << endl;
    return out;
    }

// Print information about an object.
ostream &operator<<( ostream &out, const Object &obj )
    {
    out << "\nObject " << obj.MyName() 
        << "\n  Shader   = " << long(obj.shader)
        << "\n  Envmap   = " << long(obj.envmap)
        << "\n  Material = " << long(obj.material)
        << endl;
    if( obj.material != NULL ) out << *(obj.material) << endl;
    return out;
    }

// Copy the pointer fields from object to another.
void CopyAttributes( Object *to, const Object *from )
    {
    to->shader   = from->shader;
    to->envmap   = from->envmap;
    to->material = from->material;
    }


// Count the number of objects (either just primitives, or both primitives
// and aggregates) contained in a scene graph.
int CountObjects( const Object *root, bool just_primitives )
    {
    if( root == NULL ) return 0;
    if( root->PluginType() != aggregate_plugin ) return 1;
    Aggregate *agg = (Aggregate *)root;
    agg->Begin();
    int count = ( just_primitives ? 0 : 1 );
    for(;;)
        {
        Object *obj = agg->GetChild();
        if( obj == NULL ) break;
        count += CountObjects( obj, just_primitives );
        }
    return count;
    }


inline Vec3 getRefractedDirection(Vec3 V,Vec3 N,double n){
	double c1,c2,d;
	c1 = - N*V;
	d = 1 - n*n*(1-c1*c1);
	if(d<0)
		return Vec3(0,0,0);

	c2 = sqrt(d);
	return Unit( (n * V) + (n * c1 - c2) * N);
}

