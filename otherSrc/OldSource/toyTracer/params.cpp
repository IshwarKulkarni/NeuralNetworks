/***************************************************************************
* params.cpp                                                               *
*                                                                          *
* A tool for reading parameter strings from sdf files.  Each method        *
* looks for some specific pattern at the beginning of a string and does    *
* three things if it is found:                                             *
*    1) The pattern is removed from the string.                            *
*    2) The pattern is parsed and assigned to the argument, if appropriate.*
*    3) True is returned as the function value.                            *
* If the pattern is not found, the string is left intact and false is      *
* returned as the function value.                                          *
*                                                                          *
* History:                                                                 *
*   05/01/2010  Added support for indexed vertices.                        *
*   04/15/2010  Added a method for fetching quoted strings.                *
*   10/04/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include <map>
#include "params.h"

using std::string;
using std::cerr;
using std::endl;

static const char DoubleQuote( '\"' );

// This table is used for associating indices with 3D coordinates.  This
// allows us to use vertex lists for triangle vertices or control points.
static std::map<int,Vec3> vec3_table;

void ParamReader::DefineIndex( int i, const Vec3 &v )
    {
    // Insert a new Vec3 into the table, and associate it with index "i".
    vec3_table[i] = v;
    }

bool ParamReader::LookupIndex( int i, Vec3 &v )
    {
    if( vec3_table.find( i ) == vec3_table.end() ) return false;
    v = vec3_table[ i ];
    return true;
    }

// Remove leading white space from the parameter string.
void ParamReader::SkipBlanks()
    {
    for( unsigned n = 0; n < params.size(); n++ )
        {
        const char c( params[n] );
        if( c != ' ' && c != '\t' )
            {
            // Remove all characters before this.
            if( n > 0 ) params.erase( 0, n );
            return;
            }
        }
    // There are no non-white characters.
    params.clear();
    }

// Check for the presence of the given string.  This method does NOT
// retrieve the next string.  This is used to check for expected keywords,
// not for fetching string parameters.
bool ParamReader::operator[]( string keyword )
    {
    SkipBlanks();
    if( params.find( keyword, 0 ) == 0 )
        {
        params.erase( 0, keyword.length() );
        return true;
        }
    return false;
    }

// See if there are any more non-white characters left.
bool ParamReader::isEmpty()
    {
    SkipBlanks();
    return params.empty();
    }

// Fetch the next string of characters that is either surrounded by double
// quotes or is delimited by white space.  This method is distinguished from
// the previous one by passing it a *pointer* to a string rather than a string
// reference.
bool ParamReader::operator[]( string *field )
    {
    SkipBlanks();
    if( params.empty() )
        {
        return false;
        }
    else if( params[0] == DoubleQuote )  // Handle double-quoted strings.
        {
        // Remove the opening double quote.
        params.erase( 0, 1 );
        // Locate the closing quote.  (Should check for failure here.)
        int k = params.find( DoubleQuote );
        // Copy the quoted string into "field", without the quotes.
        field->insert( 0, params, 0, k );
        // Erase string and closing quote.
        params.erase( 0, k + 1 );  
        return true;
        }

    // Handle unquoted strings of chars, which will either be terminated
    // by white space or the end of the buffer.
    unsigned k = 0;
    while( k < params.size() )
        {
        if( params[k] == ' ' || params[k] == '\t' ) break;
        k++;
        }
    // Copy the quoted string into "field", without the quotes.
    field->insert( 0, params, 0, k );
    // Erase string that was just copied to "field".
    params.erase( 0, k );  
    return true;
    }

// Read a color if possible: [r,g,b]
bool ParamReader::operator[]( Color &c )
    {
    if( sscanf( params.c_str(), " [ %f , %f , %f ]", &c.red, &c.green, &c.blue ) == 3 )
        {
        int len = 1 + params.find( "]", 0 );
        params.erase( 0, len );
        return true;
        }
    return false;
    }

// Read a Vec3 if possible: (x,y,z)
bool ParamReader::operator[]( Vec3 &v )
    {
    // If indexing is enabled, look for an integer instead of an explicit three-tuple.
    unsigned i;
    if( indexing && operator[](i) )
        {
        if( !LookupIndex( i, v ) )
            {
            Warning( "Indexed point not found" );
            v.Zero();
            }
        return true;
        }
    // Look for an explicit three-tuple of numbers.
    if( sscanf( params.c_str(), " ( %lf , %lf , %lf )", &v.x, &v.y, &v.z ) == 3 )
        {
        int len = 1 + params.find( ")", 0 );
        params.erase( 0, len );
        return true;
        }
    return false;
    }

// Read a Vec2 if possible: (x,y)
bool ParamReader::operator[]( Vec2 &v )
    {
    if( sscanf( params.c_str(), " ( %lf , %lf )", &v.x, &v.y ) == 2 )
        {
        int len = 1 + params.find( ")", 0 );
        params.erase( 0, len );
        return true;
        }
    return false;
    }

// Read an Interval if possible: (a,b)
bool ParamReader::operator[]( Interval &I )
    {
    if( sscanf( params.c_str(), " ( %lf , %lf )", &I.min, &I.max ) == 2 )
        {
        int len = 1 + params.find( ")", 0 );
        params.erase( 0, len );
        return true;
        }
    return false;
    }

// Read a single floating point number if possible.
bool ParamReader::operator[]( double &x )
    {
    SkipBlanks();
    if( sscanf( params.c_str(), "%lf", &x ) == 1 )
        {
        int len = 1 + params.find( " ", 0 );
        if( len > 1 ) params.erase( 0, len ); else params.clear();
        return true;
        }
    return false;
    }

// Read an unsigned integer if possible.
bool ParamReader::operator[]( unsigned &x )
    {
    SkipBlanks();
    if( sscanf( params.c_str(), "%u", &x ) == 1 )
        {
        int len = 1 + params.find( " ", 0 );
        if( len > 1 ) params.erase( 0, len ); else params.clear();
        return true;
        }
    return false;
    }

// Read a 3x3 matrix if possible: ( a, b, c;  d, e, f;  g, h, i )
bool ParamReader::operator[]( Mat3x3 &M )
    {
    // Attempt to read a 3x3 matrix, in row order, enclosed in parens.
    // The rows of the matrix are to be separated by semicolons, which
    // is the same convention used by Matlab.
    int count = sscanf(
        params.c_str(),
        " ( %lf , %lf , %lf ; %lf , %lf , %lf ; %lf , %lf , %lf ) ",
        &M(0,0), &M(0,1), &M(0,2), 
        &M(1,0), &M(1,1), &M(1,2),
        &M(2,0), &M(2,1), &M(2,2)
        );
    // If the read was completely successful, erase the matrix from the string.
    if( count == 9 )
        {
        int len = 1 + params.find( ")", 0 );
        params.erase( 0, len );
        return true;
        }
    return false;
    }


// Read a 3x4 matrix if possible: ( a, b, c, d;  e, f, g, h;  i, j, k l )
bool ParamReader::operator[]( Mat3x4 &M )
    {
    Mat3x3 A;
    // Attempt to read a 3x4 matrix, in row order, enclosed in parens.
    // The rows of the matrix are to be separated by semicolons, which
    // is the same convention used by Matlab.
    int count = sscanf(
        params.c_str(),
        " ( %lf , %lf , %lf , %lf ; %lf , %lf , %lf , %lf ; %lf , %lf , %lf , %lf ) ",
        &M.mat(0,0), &M.mat(0,1), &M.mat(0,2), &M.vec.x, 
        &M.mat(1,0), &M.mat(1,1), &M.mat(1,2), &M.vec.y,
        &M.mat(2,0), &M.mat(2,1), &M.mat(2,2), &M.vec.z
        );
    // If the read was completely successful, erase the matrix from the string.
    if( count == 12 )
        {
        int len = 1 + params.find( ")", 0 );
        params.erase( 0, len );
        return true;
        }
    else if( operator[](A) )
        {
        // We successfully read a 3x3 matrix, so promote it to a 3x4.
        M.mat = A;
        M.vec.Zero();
        return true;
        }
    return false;
    }

void ParamReader::Error( string msg ) const
    {
    cerr << "\nError: " << msg << ": " << params << endl;
    }

void ParamReader::Warning( string msg ) const
    {
    cerr << "\nWarning: " << msg << ": " << params << endl;
    }

