/***************************************************************************
* params.h                                                                 *
*                                                                          *
* A tool for reading parameter strings from sdf files.  A "ParamReader"    *
* is a wrapper for a string read from a file that allows successive tokens *
* to be removed from the string.  Each "[]" operator looks for a specific  *
* type of token, then parses it and removes it from the string if it is    *
* found.                                                                   *
*                                                                          *
* History:                                                                 *
*   04/30/2010  Added support for indexed vertices and normals.            *
*   04/15/2010  Added a method for fetching quoted strings.                *
*   10/04/2005  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __PARAMS_INCLUDED__
#define __PARAMS_INCLUDED__

#include <string>
#include "color.h"
#include "vec2.h"
#include "vec3.h"
#include "interval.h"
#include "mat3x3.h"
#include "mat3x4.h"

// The ParamReader class makes the ReadString method of each Plugin very
// easy to write.  Given the parameters as a string, each "[]" operator
// attempts to strip off the given item, and returns "true" if successful.

class ParamReader {
    public:
        inline ParamReader( std::string s ) : params(s), indexing(false) {}
        bool operator[]( std::string  keyword );  // Check for a given string.
        bool operator[]( std::string *field   );  // Fetch the next string.
        bool operator[]( Color    & );
        bool operator[]( Vec3     & );
        bool operator[]( Vec2     & );
        bool operator[]( double   & );
        bool operator[]( unsigned & );
        bool operator[]( Interval & );
        bool operator[]( Mat3x3   & );
        bool operator[]( Mat3x4   & );
        bool isEmpty();
        const std::string &Remainder() const { return params; }
        void Error  ( std::string ) const;
        void Warning( std::string ) const;
        void EnableIndexing( bool on_off = true ) { indexing = on_off; }
        static void DefineIndex( int i, const Vec3 & );
        static bool LookupIndex( int i, Vec3 & );
    private:
        void SkipBlanks();
        std::string params;
        bool indexing;
	};

#endif

