/***************************************************************************
* structure.cpp   (container plugin)                                       *
*                                                                          *
* The "structure" plugin is a subclass of Container that is designed to    *
* allow groups of primitves to be created and instanced.  This can         *
* simplify model building when there are repeated instances of the same    *
* geometric object, such as desks in a room.                               *
*                                                                          *
* History:                                                                 *
*   10/17/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <iostream>
#include <string>
#include <map>
#include "toytracer.h"
#include "params.h"
#include "mat3x4.h"

namespace __structure_container__ {

struct structure : Container {
    structure() { index = 0; }
    virtual ~structure() {}
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "structure"; }
    static std::map< std::string, structure* > name_table;
    };

REGISTER_PLUGIN( structure );

std::map< std::string, structure* > structure::name_table;

Plugin *structure::ReadString( const std::string &params )
    {
    std::string name;
    ParamReader get( params );
    if( get["define"] && get[MyName()] && get[&name] )
        {
        // Create a new structure and save its name in the name table.
        structure *s = new structure();
        name_table[ name ] = s;
        return s;
        }
    if( get[MyName()] && get[&name] )
        {
        // Check that the name is already in the table.
        if( name_table.find( name ) == name_table.end() )
            {
            std::cerr << "Warning: undefined structure " << name << std::endl;
            return Plugin::Null;
            }
        // Return a pointer to the named structure.  This is used
        // for instancing previously-defined structures.
        return name_table[ name ];
        }
    return NULL;
    }

} // namespace __structure_container__





