/***************************************************************************
* printer.cpp   (preprocess plugin)                                        *
*                                                                          *
* The "printer" plugin is a subclass of Preprocess that simpy prints the   *
* scene information that has been constructed by the builder to standard   *
* out (std::cout).  This is useful for verifying that the scene object was *
* built correctly (e.g. that structures were instanced properly, and       *
* transforms and warps were applied properly).                             *
*                                                                          *
* History:                                                                 *
*   04/27/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <iostream>
#include <string>
#include <map>
#include "toytracer.h"
#include "params.h"

using std::string;
using std::cout;
using std::endl;

namespace __printer_preprocess__ {

struct printer : Preprocess {
    printer( unsigned lim = 0 ) { limit = lim; }
    virtual ~printer() {}
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "printer"; }
    virtual bool Run( string command, Camera &, Scene & );
    void print_it( std::ostream &out, Object *root );
    string indent() const;
    unsigned indent_n;
    static unsigned limit;
    };

// Uncomment the following line if you wish to have the printer object write
// a version of the scene graph to the console after it is built.
// REGISTER_PLUGIN( printer );

unsigned printer::limit = 10;

Plugin *printer::ReadString( const string &params )
    {
    ParamReader get( params );
    if( get["preprocess"] && get[MyName()] )
        {
        // See if the optional limit was specified.
        if( get["limit"] && get[limit] ) {}
        return Plugin::Null;
        }
    return NULL;
    }

string printer::indent() const
    {
    static string spaces( "" );
    while( spaces.size() < indent_n ) spaces.append( " " );
    return spaces.substr( 0, indent_n );
    }

bool printer::Run( string command, Camera &camera, Scene &scene )
    {
    indent_n = 0;
    cout << "\n----- Scene printed by " << MyName() << " plugin ------" << endl;
    print_it( cout, scene.object );
    cout << "\n" << endl;
    return true;
    }

// Recursively print the hierarchy starting from the root.
// If there is only one aggregate object, the hierarchy will
// only be one level deep (i.e. the children of the one
// aggregate).  However, if there are other aggregates nested 
// within the top one, the printer will print each nested
// aggregate with further indentation.
void printer::print_it( std::ostream &out, Object *root )
    {
    if( root == NULL ) return;
    if( root->PluginType() == aggregate_plugin )
        {
        Aggregate *agg = (Aggregate *)root;
        out << indent() << "begin " << root->MyName() << endl;
        indent_n += 4;
        agg->Begin();
        for( int n = 0;  ; n++ )
            {
            Object *obj = agg->GetChild();
            if( obj == NULL ) break;
            if( limit > 0 && n == limit )
                {
                // Indicate that the list has been truncated.
                out << indent() << "..." << endl;
                break;
                }
            else print_it( out, obj );
            }
        indent_n -= 4;
        out << indent() << "end" << endl;
        }
    else if( root->PluginType() == primitive_plugin )
        {
        out << indent() << root->MyName() << endl;
        }
    }


} // namespace __printer_preprocess__





