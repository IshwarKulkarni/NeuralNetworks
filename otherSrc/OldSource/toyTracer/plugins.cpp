/***************************************************************************
* plugins.cpp                                                              *
*                                                                          *
* This file defines the functions that allow plugins to be registered,     *
* accessed, and deleted.  These plugins are based on a very simple idea    *
* that makes use of static linking rather than dynamic loading.  Plugins   *
* are "registered" using a macro that simply adds an instance of the       *
* plugin object to a global list that is created when global variables are *
* initialized (which is before "main" is called).                          *
*                                                                          *
* History:                                                                 *
*   04/23/2003  Split off from reader.                                     *
*                                                                          *
***************************************************************************/
#include <iostream>
#include <string>
#include <list>
#include "toytracer.h"
#include "util.h"
#include "params.h"

// Create a pointer to a plugin that does nothing.  This is useful for
// "ReadString" methods that simply set parameters or cause other side-effects
// rather than creating new objects.  In such a case, the reader can pass back
// a pointer to this dummy plugin to signal that the read was successful.
Plugin *Plugin::Null = new Plugin; 

// Maintain a list of all registered plug-ins.  Access to this list is restricted
// to the functions in this module.
static std::list< Plugin* > *all_plugins = NULL;  

// RegisterPlugin is called by the REGISTER_PLUGIN macro.  It adds an instance
// of each registered object to a global list.  The list can then be accessed
// by the functions PrintRegisteredPlugins, Instance_of_Plugin, LookupPlugin,
// and DestroyRegisteredPlugins.
bool RegisterPlugin( Plugin *plg ) 
    {
    if( all_plugins == NULL ) all_plugins = new std::list< Plugin* >; 
    all_plugins->push_back( plg );
    return true;
    }

// PrintRegisteredPlugins writes a list of all plugins that have been
// registered to the outstream "out" (which will probably be cout).
void PrintRegisteredPlugins( std::ostream &out )
    {
    out << "\nRegistered Plugins:\n";
    int count = 0;
    std::list< Plugin* >::const_iterator iter;
    if( all_plugins != NULL )
    for( iter = all_plugins->begin(); iter != all_plugins->end(); iter++ )
        {
        ++count;
        const Plugin *plg = *iter;
        const std::string type( ToString( plg->PluginType() ) );
        if( count < 10 ) out << " "; // Keep numbers right-justified.
        out << "  " << count
            << ": " << type;
        for( int i = type.length(); i < 20; i++ ) out << ".";
        out << plg->MyName();
        // Indicate whether this plug-in is designated as a default or not.
        if( plg->Default() ) out << " (default)";
        out << std::endl;
        }
    out << std::endl;
    }

// Call the readers of all the plugins to see if any of them recognize
// this string.  If so, create an instance of the plugin that recognizes
// the string as its parameters, and return a pointer to the newly-created
// instance as the function value.
Plugin *Instance_of_Plugin( const std::string &str )
    {
    if( all_plugins == NULL ) return NULL;
    Plugin *plg = NULL;
    std::list< Plugin* >::iterator iter;
    for( iter = all_plugins->begin(); iter != all_plugins->end(); iter++ )
        {
        plg = (*iter)->ReadString( str );
        if( plg != NULL ) break;
        }
    return plg;
    }

// Return a pointer to the first plugin in the list of registered plugins
// that is of the specified type.  If "after" is set to a plugin pointer,
// this function will return a pointer to the plugin of the specified type
// that occurs *after* that one in the list of registered plugins.  This
// allows all plugins of a given type to be accessed.
Plugin *LookupPlugin( plugin_type type, const Plugin *after )
    {
    if( all_plugins == NULL ) return NULL;
    Plugin *pin = NULL;
    bool use_next_match = ( after == NULL );
    std::list< Plugin* >::iterator iter;
    for( iter = all_plugins->begin(); iter != all_plugins->end(); iter++ )
        {
        if( (*iter)->PluginType() == type )
            {
            if( use_next_match )
                {
                pin = *iter;
                // If the plugin is NOT maked as the default, use it.
                // Otherwise, keep looking for another match, and use
                // the default only if we find nothing else.
                if( !pin->Default() ) return pin;
                }
            if( after == *iter ) use_next_match = true;
            }
        }
    // Return either NULL, if no plugin was found, or the default
    // plugin of this type.  Default plugins are only returned if
    // there are no other plugins of the specified type.
    return pin;
    }

// Delete all the instances of the plugins that were created at start-up
// by the REGISTER_PLUGIN macro and were used exclusively for plugin lookups
// and parsing the scene description.
void DestroyRegisteredPlugins( )
    {
    if( all_plugins == NULL ) return;
    std::list< Plugin* >::const_iterator iter;
    for( iter = all_plugins->begin(); iter != all_plugins->end(); iter++ )
        {
        Plugin *plg = *iter;
        delete plg;
        }
    all_plugins->clear();
    }



