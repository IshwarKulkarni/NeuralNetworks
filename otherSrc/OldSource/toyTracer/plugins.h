/***************************************************************************
* plugins.h                                                                *
*                                                                          *
* This file defines the classes and functions that allow plug-ins to be    *
* created, registered, accessed, and destroyed.                            *
*                                                                          *
* History:                                                                 *
*   04/26/2010  Added Preprocess plugin.                                   *
*   04/15/2010  Added Container plugin.                                    *
*   04/23/2003  Split off from reader.                                     *
*                                                                          *
***************************************************************************/
#ifndef __PLUGINS_INCLUDED__
#define __PLUGINS_INCLUDED__

#include <iostream>
#include <string>

// Here are all the plug-in types that are currently supported.
enum plugin_type {
    null_plugin,        // A dummy plugin, possibly used for side-effects.
    primitive_plugin,   // Primitive objects, like sphere, block, cone, etc.
    aggregate_plugin,   // Aggregate objects, like list, BVH, Octree, etc.
    shader_plugin,      // A surface shader; one associated with each object.
    envmap_plugin,      // Environment maps defining what surrounds the scene.
    rasterizer_plugin,  // Raserizers, which create rasters by tracing rays.
    builder_plugin,     // Scene builder, reads and/or constructs the scene.
    writer_plugin,      // Writes a Raster to a file.
    container_plugin,   // A container for objects.
    preprocess_plugin   // Process after builder but before rasterizer.
    };

// This is the base class of all plugins.  Each object, shader, rasterizer, etc. must
// be a subclass of this class, and they must fill in these virtual functions:
// ReadString, MyName, and PluginType, and possibly Default if the plugin is to
// serve as a default for the entire class of objects.
struct Plugin {  
    Plugin() {}
    virtual ~Plugin() {}
    virtual Plugin *ReadString( const std::string & ) { return NULL; }  // Instance from string.
    virtual std::string MyName() const { return "undefined"; }
    virtual plugin_type PluginType() const { return null_plugin; }
    virtual bool Default() const { return false; } // Use default if no other of this type is defined.
    static Plugin *Null;  // A dummy plugin that does nothing.
    };

// RegisterPlugin is called by the REGISTER_PLUGIN macro.  This function should never
// be called except via the macro.
extern bool RegisterPlugin(
    Plugin *plugin
    );

// PrintRegisteredPlugins writes a list of all plugins that have been reistered +
// their type to the specified output stream.  This is useful at startup, to verify 
// that all the desired plugins are present.
extern void PrintRegisteredPlugins(
    std::ostream &out
    );

// Return a pointer to the first plugin in the list of registered plugins
// that is of the specified type.  If "after" is set to a plugin pointer,
// this function will return a pointer to the first plugin of the specified
// type that occurs *after* that one in the list of registered plugins.  This
// allows all plugins of a given type to be accessed.
extern Plugin *LookupPlugin(
    plugin_type type,
    const Plugin *after = NULL
    );

// Invoke the readers of all the plugins to see if any of them recognize
// this string.  If so, create an instance of the plugin that recognizes
// the string as its parameters, and return a pointer to the newly-created
// instance as the function value.  This is typically how objects added to
// a scene are created.
extern Plugin *Instance_of_Plugin(
    const std::string &
    );

// DestroyRegisteredPlugins calls the destructor for each plugin.
extern void DestroyRegisteredPlugins(
    );

// The following macro is used for registering new plugins in the toytracer.
// By virtue of this macro, plugins can be added to the toytracer without modifying
// any of the existing code.  Simply define a new plugin (e.g. object, aggregate,
// shader, etc.) in a separate source file and invoke this macro using the class name
// of the new plugin.  Linking the new module into the toytracer will automatically
// add the new plugin into the list of "registered" plugins at run time.  (This macro
// uses the fact that global variables are initialized before "main" is invoked.)

#define REGISTER_PLUGIN( class_name ) \
    static bool dummy_variable_##class_name = RegisterPlugin( new class_name );

#endif


