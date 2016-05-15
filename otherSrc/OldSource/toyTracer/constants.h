/***************************************************************************
* constants.h                                                              *
*                                                                          *
* This file defines a number of useful mathematical constants (e.g. Pi),   *
* default values (e.g. resolution), error codes, and constants that are    *
* dependent upon the host machine (e.g. largest representable number and   *
* machine epsilon).                                                        *
*                                                                          *                                                                         *
* History:                                                                 *
*   12/11/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#ifndef __CONSTANTS_INCLUDED__    // Include this file only once.
#define __CONSTANTS_INCLUDED__

#include <limits>

// Miscellaneous numerical constants.

static const double 
    Pi        = 3.14159265358979,
    TwoPi     = 2.0 * Pi,
    FourPi    = 4.0 * Pi,
    DegToRad  = Pi / 180.0,  // Convert degrees to radians.
    Infinity  = std::numeric_limits<double>::max(),
    MachEps_d = std::numeric_limits<double>::epsilon(),
    MachEps_f = std::numeric_limits<float>::epsilon(),
	MachEps     = std::numeric_limits<double>::epsilon(),
	OneMinusEps = MachEps-1,
	OnePlusEps = MachEps+1;
    ;
    
// Miscellaneous default values.

static const int
    default_image_width  = 512,  // Default image width (x resolution).
    default_image_height = 512,  // Default image height (y resolution).
    default_max_tree_depth = 4   // Default cap on ray tree depth.
    ;

enum toytracer_error {
    no_errors = 0,
    error_opening_input_file,
    error_reading_input_file,
    error_opening_image_file,
    error_no_builder,
    error_building_scene,
    error_preprocessing,
    error_no_rasterizer,
    error_rasterizing,
    error_no_writer,
    error_writing
    };

#endif

