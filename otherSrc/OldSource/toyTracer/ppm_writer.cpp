/***************************************************************************
* ppm_writer.cpp    (writer plugin that generates PPM images directly)     *
*                                                                          *
* This is a trivial writer plugin that simply maps the range [0,1] encoded *
* in floating point to the range [0,255] encoded as a byte.  The sequence  *
* of bytes is then written to a PPM image file, which is the most          *
* primitive image file format there is.                                    *
*                                                                          *
* If this module is compiled and linked into the toytracer, it will replace*
* the default writer, which produces an FPM image file.  To disable this   *
* ppm_writer (and revert to the default) you can simply comment out the    *
* REGISTER_PLUGIN macro.                                                   *
*                                                                          *
* History:                                                                 *
*   04/13/2010  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include "toytracer.h"
#include "params.h"
#include "utils/exceptions.hxx"

using std::string;
using std::ofstream;

namespace __ppm_writer__ {

struct ppm_writer : Writer {  
    ppm_writer() {}
    virtual ~ppm_writer() {}
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "ppm_writer"; }
    virtual string Write( string file_name, const Raster * );
    };

// Comment out the next line to prevent this writer from overriding the default writer.
// Then the toytracer will generate an FPM image by default (unless there is another
// writer plugin that is registered).
REGISTER_PLUGIN( ppm_writer );

Plugin *ppm_writer::ReadString( const string &params ) 
    {
    ParamReader get( params );
    if( get["writer"] && get[MyName()] ) return new ppm_writer();
    return NULL;
    }

static inline char clamp( double x )
    {
    if( x > 1.0 ) x = 1.0;
    if( x < 0.0 ) x = 0.0;
    return char( floor( 255.99 * x ));
    }

// Convert the raw floating point numbers to bytes, then write them to a file
// after a simple header encoding the file type, the resolution, and number of bits
// per channel.
string ppm_writer::Write( string file_name, const Raster *raster )
    {
    const unsigned width  = raster->width;
    const unsigned height = raster->height;
    string full_name = file_name + ".ppm";

    // Don't attempt to write anything if the raster is empty.
    if( width == 0 || height == 0 ) return "";

    // Open the file for writing in binary.  Return an empty
    // string on failure.
    ofstream fout( full_name.c_str(), ofstream::binary );
	THROW_IF(!fout, FileIOException, "Could not create file %s", full_name.c_str());

    // Write the header information, which consists of five fields,
    // each terminated by a newline character.
    fout << "P6"    << "\n" 
         << width   << " "   // width and height are separated by a space.
         << height  << "\n"
         << 255     << "\n";

    int  num_bytes = 3 * width * height;
    char *ppm = new char[ num_bytes ];

    Color *p = raster->pixels;
    char  *q = ppm;
    for( unsigned i = 0; i < width * height; i++ )
        {
        Color c = *p++;
        *q++ = clamp( c.red   );
        *q++ = clamp( c.green );
        *q++ = clamp( c.blue  );
        }

    // Write the actual data, which consists of triples of bytes.
    fout.write( ppm, num_bytes );
    fout.close();

    // Return the full file name to indicate success.
    return full_name;
    }

} // namespace __ppm_writer__