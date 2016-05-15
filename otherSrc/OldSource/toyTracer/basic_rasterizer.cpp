/***************************************************************************
* basic_rasterizer.cpp    (rasterizer plugin)                              *
*                                                                          *
* A "rasterizer" is responsible for tracing all the primary rays needed to *
* create an image.  It saves the resulting colors to a "raster", which is  *
* simply a two dimensional array of colors, which it allocates and returns *
* as the function value.  The raster can then be used to create the actual *
* an image, generally after some form of tone mapping.  This "basic"       *
* rasterizer casts a single ray per pixel; it does no anti-aliasing.       *                                                           *
*                                                                          *
* A rasterizer can also be used to cast specific rays for testing purposes *
* without generating a true raster. In such a case, it should still        *
* allocate and return a raster, but one with zero dimensions.              *
*                                                                          *
* History:                                                                 *
*   05/01/2010  Added "pixel" parameters for testing & debugging.          *
*   10/03/2005  Made rasterizer a plugin.  Line numbers written in place.  *
*   12/19/2004  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include <vector>
#include "toytracer.h"
#include "color.h"
#include "params.h"
#include "util.h"
#include "omp.h"

namespace __basic_rasterizer__ {

using std::string;
using std::vector;
using std::cout;
using std::endl;

struct basic_rasterizer : Rasterizer {
    basic_rasterizer() {}
    virtual ~basic_rasterizer() {}
    virtual Raster *Rasterize( const Camera &, const Scene & );
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "basic_rasterizer"; }
    virtual bool Default() const { return true; }
    vector<Vec3> test_pixels;
    };

REGISTER_PLUGIN( basic_rasterizer );

// This raserizer requires no parameters unless it is being used to gather
// statistics at specific pixels.  In the latter case, we need to know which
// pixels and how many times it should be sampled.  Look for the keyword
// "pixel" followed by triples of numbers, which will be interpreted as
// the 2D pixel coordinates plus the number of samples.
Plugin *basic_rasterizer::ReadString( const string &params ) 
    {
    ParamReader get( params );
    if( get["rasterizer"] && get[MyName()] )
        {
        // See if there are specific pixels listed after the name of
        // the rasterizer.  If so, collect these in a list and trace them
        // rather than creating a raster.  This is used for debugging
        // and testing.
        Vec3 v;
        test_pixels.clear();
        while( get["pixel"] && get[v] ) test_pixels.push_back(v);
        // Copy the parameters we just set into the new instance.
        return new basic_rasterizer( *this );
        }
    return NULL;
    }

// Perform some simple statistical tests on the values returned by tracing
// a given ray repeatedly.  The vector "test" encodes the pixel coordinates
// as well as the number of times to trace the same ray.  Print out what is being
// tested followed by the sample mean and variance.
static void TestPixel( const Scene &scene, const Ray &ray, const Vec3 &test )
    {
    const int n( int( floor( max( 1, test.z ) ) ) );
    // Print out which pixel we're tracing, and how many times.
    cout << "\nTracing pixel (" << test.x << "," << test.y << ") "
         << n << ( n > 1 ? " times" : " time" )
         << endl;
    // Trace the same ray the requested number of times.  This is useful for
    // Monte Carlo estimators, as it allows us to estimate the variance.
    Color sample_sum;
    Color sample_sqr;
    for( int i = 0; i < n; i++ )
        {
        const Color sample( scene.Trace( ray ) );
        // Print out the individual samples if there aren't too many of them.
        if( n <= 100 ) cout << "  " << sample << endl;
        sample_sum += sample;
        sample_sqr += sample * sample;
        }
    // Finally, print the sample mean and variance if we have more than one sample. 
    if( n > 1 )
        {
        cout << "  Mean: "   
             << ( sample_sum / n ) << ", "
             << "Variance: "
             << fabs( (sample_sqr/n) - (sample_sum * sample_sum)/(n*n) )
             << endl;
        }
    }

// Rasterize casts all the initial rays starting from the eye.  This trivial version
// casts one ray through the center of the pixel and collects the results in a Raster.
// A pointer to the newly-allocated and filled-in Raster is returned as the function value.
Raster *basic_rasterizer::Rasterize(const Camera &cam, const Scene &scene)
{
	const double xmin(cam.x_win.min);
	const double ymax(cam.y_win.max);
	const double width(cam.x_win.Length());
	const double height(cam.y_win.Length());

	// Compute increments etc. based on the camera geometry.  These will be used
	// to define the ray direction at each pixel.

	const Vec3 G(Unit(cam.lookat - cam.eye));          // Gaze direction.
	const Vec3 U(Unit(cam.up / G));                    // Up vector.
	const Vec3 R(Unit(G ^ U));                         // Right vector.
	const Vec3 Or(cam.vpdist * G + xmin * R + ymax * U);  // "Origin" of the 3D raster.
	const Vec3 dR(width  * R / cam.x_res);                // Right increments.
	const Vec3 dU(height * U / cam.y_res);                // Up increments.

	Raster *raster(new Raster(cam.x_res, cam.y_res));

	unsigned row, col;
	int percentDone = 0;
	float LineDone = 0;
	int k = 0;
	cout << "Percentage done:   0";
	cout.flush();
	
	for (row = 0; row < cam.y_res; row++)
	{
		// Overwrite the line number written to the console.
		Ray ray;
		ray.origin = cam.eye;     // All primary rays originate from the eye.
		ray.type = generic_ray; // These rays have no special meaning.
		ray.generation = 1;           // Primary rays are first-generation.
		k = (int)(100 * (LineDone / (double)cam.y_res));
		cout << rubout((int)(100 * (LineDone / (double)cam.y_res))) << ((int)(100 * (LineDone / (double)cam.y_res)));
		cout.flush();

		for (col = 0; col < cam.x_res; col++)
		{
			ray.direction = Unit(Or + (col + 0.5) * dR - (row + 0.5) * dU);
			raster->pixel(row, col) = scene.Trace(ray);
		}
		LineDone++;
	}
	cout << endl;
	return raster;
}


} // namespace __basic_rasterizer__
