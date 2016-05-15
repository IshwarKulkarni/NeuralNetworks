/****************************	***********************************************
* main.h                                                                   *
*                                                                          *
* This is the main program for the "toytracer".  It reads in the camera    *
* and scene from a file, then invokes the rasterizer, which casts all the  *
* initial rays and writes the resulting image to a file.                   *
*                                                                          *
* History:                                                                 *
*   10/18/2010  Gather and print basic statistics.                         *
*   10/04/2005  Updated for 2005 graphics class.                           *
*   10/10/2004  Print registered objects, get optional file name from argv.*
*   04/03/2003  Main program now defines scene geometry.                   *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include "toytracer.h"
#include "constants.h"
#include<time.h>
#include<string>
#include<omp.h>
#include "util.h"
#include <fstream>
#include "quad.h"
#include "utils\Logging.hxx"
#include "utils\exceptions.hxx"

using namespace std;
using namespace Logging;

int main(int argc, char *argv[]){

	Log.Tee(cout);

	Log << "**** ToyTracer Version " 
		<< TOYTRACER_MAJOR_VERSION << "."
		<< TOYTRACER_MINOR_VERSION
		<< " ****" << LogEndl << LogFlush();

	PrintRegisteredPlugins(Log);

	string	fileName = string(TOYTRACER_SCENES) + (argc > 1 ? argv[1] : "CornellBox-and-balls");	
	Builder *builder = (Builder *)LookupPlugin(builder_plugin);
	Scene  scene;  Camera camera;
		
	bool SceneBuilt  = builder->BuildScene(fileName, camera, scene);
	THROW_IF(!SceneBuilt, InvalidOptionException, "Scene building failed");
	
	Log << "\n Scene built from " << fileName
		<< "\n  Number of primitives: " << Primitive::num_primitives
		<< "\n  Number of aggregates: " << Aggregate::num_aggregates
		<< "\n  Number of containers: " << Container::num_containers << "\n\n" << LogFlush();

	Timer rasterizer;
	Raster *raster = scene.rasterize->Rasterize(camera, scene);
	double rasterTime = rasterizer.Stop();

	
	string imageFilename = scene.writer->Write((argc > 2 ? argv[2] : fileName), raster);
	
	Log << "\n Image written to " << imageFilename << "\n"
		<< "\n  Total Number of rays cast    : " << Scene::NumRaysCast()
		<< "\n  Total Number of intersection : " << Object::NumIntersectionTests()
		<< "\n  Average intersections per ray: " << Object::NumIntersectionTests() / Scene::NumRaysCast()
		<< "\n  Total Time taken in rendering: " << rasterTime << "s.\n"
		<< "\nDone!" << LogEndl << LogFlush();

	
	delete raster;
	DestroyRegisteredPlugins();

	cin.get();
	return no_errors;
	
}

