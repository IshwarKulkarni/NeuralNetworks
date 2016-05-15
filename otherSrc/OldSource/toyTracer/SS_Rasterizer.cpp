#include <string>
#include "toytracer.h"
#include "color.h"
#include "params.h"
#include "util.h"
using namespace std;
struct SS_Rasterizer : Rasterizer {
    SS_Rasterizer() {
	
		type = "average";
		level = 1;
	
	}
	SS_Rasterizer(std::string s, unsigned a) {

		type = s;
		if( a < 1)
			a++;
		if( a > 17 && s != "random")	//289 samples per pixel is large enough
			a = 17;
		level = a;

	}
    virtual ~SS_Rasterizer() {}
    virtual Raster *Rasterize( const Camera &, const Scene & );
    virtual Plugin *ReadString( const std::string &params );
    virtual std::string MyName() const { return "SS_Rasterizer"; }
    virtual bool Default() const { return false; }
	std::string type;
	unsigned int level;
    };

REGISTER_PLUGIN( SS_Rasterizer );

Plugin *SS_Rasterizer::ReadString( const std::string &params ) 
    {
    ParamReader p( params );
	unsigned a;
    if( p["rasterizer"] && p[MyName()] )
		if( p["average"] && p[a] )
			return new SS_Rasterizer("average",a);
		else if( p["gaussian"] && p[a])
			return new SS_Rasterizer("gaussian",a);
		else if( p["random"] && p[a])
			return new SS_Rasterizer("random",a);
		else
			return new SS_Rasterizer();
    return NULL;
    }

double** getKernel(std::string type, unsigned level){

	double** kernel = NULL;
	double sigmaSquared = 6.25;	
			//only one  sigma ditribution

	double center = (double)(level/2);
	kernel = new double*[level];
	
	for(unsigned int x = 0; x < level; ++x ){
		kernel[x] = new double[level];
		for( unsigned int y = 0; y < level; ++y ){
			if(type == "gaussian"){
				double power = ( (double)x - center ) * ((double)x - center ) + ((double)y - center ) * ( (double)y - center );
				double baseEexponential = pow( 2.718281828459, -power / ( 2.0 * sigmaSquared ) );
				kernel[x][y] = baseEexponential / (2.0*Pi*sigmaSquared );
				
			}
			else if(type == "average"){
				kernel[x][y] = 1.0/(level*level);
				
			}
		}
	}
	if(type == "gaussian"){ // normalizing here
		double s = 0.0;
		for( unsigned x = 0; x < level; ++x )
			for( unsigned y = 0; y < level; ++y )
				s += kernel[x][y];


		for( unsigned x = 0; x < level; ++x )
			for( unsigned y = 0; y < level; ++y )
				 kernel[x][y] /= s;
	}
			
	return kernel;

}


Raster *SS_Rasterizer::Rasterize( const Camera &cam, const Scene &scene ){
    

    Raster *raster = new Raster( cam.x_res, cam.y_res );

    Ray ray;
    ray.origin     = cam.eye;     // All initial rays originate from the eye.
    ray.type       = generic_ray; // These rays are given no special meaning.
    ray.generation = 1;           // Rays cast from the eye are first-generation.

    const double xmin   = cam.x_win.min;
    const double ymax   = cam.y_win.max;
	const double width  = ( cam.x_win.Length() );
	const double height = ( cam.y_win.Length() );

    const Vec3 G ( Unit( cam.lookat - cam.eye ) );          // Gaze direction.
    const Vec3 U ( Unit( cam.up / G ) );                    // Up vector.
    const Vec3 R ( Unit( G ^ U ) );                         // Right vector.
    const Vec3 Or( cam.vpdist * G + xmin * R + ymax * U );  // "Origin" of the 3D raster.
    const Vec3 dR( width  * R / cam.x_res );                // Right increments.
    const Vec3 dU( height * U / cam.y_res );                // Up increments.

	Color c , black = Color::Black;

	double** kernel;

	if (!(type == "random" || (type == "average" && level == 1))){

		if (type == "gaussian" && level % 2 == 0)
			level++;

		kernel = getKernel(type, level);
	}
	else
		return raster;

    int row;
    unsigned col;
	int percentDone = 0;
	int LineDone = 0;

	std::cout << "Super Sampling Rasterizer. Type: "<<type<<", Level: "<<level<< ", line 0";
	#pragma omp parallel for private(row,col,c) shared(scene,raster)
    for(   row = 0; row < cam.y_res; row++ ){
		
		cout<<"\b\b\b\b\b\b\b\b\b\b\b\b\b";
		cout.flush();
		cout<<(int)((float)LineDone/(double)(cam.y_res))<<"% Done";
		cout.flush();
		
        for(   col = 0 ; col < cam.x_res; col++ )
            {
				c = black;
				if(level ==1 && type == "average"){
					ray.direction = Unit( Or + (col + 0.5) * dR - (row + 0.5) * dU  );
					c = scene.Trace( ray );
				}
				else{
					if( type == "random"){
						for(unsigned k1 = 0;k1<level;k1++){
							ray.direction = Unit( Or + ((double)col + rand(0,1)) * dR - ((double)row + rand(0,1)) * dU  );
							c += scene.Trace( ray );
						}
						c = c/(double)level;
					}
					else{
						for(unsigned k1 = 0; k1<level;k1++){
							for( unsigned k2 = 0;k2 < level; k2++){
								ray.direction = Unit(Or +
								(col + (k1*(1.0/(double)(level+1)))) * dR -
								(row + (k2*(1.0/(double)(level+1)))) * dU  );
								c += scene.Trace( ray )* kernel[k1][k2];
							}
						}
					}

				}
				raster->pixel( row, col ) = c;

			}
			LineDone++;
        }
    std::cout << "... done." << std::endl;

    // Return a pointer to the floating-point raster as the function value. 

    return raster;
    }
