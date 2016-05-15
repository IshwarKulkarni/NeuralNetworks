/***************************************************************************
* basic_shader.cpp   (shader plugin)                                       *
*                                                                          *
* This file defines a very simple ray tracing shader for the toy tracer.   *
* The job of the shader is to determine the color of the surface, viewed   *
* from the origin of the ray that hit the surface, taking into account the *
* surface material, light sources, and other objects in the scene.         *
*                                                                          *
* History:                                                                 *
*   10/03/2005  Updated for Fall 2005 class.                               *
*   09/29/2004  Updated for Fall 2004 class.                               *
*   04/14/2003  Point lights are now point objects with emission.          *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"

using std::string;

struct basic_shader : Shader {
    // Modify as necessary...
    basic_shader() {}
   ~basic_shader() {}
    virtual Color Shade( const Scene &, const HitInfo & ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "basic_shader"; }
    virtual bool Default() const { return true; }
    // Fill in as necessary...
    };

Plugin *basic_shader::ReadString( const string &params ) 
    {
    ParamReader get( params );
    if( get["shader"] && get[MyName()] ) return new basic_shader();
    return NULL;
    }

Color basic_shader::Shade( const Scene &scene, const HitInfo &hit ) const{
		

	if( !( hit.object->material->emission == Color::Black ) )
		return hit.object->material->emission;
 
	Material *mat   = hit.object->material;			
    Color  diffuse  = mat->diffuse;
    Color  specular = mat->specular;
	Color  color = hit.object->material->ambient*diffuse;
	Color ref_color = Color::Black;

	Ray ray;
    HitInfo otherhit;
    static const double epsilon = 1.0E-4;
	  
	
    Vec3   O = hit.ray.origin;
    Vec3   P = hit.point;
    Vec3   N = hit.normal;
    Vec3   E = Unit( O - P );
    Vec3   R = Unit( ( 2.0 * ( E * N ) ) * N - E );  // The reflected ray.
    Color  r = mat->reflectivity;
    double e = mat->Phong_exp;
	if( E * N < 0.0 ) N = -N;		

	for( unsigned i = 0; i < scene.NumLights(); i++ ){
        
		const Object *light = scene.GetLight(i);
        Color emission = light->material->emission;
        Vec3 LightPos( Center( GetBox( *light )) ); 

		ray.direction = Unit(LightPos -P) ;		// shoot a shadow ray to each light
		ray.origin = P + N*epsilon;

		if(scene.Cast(ray,otherhit) && &otherhit.object != &light){
			color += mat->emission;

		}
		else{
			color += (N * Unit(LightPos -P))*diffuse;			
			if( e !=0 )
				color += (Color::White-r)*pow( Unit( LightPos- P) * R, e) * mat->specular;
		}
	}
	
	
	ray.direction = R;
	ray.from = hit.object;
	if(r!=Color::Black)
		color += r*scene.Trace(ray);
	
	if(ref_color != Color::Black)
		return color*(Color::White - hit.object->material->translucency) + ref_color  ;
	else 
		return color;
   
    }


REGISTER_PLUGIN( basic_shader );


