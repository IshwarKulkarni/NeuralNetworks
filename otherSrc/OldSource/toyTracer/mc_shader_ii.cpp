/***************************************************************************
* mc_shader.cpp   (shader plugin)                                       *
*                                                                          *
* This file defines a very simple ray tracing shader for the toy tracer.   *
* The job of the shader is to determine the color of the surface, viewed   *
* from the origin of the ray that hit the surface, taking into account the *
* surface material, light sources, and other objects in the scene.         *                          *
*                                                                          *
* History:                                                                 *
*   10/03/2005  Updated for Fall 2005 class.                               *
*   09/29/2004  Updated for Fall 2004 class.                               *
*   04/14/2003  Point lights are now point objects with emission.          *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <string>
#include <cmath>
#include "toytracer.h"
#include "params.h"
#include "util.h"

using std::string;

struct mc_shader_ii : public Shader {
    mc_shader_ii(unsigned a, unsigned b,unsigned c,bool h) {
		if(a > 20)
			a = 20;
		numDirectRays = a;
		numIndirectRays = b;
		max_depth = c;
		hemisphere = h;
	}
	~mc_shader_ii() {}
	mc_shader_ii() {}
    virtual Color Shade( const Scene &, const HitInfo & ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "mc_shader_ii"; }
	unsigned numDirectRays;
	unsigned numIndirectRays;
	unsigned max_depth;
	bool hemisphere;
};

REGISTER_PLUGIN( mc_shader_ii );


Plugin *mc_shader_ii::ReadString( const std::string &params ) {
   	//bezier tesselation 8 8 triangles with-normals
	
//	mc_shader_ii *result = NULL;
    ParamReader get( params );
	unsigned a,b,c;
	bool h = false;
	std::string s;
	if(get["shader"] && get["mc_shader_ii"] && get[a]&&get[b] ){
		if(get["depth"] && get[c])
			return new mc_shader_ii(a,b,c,h);
		else{
			std::cout<<"parsing error in line "<< params;
			return NULL;			
		}
	}
	else 
		return NULL;
}

inline unsigned SampleProjectedHemisphere( const Vec3 &P, const Vec3 &N, Sample *samples, unsigned n ) {
   
	
	double weight = 2*Pi/(double)(n*n);
	Vec3 refNormal = Unit(N-Vec3(0,0,1));

	unsigned k = 0;
	double r1,r2,radSquare;

	for( unsigned i=0;i<n;i++){
		for( unsigned j =0 ;j<n ;j++){
			r1 = (i + rand(0,1))/n;
			r2 = (j + rand(0,1))/n;

			samples[k].P.x = sqrt(r1)*cos(2*Pi*r2);
			samples[k].P.y = sqrt(r1)*sin(2*Pi*r2);
			
			radSquare = samples[k].P.x*samples[k].P.x + samples[k].P.y*samples[k].P.y ;
			
			samples[k].P.z = sqrt(1-radSquare);

			samples[k].P = samples[k].P  - ( 2.0 * ( samples[k].P  * refNormal ) ) * refNormal;
			samples[k].P = samples[k].P + P;
			samples[k++].w = weight;
		}	
	}
	return n*n;
}

Color mc_shader_ii::Shade(const Scene & scene, const HitInfo &hit) const
{    
	if(hit.object->material->reflectivity != Color::Black && hit.ray.type == indirect_ray)
			return Color::Black;

	if(hit.ray.generation >= max_depth)
			return scene.envmap->Shade(hit.ray);

	if( hit.object->material->emission != Color::Black )
		if (hit.ray.type == indirect_ray)
			return Color::Black;
		else
			return hit.object->material->emission; 				


	Vec3   O = hit.ray.origin;
	Vec3   P = hit.point;
	Vec3   N = hit.normal;
	Vec3   E = Unit( O - P );
	Vec3   R = Unit( ( 2.0 * ( E * N ) ) * N - E );

	if( E * N < 0.0 ) N = -N; 

	unsigned k=1,n=1;
		
	if(hit.ray.generation == 1){
		n = numDirectRays;
		k = numIndirectRays;
	}

	Color color = hit.object->material->diffuse * hit.object->material->ambient;
		 

	Sample *dSamples = new Sample[n*n];
	unsigned numSamples;
	HitInfo dHit;		
	Ray dRay,iRay;		

	for( unsigned i = 0; i < scene.NumLights() ; i++ )
	{
		const Object *light = scene.GetLight(i);
		numSamples = light->GetSamples (hit.point, hit.normal, dSamples, n);
		#pragma omp parallel for
		for(unsigned j=0;j<numSamples ;j++){
			dRay.origin = hit.point + Unit(N)*10E-4;
			dRay.direction = Unit( dSamples[j].P - hit.point );
			dRay.generation = hit.ray.generation + 1;
			dHit.ray = dRay;
			dHit.distance = Infinity;
			dRay.type = hit.ray.type;
			if(scene.Cast(dRay,dHit))
				if( dHit.object == light)
					color += max( 0, dRay.direction*Unit(N) )*
							hit.object->material->diffuse*
							light->material->emission*dSamples[j].w/Pi;
		}
	} 
	
	if(hit.object->material->reflectivity == Color::Black){
		
		if(max_depth > 2){
			Sample*	iSamples = new Sample[k*k];
			numSamples = SampleProjectedHemisphere (hit.point, hit.normal, iSamples, k);
			#pragma omp parallel for
			for(unsigned j = 0; j < numSamples ;j++){
				iRay.origin = hit.point + Unit(N)*10E-4; 
				iRay.from = hit.object;
				iRay.direction = Unit( hit.point - iSamples[j].P );
				iRay.generation = hit.ray.generation + 1;
				iRay.type = indirect_ray;
				color += (hit.object->material->diffuse*iSamples[j].w*scene.Trace(iRay))/TwoPi;
			}
			delete [] iSamples;
			iSamples = NULL;
		}
	}
	else{
		
		Ray rr;
		rr.origin = P+ N*10E-4;
		rr.direction = R;
		rr.from = hit.object;
		rr.ignore = NULL;
		rr.generation = hit.ray.generation;
		color = color*(Color::White - hit.object->material->reflectivity);
		color += hit.object->material->reflectivity*scene.Trace(rr);
	}

	delete [] dSamples;
	dSamples = NULL;
	return color; 
}


