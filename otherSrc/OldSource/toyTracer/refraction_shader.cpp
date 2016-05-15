/***************************************************************************
* refrraction_shader                                                       *
* shader that handle refractions          
*
* History:
* 29th may 2010 coded                      
*                                                                          *
***************************************************************************/
#include <string>
#include <cmath>
#include "toytracer.h"
#include "params.h"
#include "util.h"

using std::string;
namespace _refraction_shader_{


//some macros that show how lazy i am
#define _ray (hit.ray)
#define obj (hit.object)
#define mat (hit.object->material)
#define black (Color::Black)
#define white (Color::White)
#define gen (_ray.generation) 


struct refraction_shader : public Shader {
    refraction_shader(unsigned a, unsigned b,unsigned c,bool h) {
		if(a > 20)
			a = 20;
		numDirectRays = a;
		numIndirectRays = b;
		max_depth = c;
		hemisphere = h;
	}
	~refraction_shader() {}
	refraction_shader() {}
    virtual Color Shade( const Scene &, const HitInfo & ) const;
    virtual Plugin *ReadString( const string &params );
    virtual string MyName() const { return "refraction_shader"; }
	unsigned numDirectRays;
	unsigned numIndirectRays;
	unsigned max_depth;
	bool hemisphere;
};

REGISTER_PLUGIN( refraction_shader );


Plugin *refraction_shader::ReadString( const std::string &params ) {
   	//bezier tesselation 8 8 triangles with-normals
	
//	refraction_shader *result = NULL;
    ParamReader get( params );
	unsigned a,b,c;
	bool h = false;
	std::string s;
	if(get["shader"] && get["refraction_shader"] && get[a]&&get[b] ){
		if(get["depth"] && get[c])
			return new refraction_shader(a,b,c,h);
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

Color refraction_shader::Shade( const Scene &scene, const HitInfo &hit ) const{    
	  
	if(gen >= max_depth)
		return scene.envmap->Shade(hit.ray);

	if( mat->emission != black )
		if (_ray.type == indirect_ray) //turnign off the light
			return black;
		else
			return mat->emission; //this si direct sample, so return illum of the light
	
	if(mat->reflectivity != black && _ray.type == indirect_ray)
		return black;
		
	Vec3   O = _ray.origin;
	Vec3   P = hit.point;
	Vec3   N = hit.normal;
	Vec3   E = Unit( O - P );
	Vec3   R = Unit( ( 2.0 * ( E * N ) ) * N - E );
	if( E * N < 0.0 ) N = -N; //flip the normal to make it point into the space of the ray origin

	if(mat->translucency != black ){
		Ray rr = obj->getRefracetedRay(_ray,P,N);
		if(rr.direction!=Vec3(0,0,0)) 
			return scene.Trace(rr)*mat->translucency 
			+ scene.Trace(obj->getReflectedRay(_ray,P,N))*(white-mat->translucency);
		else return black;//this is TIR even after 4 level of internal reflections
	}
	
	unsigned k = 1,n = 1;
		
	if(gen == 1 ){ // rays of tree depth = 1
		n = numDirectRays;
		k = numIndirectRays;
	}

	Color color = mat->diffuse * mat->ambient;

	Sample *dSamples = new Sample[n*n];
	unsigned numSamples;
	HitInfo dHit;		
	Ray dRay,iRay;		
	unsigned Y = scene.NumLights() ;

	for( unsigned i = 0; i < Y; i++ ){
		const Object *light = scene.GetLight(i);
		numSamples = light->GetSamples (hit.point, hit.normal, dSamples, n);
			//#pragma omp parallel for 
		for(unsigned j=0;j<numSamples ;j++){
			dRay.origin = hit.point + Unit(N)*10E-4;
			dRay.direction = Unit( dSamples[j].P - hit.point );
			dRay.generation = gen + 1; //next level in tree
			dHit.ray = dRay;
			dHit.distance = Infinity;
			dRay.type = _ray.type;
			if(scene.Cast(dRay,dHit))
				if( dHit.object == light){
					color += max( 0, dRay.direction*Unit(N) )*
							mat->diffuse*
							light->material->emission*dSamples[j].w/Pi;
					if( mat->Phong_exp !=0 && mat->specular != black) 
							//this si handling specularity
						color += (white-mat->reflectivity)* 
							pow( max( dRay.direction*R,0), mat->Phong_exp) * 
							mat->specular*light->material->emission*dSamples[j].w;
					}
				else if(dHit.object->material->translucency != black){
					//hit a tranclusive surface, so cast the transmitted ray
					Ray rr = dHit.object->getRefracetedRay(dRay,dHit.point, dHit.normal);
					HitInfo rrHit;
					rrHit.distance = Infinity;
					rrHit.ray = rr;
					if(scene.Cast(rr,rrHit))
						if(rrHit.object == light)
							color += max( 0, dRay.direction*Unit(N) )*
							mat->diffuse*
							light->material->emission*dSamples[j].w/Pi;
					if( mat->Phong_exp !=0 && mat->specular != black)
						color += (white-mat->reflectivity)*
							pow( max( dRay.direction*R,0), mat->Phong_exp) * 
							mat->specular*light->material->emission*dSamples[j].w;
					
					}
				}
		 
	}	
	
	if(mat->reflectivity == black){
		//sample hmisphere (stratified)
		if(max_depth > 2){
			Sample*	iSamples = new Sample[k*k];
			numSamples = SampleProjectedHemisphere (hit.point, hit.normal, iSamples, k);
			//#pragma omp parallel for
			for(unsigned j = 0; j < numSamples ;j++){
				iRay.origin = hit.point + Unit(N)*10E-4; 
				iRay.from = obj;
				iRay.direction = Unit( hit.point - iSamples[j].P );
				iRay.generation = gen+ 1;
				iRay.type = indirect_ray;
				color += (mat->diffuse*iSamples[j].w*scene.Trace(iRay))/TwoPi;
			}
			delete [] iSamples;
			iSamples = NULL;
		}
	}
	else{
		
		Ray rr;
		rr.origin = P+ N*10E-4;
		rr.direction = R;
		rr.from = obj;
		rr.ignore = NULL;
		rr.generation = gen+1;
		color = color*(white - mat->reflectivity);
		color += mat->reflectivity*scene.Trace(rr);
	}

	

	delete [] dSamples;
	dSamples = NULL;
	return color; 
	

}


}