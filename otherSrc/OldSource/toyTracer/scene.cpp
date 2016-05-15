/***************************************************************************
* scene.cpp                                                                *
*                                                                          *
* The "scene" structure defines two of the most fundamental functions of   *
* the ray tracer: "Cast" and "Trace".  Both are rather trivial, as the     *
* real computations are encapsulated in the actual objects comprising      *
* the sceen.                                                               *
*                                                                          *
* History:                                                                 *
*   05/01/2010  Added "Visible" method to Scene class.  Rewrote "Trace".   *
*   04/15/2010  Initialize scene with default plug-ins.                    *
*   09/29/2005  Updated for 2005 graphics class.                           *
*   10/16/2004  Check for "ignored" object in "Cast".                      *
*   09/29/2004  Updated for Fall 2004 class.                               *
*   04/14/2003  Point lights are now point objects with emission.          *
*   04/01/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include "toytracer.h"
#include "constants.h"

// Allocate & initialize all the static counters used for statistics.

double Scene::num_rays_cast = 0;
double Object::num_intersection_tests = 0;
double Object::num_visibility_tests = 0;
double Primitive::num_primitives = 0;
double Aggregate::num_aggregates = 0;
double Container::num_containers = 0;

static const Color
default_background_color = Color(0.15, 0.25, 0.35),
default_termination_color = Color(0.15, 0.15, 0.15);


// Initialize the environment map, the shader, the rasterizer, and the writer
// to something reasonable (i.e. to the defaults or to the custom versions)
// at the outset, in case the builder doesn't set them.

void Scene::init()
{
	object = NULL;
	envmap = (Envmap     *)LookupPlugin(envmap_plugin);
	shader = (Shader     *)LookupPlugin(shader_plugin);
	rasterize = (Rasterizer *)LookupPlugin(rasterizer_plugin);
	writer = (Writer     *)LookupPlugin(writer_plugin);
	num_rays_cast = 0;
	max_tree_depth = default_max_tree_depth;
}

// Cast finds the first point of intersection (if there is one)
// between a ray and a list of geometric objects.  If no intersection
// exists, the function returns false.  Information about the
// closest object hit is returned in "hitinfo".  Note that the single
// object associated with the scene will typically be an aggregate (so
// that the scene can contain many primitive objects).

bool Scene::Cast(const Ray &ray, HitInfo &hitinfo) const
{
	num_rays_cast++;
	if (!object->Hit(ray, hitinfo)) return false;

	hitinfo.ray = ray;

	return true;
}

// Trace is the most fundamental of all the ray tracing functions.  It
// answers the query "What color do I see looking along the given ray
// in the current scene?"  This is an inherently recursive process, as
// trace may again be called as a result of the ray hitting a reflecting
// or transparent object.  To prevent the possibility of infinite recursion,
// a maximum depth is placed on the resulting ray tree.

Color Scene::Trace(const Ray &ray) const
{
	Color   color;   // The color to return.
	HitInfo hitinfo;

	// If the recursion has bottomed out, we must do something drastic.
	if (ray.generation > max_tree_depth) return default_termination_color;
	if ( (ray.type == sparse_rast_ray && rasterize->GetHitInfo(ray,hitinfo)) || Cast(ray, hitinfo))
	{
		if (hitinfo.object->shader != NULL)
		{
			// Use the shader associated with this object.
			color = hitinfo.object->shader->Shade(*this, hitinfo);
		}
		else if (shader != NULL)
		{
			// Use the global shader associated with the scene.
			color = shader->Shade(*this, hitinfo);
		}
		else if (hitinfo.object->material != NULL)
		{
			// Just use the diffuse color.
			color = hitinfo.object->material->diffuse;
		}
	}
	else if (ray.from != NULL && ray.from->envmap != NULL)
	{
		// Use the environment map associated with the object.
		color = ray.from->envmap->Shade(hitinfo.ray);
	}
	else if (envmap != NULL)
	{
		// Use the global environment map associated with the scene.
		color = envmap->Shade(hitinfo.ray);
	}
	else
	{
		// Just use the background color.
		color = default_background_color;
	}

	return color;
}

// Determine whether points P and Q are mutually visible.  If "epsilon" is specified,
// shrink the line PQ by epsilon at each end.
bool Scene::Visible(const Vec3 &P, const Vec3 &Q, const Object *ignore, double epsilon) const
{
	Ray ray;
	HitInfo hitinfo;
	double dist;
	const Vec3 V(Q - P);
	const Vec3 U(Unit(V, dist));
	ray.type = visibility_ray;
	ray.origin = P + epsilon * U;
	ray.direction = U;
	ray.generation = 1;
	ray.from = NULL;
	ray.ignore = ignore;
	hitinfo.distance = dist - 2.0 * epsilon;
	return !Cast(ray, hitinfo);
}

// Miscellaneous constructors and destructors that were not defined inline
// in the toytracer.h header file.

Material::Material()
{
	diffuse = Color::White;
	specular = Color::White;
	emission = Color::Black;
	ambient = Color::Black;
	reflectivity = Color::Black;
	translucency = Color::Black;
	ref_index = 0.0;
	Phong_exp = 0.0;
}

HitInfo::HitInfo()
{
	object = NULL;
	distance = Infinity;
}

Raster::Raster(int w, int h)
{
	width = w;
	height = h;
	pixels = new Color[w * h];
}

Raster::~Raster()
{
	delete[] pixels;
}

Object::Object()
{
	material = NULL;
	shader = NULL;
	envmap = NULL;
}

const Color
Color::White(1.0f, 1.0f, 1.0f),
Color::Gray(0.5f, 0.5f, 0.5f),
Color::Black(0.0f, 0.0f, 0.0f),
Color::Red(1.0f, 0.0f, 0.0f),
Color::Green(0.0f, 1.0f, 0.0f),
Color::Blue(0.0f, 0.0f, 1.0f),
Color::Yellow(1.0f, 1.0f, 0.0f),
Color::Magenta(1.0f, 0.0f, 1.0f),
Color::Cyan(0.0f, 1.0f, 1.0f);


