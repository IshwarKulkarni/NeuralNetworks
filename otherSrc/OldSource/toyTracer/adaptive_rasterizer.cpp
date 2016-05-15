/***************************************************************************
* adaptive_rasterizer.cpp    (rasterizer plugin)                           *
*   03/17/2016  Initial coding based on basic_rasterizer                   *
*                                                                          *
***************************************************************************/
#include <string>
#include <vector>
#include "toytracer.h"
#include "color.h"
#include "params.h"
#include "util.h"
#include "omp.h"
#include "utils/SimpleMatrix.hpp"

namespace __adaptive_rasterizer__ {

    using std::string;
	using std::vector;
	using std::cout;
	using std::endl;

    struct HitInfoCache
    {
        unsigned Sparseness, Width, Height;
        Vec3 Start, End;

        SimpleMatrix::Matrix<HitInfo> Cache;
        inline void Create(Vec3 eye, Vec3 Or, Vec3 dR, Vec3 dU, Vec::Size2 sz, const Scene& scene) {
            Cache.Clear();
            Cache = SimpleMatrix::Matrix<HitInfo>(Vec::Size2(sz.w / Sparseness, sz.h / Sparseness));

            for (uint y = 0; y < sz.h; y += Sparseness)
                for (uint x = 0; x < sz.w; x += Sparseness)
                {
                    Ray ray;
                    ray.origin = eye;
                    ray.generation = 1;
                    ray.type = sparse_rast_ray;
                    ray.loc = { int(x),int(y) };
                    ray.direction = Unit(Or + (x + 0.5) * dR - (y + 0.5) * dU);

                    auto& hit = Cache[y / Sparseness][x / Sparseness];
                    hit.distance = Infinity;
                    if (scene.Cast(ray, hit)) hit.ray = ray;
                }

            SimpleMatrix::Matrix <double> dist(Cache.size);

            double mn = Infinity, mx = 0;
            for (uint y = 0; y < sz.h; y += Sparseness)
                for (uint x = 0; x < sz.w; x += Sparseness)
                {
                    double d = Cache[y / Sparseness][x / Sparseness].distance;
                    if (d < mn) mn = d;
                    if (d > mx) mx = d;
                    dist[y / Sparseness][x / Sparseness] = d;
                }

            SimpleMatrix::Matrix <double> distV(Cache.size), distH(Cache.size);

            for2d(dist.size) distV[y][x] = dist.DotAt({int(x), int(y)}, SimpleMatrix::SobelV);
            for2d(dist.size) distH[y][x] = dist.DotAt({int(x), int(y)}, SimpleMatrix::SobelH);

            std::ofstream  fv("gradV.raw", ios::binary|ios::out); 
            std::ofstream  fh("gradH.raw", ios::binary|ios::out); 
            
            fv.write((char*)(distV[0]), distV.size()*sizeof(double));
            fh.write((char*)(distH[0]), distH.size()*sizeof(double));

            fv.flush(); fv.close();
            fh.flush(); fh.close();
            
            mx -= mn;
            SimpleMatrix::Matrix <uchar> udist(dist.size);
            for (uint i = 0; i < dist.size(); ++i) {
                double dh = distH[0][i], dv = distV[0][i];
                dist[0][i] = sqrt(dh*dh + dv*dv);
            }

            for (uint i = 0; i < dist.size(); ++i) {
                double& d = dist[0][i];
                d = 255 * (d - mn) / mx; 
                udist[0][i] = d;
            }

            udist.WriteAsImage("depthMap");

            udist.Clear();
            dist.Clear();
            distH.Clear();
            distV.Clear();
        }

        bool operator()(const Ray& ray, HitInfo& hit)
        {
            HitInfo& cachedHit = Cache[ray.loc.y/Sparseness][ray.loc.x / Sparseness];
            if (!almostEqual(cachedHit.ray.direction * ray.direction, 1.0)) return false;
            
            hit = cachedHit; return true;
        }

        inline void Destroy() { Cache.Clear();  }
        HitInfoCache() : Cache(Vec::Size2(0,0)) {}
        ~HitInfoCache() { Destroy(); }
    };

	struct adaptive_rasterizer : Rasterizer {
        adaptive_rasterizer() : SparseCache() {}
		virtual ~adaptive_rasterizer() {}
		virtual Raster *Rasterize(const Camera &, const Scene &);
		virtual Plugin *ReadString(const string &params);
		virtual string MyName() const { return "adaptive_rasterizer"; }
		virtual bool Default() const { return true; }
		virtual bool  GetHitInfo(const Ray& , HitInfo&);
        HitInfoCache SparseCache;
	};

	REGISTER_PLUGIN(adaptive_rasterizer);

	Plugin *adaptive_rasterizer::ReadString(const string &params)
	{
        ParamReader get(params);
        if (get["rasterizer"] && get[MyName()]
            && get["sparseness"] && get[SparseCache.Sparseness])
        {
            return new adaptive_rasterizer(*this);
        }
        return NULL;
	}

    bool adaptive_rasterizer::GetHitInfo(const Ray& ray, HitInfo& hit) { return SparseCache(ray, hit); }

	Raster *adaptive_rasterizer::Rasterize(const Camera &cam, const Scene &scene)
	{
		const double xmin(cam.x_win.min);
		const double ymax(cam.y_win.max);
		const double width(cam.x_win.Length());
		const double height(cam.y_win.Length());

		const Vec3 G(Unit(cam.lookat - cam.eye));          // Gaze direction.
		const Vec3 U(Unit(cam.up / G));                    // Up vector.
		const Vec3 R(Unit(G ^ U));                         // Right vector.
		const Vec3 Or(cam.vpdist * G + xmin * R + ymax * U);  // "Origin" of the 3D raster.
		const Vec3 dR(width  * R / cam.x_res);                // Right increments.
		const Vec3 dU(height * U / cam.y_res);                // Up increments.

	    Ray ray;
		ray.origin = cam.eye;     // All primary rays originate from the eye.
		ray.generation = 1;           // Primary rays are first-generation.
		ray.type = sparse_rast_ray;

        SparseCache.Create(cam.eye, Or, dR, dU, { cam.x_res, cam.y_res }, scene);

		Raster *raster(new Raster(cam.x_res, cam.y_res));

        cout << "Percentage done:   0"; cout.flush();
		ray.type = sparse_rast_ray; // These rays have no special meaning.
		for (unsigned y = 0; y < cam.y_res; y++)
		{
            int k = int(100 * (double(y) / cam.y_res));
            cout << rubout(k) << k;  cout.flush();

			for (unsigned x = 0; x < cam.x_res; x++)
			{
                ray.loc = { int(x),int(y) };
				ray.direction = Unit(Or + (x + 0.5) * dR - (y + 0.5) * dU);
				raster->pixel(y, x) = scene.Trace(ray);
			}
		}
		cout << endl;

        SparseCache.Destroy();
		return raster;
	}


} // namespace __basic_rasterizer__
