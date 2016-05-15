#ifndef FILTER_INCLUDED
#define FILTER_INCLUDED

#include "utils/Exceptions.hxx"
#include "imageprocessing/ImageGeneric.hxx"

#include <math.h>

using namespace std;

struct FilterParameters;

static const float PI     = 3.1415926535897932384626433832795f;
static const float E      = 2.7182818284590452353602874713527f;
static const float Sqrt2  = 1.4142135623730950488016887242097f;

struct FilterParameters
{
    enum FilterDirr
    {
        VERTICAL, 
        HORIZONTAL
    };

    FilterParameters(uint h, uint w, uint nc):Height(h), Width(w), NumComponents(nc)
    {
    }
    uint Height;
    uint Width;
    uint NumComponents;

    virtual Matrix2D<float>* GetFilterValues(uint channel) const
    {
        return nullptr;
    }

protected:
    FilterParameters(){}
};

struct Gaussian2DParameters : public FilterParameters
{
    float Sigma1; // along height, i.e vertical
    float Sigma2; // along width, i.e Horizontal
    Gaussian2DParameters(uint radiusV, uint radiusH, float s1, float s2=-1): 
        FilterParameters(radiusV*2+1,radiusH*2+1,2),
        Sigma1(s1), 
        Sigma2(s2 == -1 ? s1 : s2){}

    virtual Matrix2D<float>* GetFilterValues(uint nc)  const
    {
        int length = (int)(nc == VERTICAL ? Height : Width);
        float   sig = (nc == 1 ? Sigma1 : Sigma2);

        float* kernel = new float[length];

        for (int i = 0; i < length; i++)
            kernel[i] = 1/( sig * sqrt(2*PI));

        for (int x = -length/2; x <= length/2; ++x)
            kernel[x + length/2] *= (float) exp( -0.5 * (SQR( ((float)x)/sig ) ) ) ;

        auto* ret = (nc == 1 ?  new Matrix2D<float>(length,1,kernel) : new Matrix2D<float>(1,length,kernel));

        return ret;
    }
private:
    Gaussian2DParameters(){}
};


struct Gaussian1DParameters : public Gaussian2DParameters
{
    Gaussian1DParameters(FilterDirr t, uint length, float s1) : 
        Gaussian2DParameters(
        t ==  VERTICAL ? length : 0, 
        t ==  HORIZONTAL ? length : 0, s1), Dirr(t)
    {
        NumComponents = 1; // gaussian sets it 2
    }

    FilterDirr Dirr;

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {
        return Gaussian2DParameters::GetFilterValues(Dirr);
    }
};

struct BoxParameters: public FilterParameters
{
    BoxParameters(uint h, uint w=0): 
        FilterParameters(h,w==0?h:w,2){}

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {
        int length = (int)(nc == 1 ? Height : Width);

        float* kernel = new float[length];

        for (int i = 0; i < length; i++)
            kernel[i] = 1.f/length;

        auto* ret = nc == 1 ?  new Matrix2D<float>(length,1,kernel) 
                             : new Matrix2D<float>(1,length,kernel);
        return ret;
    }

};


struct SharpenFilterParameters : public FilterParameters
{
    enum SharpenType
    {
        NARROW, 
        WIDE
    };

    SharpenType ThisType;
    SharpenFilterParameters(SharpenType T = NARROW) : ThisType(T), FilterParameters(1,1,2){}

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {          
        static const float Narrow[] = { -0.5,1.f,-0.5f };
        static const float Wide[] =   {-.25f,-0.25f,1,-.25f,-.25f };
        
        if (ThisType == SharpenType::NARROW)
            if(nc == 0) 
                return new Matrix2D<float>(1,3,Narrow);
            else                              
                return new Matrix2D<float>(3,1,Narrow);
        else
            if(nc==0)
                return new Matrix2D<float>(1,5,Wide);
            else
                return new Matrix2D<float>(5,1,Wide);
    }
};


struct NonMaxSupFilter : public FilterParameters
{
    NonMaxSupFilter() : FilterParameters(0,0,1){}

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {          
        static const float Narrow[] = { -1,-1,-1,-1,8,-1,-1,-1,-1,};
        
        return new Matrix2D<float>(3,3,Narrow);
    }
};

struct SobelParameters : public FilterParameters
{

    FilterDirr ThisType;
    
    SobelParameters(FilterDirr t) : FilterParameters(0,0,2), ThisType(t){}

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {          
        //      | 1 |                | 1 |
        //Gx =  | 2 |[1,0,-1]   Gy = | 0 |[1,2,1]
        //      | 1 | ,              |-1 |

        //static const float k1[] = {3, 10, 3};
        //static const float k2[] = {3, 0, -3};

        static const float k1[] = {1, 2,  1};
        static const float k2[] = {1, 0, -1};
        
        if (ThisType == FilterDirr::HORIZONTAL)
            if(nc == 0) 
                return new Matrix2D<float>(1,3,k1);
            else
                return new Matrix2D<float>(3,1,k2);
        else
            if(nc==0)
                return new Matrix2D<float>(1,3,k2);
            else
                return new Matrix2D<float>(3,1,k1);
    }

};

struct GaussianDiffParameters : public Gaussian1DParameters // Gaussian Difference, Gx' (not DoG) filter params;
{
    GaussianDiffParameters(FilterDirr t, uint length, float s1) : Gaussian1DParameters(t,length, s1)
    {
    }

    virtual Matrix2D<float>* GetFilterValues(uint nc) const
    {
        auto* g = Gaussian1DParameters::GetFilterValues(Dirr);
        
        (*g)(0) = 0;
        for (uint i = 1; i < Width*Height-1; i++)
            (*g)(i)  = (*g)(i) - (*g)(i-1);

        (*g)(Width*Height-1) = 0 ; // one wasted support value but now it's symmetric
        return g;
    }
};

#endif
