#ifndef __AVERAGE_POOLING_LAYER_INCLUDED__
#define __AVERAGE_POOLING_LAYER_INCLUDED__

#include "Layer.hxx"

struct AvgPooLayerDesc
{
    std::string Name;
    std::string Activation;
    Vec::Size2  WindowSize;
    Vec::Size3  InputSize;
    
    inline AvgPooLayerDesc() :
        Name("<Unnamed>"),
        Activation("Sigmoid"),
        WindowSize(2, 2){}

    inline AvgPooLayerDesc(std::string name, std::string actName, Vec::Size2 windwSize) :
        Name(name),
        Activation(actName),
        WindowSize(windwSize){};


private:
    AvgPooLayerDesc(AvgPooLayerDesc&);
    AvgPooLayerDesc& operator=(AvgPooLayerDesc&);
};

//small windows that average inputs (& add bias) over that window and apply activation 
// Each average is further multiplied by a weight (same across all windows for a frame)
struct AveragingKernels : public std::vector<Vec::Vec2<double>>  // x is weight, y is bias
{
     Vec::Size2  WindowSize;
     const double Scale;
    
     inline AveragingKernels(Vec::Size2 inSize, size_t inSizeZ) :
         std::vector<Vec::Vec2<double>>(inSizeZ),
         WindowSize(inSize),
         Scale(1./WindowSize())
     {
         double rs = 1. / sqrt(size());

         if (CNN_DEBUG) std::fill(begin(), end(), Vec::Vec2<double>(.1, .1));
         else 
             for (auto& d : *this) d = Vec::Vec2<double>(Utils::URand(-rs, rs), Utils::URand(-rs, rs));
     }

    inline void Apply(Volume& in, const Activation* act, Volume out, Volume LGrads) 
    {
        for (size_t z = 0; z < out.size.z; ++z)
            for (size_t y = 0; y < out.size.y; ++y)
            for (size_t x = 0; x < out.size.x; ++x)
            {
                double res = 0;
                for (size_t wy = 0; wy < WindowSize.y; ++wy)
                    for (size_t wx = 0; wx < WindowSize.x; ++wx)
                        res += in.at(z, y*WindowSize.y + wy, x*WindowSize.x + wx);

                res = res * Scale * at(z).x + at(z).y;
                out.at(z, y, x) = act->Function(res, LGrads.at(z, y, x));
            }
    }

    inline void BackwardPass(Volume grads, const Volume& inputs, Volume& pgrads, double eta)
    {
        GetPGrads(grads, pgrads), ChangeWeights(grads, inputs, eta);
    }

    inline void GetPGrads(Volume& grads, Volume& pgrads)
    {
        for3d(grads.size)
        {
            double pg = Scale * grads.at(z, y, x) * at(z).x;
            for (size_t wy = 0; wy < WindowSize.y; ++wy)
                for (size_t wx = 0; wx < WindowSize.x; ++wx)
                    pgrads.at(z, y*WindowSize.y + wy, x*WindowSize.x + wx) = pg;
        }
    }

    inline void ChangeWeights(Volume grads, const Volume& ipt, double eta)
    {
        std::vector<Vec::Vec2<double>> dW(grads.size.z);
        for (size_t z = 0; z < grads.size.z; ++z)
        {
            double diff = 0;
            for2d(grads.size)
            {
                double g = grads.at(z, y, x);
                for (size_t wy = 0; wy < WindowSize.y; ++wy)
                    for (size_t wx = 0; wx < WindowSize.x; ++wx)
                        diff += g * ipt.at(z, y*WindowSize.y + wy, x*WindowSize.x + wx);

                dW.at(z).y += g; // bias
            }
            dW.at(z).x += diff * Scale;
        }

        for (size_t z = 0; z < grads.size.z; ++z)
            (*this)[z] -= (dW[z] * eta);
    }

    static inline Vec::Size3 GetOpSize(const AvgPooLayerDesc& desc, class Layer* prev) {
        Vec::Size3 inSz = prev ? prev->Out().size : desc.InputSize;
        if (!inSz())
            throw std::invalid_argument("Input size cannot be determined for averaging layer\n");
        if (inSz() < desc.WindowSize())
            throw std::invalid_argument("Averaging window is larger than input size\n");
        return Vec::Size3( inSz.x / desc.WindowSize.x,inSz.y / desc.WindowSize.y, inSz.z);
    }
};

class AveragePoolingLayer  : public Layer
{
public:
    AveragePoolingLayer(const AvgPooLayerDesc& desc, Layer* prev = 0) :
        Layer(  "AveragePooling-" + desc.Name, prev->Out().size,
                AveragingKernels::GetOpSize(desc, prev), desc.Activation, prev),
                Windows(desc.WindowSize, prev->Out().size.z){}

    virtual Volume& ForwardPass(Volume& input)
    {
        if (!Prev) Input = input;
        Windows.Apply(input, Act, Output, LGrads);
        
        if(Next) return Next->ForwardPass(Output);
        return Output;
    }

    virtual void BackwardPass(Volume& backError)
    {

        for3d(Grads.size) Grads.at(z, y, x) = LGrads.at(z, y, x) * backError.at(z, y, x);

        //Logging::Log  << "\nBackerror: " << backError << "\nGradients " << Grads ; // << "\nLGrads :  " << LGrads << std::endl;
        
        if (Prev)
        {
            PGrads.Fill(0.);
            Windows.BackwardPass(Grads, Prev->Out(), PGrads, Eta);

            Prev->BackwardPass(PGrads);
        }
        else
            Windows.ChangeWeights(Grads, Input, Eta);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Summary") != std::string::npos)
            out << "\n--> Summary for " << Name << "\t| " << Act->Name << "\t| Eta: " << Eta
                << "\nInput Size  : " << Input.size
                << "\nWindow Size : " << Windows.WindowSize
                << "\nOutput Size : " << Output.size
                << "\n";

        if (all || printList.find("Windows") != std::string::npos)
            for (unsigned i = 0; i < Windows.size(); ++i)
                out << "\nWindow " << i << " : " << Windows[i];

        if(all || printList.find("full") != std::string::npos)
            Layer::Print(printList,out);

        out.flush();
    }

    virtual ~AveragePoolingLayer() {  }


private:
    AveragingKernels Windows;
};

#endif
 