#ifndef __CONVOLUTION_LAYER_INCLUDED__
#define __CONVOLUTION_LAYER_INCLUDED__

#include "Layer.hxx"

struct ConvLayerDesc
{
    std::string Name;
    std::string Activation;
    
    Vec::Size3  IpSize;
    Vec::Size2  KernelSize;
    Vec::Size2  KernelStride;
    
    size_t      NumberOfKernels;
    size_t      NumOuputs;
    
    inline ConvLayerDesc() :
        Name("<Unnamed>"),
        Activation("Sigmoid"),
        IpSize(0, 0, 0),
        KernelSize(0, 0),
        KernelStride(0, 0),
        NumberOfKernels(0){};

    inline ConvLayerDesc(std::string name, std::string actName, Vec::Size3 inSize, Vec::Size2 krSize, Vec::Size2 krStride,
        size_t numKr, size_t numO = 0) :
        Name(name),
        Activation(actName),
        IpSize(inSize),
        KernelSize(krSize),
        KernelStride(krStride),
        NumberOfKernels(numKr){};


private:
    ConvLayerDesc(ConvLayerDesc&);
    ConvLayerDesc& operator=(ConvLayerDesc&);
};

struct Kernel : public Volume
{
    const Vec::Size2 Stride;  // when stride is 1, max overlap; i.e. windows moves 1pixel 
    double Bias;

    inline Kernel(const ConvLayerDesc& desc, size_t ipSizeZ) :
        Volume(Vec::Size3(desc.KernelSize.x | 1, desc.KernelSize.y | 1, ipSizeZ)), // odd sizes only
        Stride(desc.KernelStride)
    {
        double rs = 1. / sqrt(size());

        Bias = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
        for (auto& d : *this) d = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
    }

    inline void Apply( const Volume& IpImage, const Activation* act, Frame Output, Frame LGrads) 
    {
        for (size_t y = 0, oy = 0; y < IpImage.size.y; y += Stride.y, ++oy)
            for (size_t x = 0, ox = 0; x < IpImage.size.x; x += Stride.x, ++ox)
                Output.at(oy,ox) = act->Function( IpImage.DotAt(Vec::Loc(int(x), int(y)), *this) + Bias, LGrads.at(oy, ox));
    }

    inline void BackwardPass(Frame gradients, const Volume& inputs, Volume& pgrads, double eta)
    {
        GetPGrads(gradients, pgrads);
        ChangeWeights(gradients, inputs, eta);
    }

    inline void GetPGrads(Frame& gradients, Volume& pgrads)
    {
        for (size_t gy = 0; gy < gradients.size.y; ++gy)
            for (size_t gx = 0; gx < gradients.size.x; ++gx)
            {
                Vec::Loc3 s = { int(gx * Stride.x - size.x / 2) , int(gy * Stride.y - size.y / 2),0 };
                auto      e = s, is = s; e += size;

                s.x = MAX(0, s.x), e.x = MIN(int(pgrads.size.x), e.x);
                s.y = MAX(0, s.y), e.y = MIN(int(pgrads.size.y), e.y);

                double grad = gradients.at(gy, gx);
                for3d2(s, e)
                    pgrads.at(z, y, x) += grad * at(z, y - is.y, x - is.x);
            }
    }

    inline void ChangeWeights(Frame grads, const Volume& ipt, double eta)
    {
        for3d(size) 
            at(z, y, x) -= ipt(z).DotAt(Vec::Loc(x + grads.size.x / 2 - size.x / 2, y + grads.size.y / 2 - size.y / 2), grads);

        double dB = 0;
        for (auto& g : grads)dB += g;
        Bias -= eta * dB;
    }

    static inline Vec::Size3 GetOpSize2(const ConvLayerDesc& desc, class Layer* prev) {
        Vec::Size3 inSz = prev ? prev->Out().size : desc.IpSize, stride = desc.KernelStride;
        return Vec::Size3(Utils::iDivUp(inSz.x, stride.x), Utils::iDivUp(inSz.y, stride.y), desc.NumberOfKernels);
    }

    std::ostream& Print(std::ostream& stream)
    {
        //double sum = 0; for (auto& w : *this) sum += w;
        stream  << " |  Stride: " << Stride
                << " | Bias: " << Bias
                << "\nWeights" 
                << *this;

        return stream;
    }
};

class ConvolutionLayer  : public Layer
{
public:
    ConvolutionLayer(const ConvLayerDesc& desc, Layer* prev = 0) :
        Layer("ConvLayer-" + desc.Name, desc.IpSize, Kernel::GetOpSize2(desc, prev), desc.Activation, prev)
    {
        for (unsigned i = 0; i < desc.NumberOfKernels; ++i)
            Kernels.push_back(Kernel(desc, Layer::InputSize().z));
    }

    virtual Volume& ForwardPass(Volume& input)
    {
        if (!Prev) Input = input;
        
        for (unsigned i = 0; i < Kernels.size(); ++i)
            Kernels[i].Apply(input, Act, Output(i), LGrads(i));

        if (Next) return Next->ForwardPass(Output);

        return Output;
    }

    virtual void BackwardPass(Volume& backError)
    {
        for3d(Grads.size) Grads.at(z, y, x) = LGrads.at(z, y, x) * backError.at(z, y, x);

        if (Prev)
        {
            PGrads.Fill(0.);
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].BackwardPass(Grads(i), Prev->Out(), PGrads, Eta);

          //  Logging::Log << "backError: " << PGrads;

            Prev->BackwardPass(PGrads);
        }
        else
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].ChangeWeights(Grads(i), Input, Eta);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        
        if (all || printList.find("Summary") != std::string::npos)
            out << "\n--> Summary for " << Name << "\t| " << Act->Name << "\t| Eta: " << Eta
                << "\nInput Size  : " << Input.size
                << "\nNum Kernels : " << Kernels.size()
                << "\nKernel Size : " << Kernels[0].size << " | Kernel Stride: " << Kernels[0].Stride
                << "\nOutput Size : " << Output.size
                << "\n";
        
        if (all || printList.find("Kernels") != std::string::npos)
            for (unsigned i = 0; i < Kernels.size(); ++i)
                out << "\nKernel " << i << " : " << Kernels[i];
        
        if (all || printList.find("full") != std::string::npos)
            Layer::Print(printList,out);

        out.flush();
    }

    virtual ~ConvolutionLayer() { for (auto& k : Kernels) k.Clear(); }


private:
    std::vector<Kernel>  Kernels;
};

#endif
 