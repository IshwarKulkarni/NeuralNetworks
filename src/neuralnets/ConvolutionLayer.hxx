/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of NeuralNetwork Project by
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source
along with  rest of the project. However any copy/redistribution,
including but not limited to compilation to binaries, must carry
this header in its entirety. A note must be made about the origin
of your copy.

NeuralNetwork is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifndef __CONVOLUTION_LAYER_INCLUDED__
#define __CONVOLUTION_LAYER_INCLUDED__

#include "Layer.hxx"

#ifdef CUDA_PROJECT
#include "CudaLayers.cuh"
#endif

struct ConvLayerDesc
{
    std::string Name;
    std::string Activation;
    std::string ConnectionTable;

    Vec::Size3  IpSize;
    Vec::Size2  KernelSize;
    Vec::Size2  KernelStride;

    size_t      NumberOfKernels;
    bool        PaddedConvolution;

    inline ConvLayerDesc() :
        Name("<Unnamed>"),
        Activation("Sigmoid"),
        IpSize(0, 0, 0),
        KernelSize(0, 0),
        KernelStride(0, 0),
        NumberOfKernels(0),
        PaddedConvolution(true){};

    inline ConvLayerDesc(std::string name, std::string actName, Vec::Size3 inSize, Vec::Size2 krSize, Vec::Size2 krStride,
        size_t numKr, size_t numO = 0) :
        Name(name),
        Activation(actName),
        IpSize(inSize),
        KernelSize(krSize),
        KernelStride(krStride),
        NumberOfKernels(numKr),
        PaddedConvolution(true) {};

private:
    ConvLayerDesc(ConvLayerDesc&);
    ConvLayerDesc& operator=(ConvLayerDesc&);
};

template<bool PConnected>
struct Kernel : public Volume
{
    const Vec::Size2 Stride;  // when stride is 1, max overlap; i.e. windows moves 1pixel 
    float_t Bias;
    const bool Padded;
    const Vec::Loc CStart, CEnd, PadSize;
    
    inline Kernel(const ConvLayerDesc& desc, Vec::Size3 ipSize) :
        Volume(Vec::Size3(desc.KernelSize.x | 1, desc.KernelSize.y | 1, ipSize.z)), // odd sizes only
        Stride(desc.KernelStride),
        Padded(desc.PaddedConvolution),
        CStart(!Padded ? size / 2 : Vec::Size3()),
        CEnd(Padded ? ipSize.x : ipSize.x - size.x/2, Padded ? ipSize.y : ipSize.y - size.y/2),
        PadSize(Padded ? size / 2 : Vec::Size3())
    {
        float_t rs = float_t(1. / sqrt(size()));

		Bias = CNN_DEBUG ? float_t(0.1) : Utils::URand(-rs, rs);
		for (auto& d : *this) d = CNN_DEBUG ? float_t(0.1) : Utils::URand(-rs, rs);
    }

    inline void Apply(const Volume& IpImage, const Activation* act, Frame Output, Frame LGrads, bool* connection)
    {
        // TODO split this so that we can do for3d, and skip on PConnected. I.e. use Matrix<>::DotAt rather than Matrix3<>
        for (int oy = 0, y = CStart.x ; y < CEnd.y; y += Stride.y, ++oy)
            for (int ox = 0, x = CStart.x ; x < CEnd.x; x += Stride.x, ++ox)
                Output.at(oy, ox) = act->Function(IpImage.DotAt<PConnected>({ x,y }, *this, connection) + Bias, LGrads.at(oy, ox));
    }

    inline void BackwardPass(Frame gradients, const Volume& inputs, Volume& pgrads, Volume& dW, float_t eta, bool* conn)
    {
        GetPGrads(gradients, pgrads, conn);
        ChangeWeights(gradients, inputs, dW, eta, conn);
    }

    inline void GetPGrads(Frame& gradients, Volume& pgrads, bool* connection)
    {
        for (size_t gy = 0; gy < gradients.Height(); ++gy)
            for (size_t gx = 0; gx < gradients.Width(); ++gx)
            {
                Vec::Loc3 s = { int(gx * Stride.x - PadSize.x) , int(gy * Stride.y - PadSize.y),0 };
                auto      e = s, is = s; e += size;

                s.x = MAX(0, s.x), e.x = MIN(int(pgrads.Width()), e.x);
                s.y = MAX(0, s.y), e.y = MIN(int(pgrads.Height()), e.y);

                float_t grad = gradients.at(gy, gx);
                for (size_t z = s.z; z < size_t(e.z); ++z)
                    if (!PConnected || connection[z])
                        for (size_t y = s.y; y < size_t(e.y); ++y)
                            for (size_t x = s.x; x < size_t(e.x); ++x)
                            {
                                pgrads.at(z, y, x) += grad * at(z, y - is.y, x - is.x);
                                /*if(x == pgrads.size.x/2 && y == pgrads.size.y / 2)
                                Logging::Log << x << " " << y << "\t" << x - is.x << " " << y - is.y
                                    << "\t" << gx << " " << gy << "\n";*/
                            }
            }
    }

    inline void ChangeWeights(Frame grads, const Volume& ipt, Volume& dW, float_t eta, bool* connection)
    {
        dW.Fill(0.);
        for (size_t z = 0; z < size.z; ++z)
            if (!PConnected || connection[z])
            for2d(size)
                dW.at(z, y, x) += 
                ipt(z).DotAt(Vec::Loc( x + grads.Width() / 2 - PadSize.x, y + grads.Height() / 2 - PadSize.y ), grads );
        
        for3d(size)
            at(z, y, x) -= eta* dW.at(z, y, x);

        float_t dB = 0;
        for (auto& g : grads)dB += g;
        Bias -= eta * dB;
    }

    static inline Vec::Size3 GetOpSize3(const ConvLayerDesc& desc, class Layer* prev) 
    {
        Vec::Size3 inSz = prev ? prev->GetOutput().size : desc.IpSize, stride = desc.KernelStride;
        
        if (!desc.PaddedConvolution) inSz.x -= desc.KernelSize.x - 1, inSz.y -= desc.KernelSize.y - 1;

        return Vec::Size3(iDivUp(inSz.x, stride.x), iDivUp(inSz.y, stride.y), desc.NumberOfKernels);
    }

    std::ostream& Print(std::ostream& stream)
    {
        //float_t sum = 0; for (auto& w : *this) sum += w;
        stream  << " |  Stride: " << Stride << " | Bias: " << Bias << "\nWeights" << *this;
        return stream;
    }
    
    template<typename OutT>
    OutT Copy(OutT out)
    {
        (*std::copy(this->begin(), this->end(), out)) = Bias;
        return ++out;
    }
};

class ConvolutionLayerBase : public Layer {
protected:
    ConvolutionLayerBase(const ConvLayerDesc& desc, Layer* prev = 0) :
        Layer("ConvLayer-" + desc.Name, desc.IpSize, Kernel<true>::GetOpSize3(desc, prev), desc.Activation, prev) {}
};

template<bool PartiallyConnected>
class ConvolutionLayer : public ConvolutionLayerBase
{
    SimpleMatrix::Matrix<bool> ConnTable;
public:
    ConvolutionLayer(const ConvLayerDesc& desc, SimpleMatrix::Matrix<bool>& connTable, Layer* prev = 0) :
        ConvolutionLayerBase(desc, prev),
        ConnTable(connTable.Copy())
#ifdef CUDA_PROJECT
        , CudaLayer(
            {desc.KernelSize.x, desc.KernelSize.y, Input.size.z}, 
            desc.NumberOfKernels, Input.size, Output.size, desc.KernelStride, Act->Id )
#endif
    {
        if (PartiallyConnected && !ConnTable.size())
            throw std::invalid_argument("Cannot have Partially connected convolution layer with no connection table.");

        if (PartiallyConnected && prev && prev->GetOutput().Depth() != ConnTable.Width())
        {
            std::cerr << "Previous Layer is "; prev->Print("Summary", std::cerr);
            std::cerr << " with [" << prev->GetOutput().Depth() << "] output frame(s) "
                << " but connection table has [" << ConnTable.Width()
                << "] rows:" << ConnTable;
            throw std::runtime_error(" Bad connection table description ");
        }

        for (unsigned i = 0; i < desc.NumberOfKernels; ++i)
        {
            Kernels.push_back(Kernel<PartiallyConnected>(desc, Layer::GetInput().size));
            dW.push_back(Volume(Kernels.back().size));
        }

#ifdef CUDA_PROJECT
        float_t* dev = CudaLayer.Kernels.devData;
        for (auto& k : Kernels) dev = k.Copy(dev);
#endif
    }

    virtual void ForwardPass()
    {
        for (unsigned i = 0; i < Kernels.size(); ++i)
            Kernels[i].Apply(Input, Act, Output(i), LGrads(i), PartiallyConnected ? ConnTable.data[i] : nullptr);

#ifdef CUDA_PROJECT
        CudaLayer.ForwardPass(Input.data[0][0], Kernels[0].Padded).CompareTo(Output.begin(), Output.end(), Name+"OP");
#endif
        if (Next) Next->ForwardPass();
    }

    virtual void BackwardPass(Volume& backError)
    {
        for3d(Grads.size) Grads.at(z, y, x) = LGrads.at(z, y, x) * backError.at(z, y, x);

        if (Prev)
        {
            PGrads.Fill(0.);
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].BackwardPass(Grads(i), Prev->GetOutput(), PGrads, dW[i], Eta, PartiallyConnected ? ConnTable.data[i] : nullptr);

            Prev->BackwardPass(PGrads);

#ifdef CUDA_PROJECT
            CudaLayer.BackwardPass(backError.begin(), Act->Eta, Kernels[0].Padded, Grads.begin())
                    .CompareTo(PGrads.begin(), PGrads.end(), Name + "PGrads");
#endif
        }
        else
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].ChangeWeights(Grads(i), Input, dW[i], Eta, PartiallyConnected ? ConnTable.data[i] : nullptr);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;

        if (all || printList.find("Summary") != std::string::npos)
        {
            out << "\n--> Summary for " << Name 
                << "\nActivation  : " << Act->Name << "\t| Eta: " << Eta
                << "\nInput Size  : " << Input.size
                << "\nNum Kernels : " << Kernels.size()
                << "\nKernel Size : " << Kernels[0].size << " | Kernel Stride: " << Kernels[0].Stride
                << "\nOutput Size : " << Output.size 
                << "\nInput Padded: " << (Kernels[0].Padded ? " True\n" : " False\n");
            if (PartiallyConnected)
                out << "Partial connection:" << std::noboolalpha <<ConnTable;
        }

        if (all || printList.find("Kernels") != std::string::npos)
            for (unsigned i = 0; i < Kernels.size(); ++i)
                out << "\nKernel " << i << " : " << Kernels[i];

        if (all || printList.find("full") != std::string::npos)
            Layer::Print(printList, out);

        out.flush();
    }

    virtual ~ConvolutionLayer() {
        for (auto& k : Kernels) k.Clear();
        for (auto& v : dW) v.Clear();
        if (PartiallyConnected) ConnTable.Clear();
    }


private:
    std::vector<Kernel<PartiallyConnected>>  Kernels;
    std::vector<Volume> dW;
#ifdef CUDA_PROJECT
	CudaConvolutionLayer CudaLayer;
#endif
};

#endif
