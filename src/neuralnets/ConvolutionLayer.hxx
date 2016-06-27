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
    double Bias;
    const bool Padded;
    
    inline Kernel(const ConvLayerDesc& desc, size_t ipSizeZ) :
        Volume(Vec::Size3(desc.KernelSize.x | 1, desc.KernelSize.y | 1, ipSizeZ)), // odd sizes only
        Stride(desc.KernelStride),
        Padded(desc.PaddedConvolution)
    {
        double rs = 1. / sqrt(size());

        Bias = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
        for (auto& d : *this) d = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
    }

    inline void Apply(const Volume& IpImage, const Activation* act, Frame Output, Frame LGrads, bool* connection)
    {
        // TODO split this so that we can do for3d, and skip on PConnected. I.e. use Matrix<>::DotAt rather than Matrix3<>
        if(Padded)
        for (size_t y = 0, oy = 0; y < IpImage.Height(); y += Stride.y, ++oy)
            for (size_t x = 0, ox = 0; x < IpImage.Width(); x += Stride.x, ++ox)
                Output.at(oy, ox) = act->Function(IpImage.DotAt<PConnected>(Vec::Loc(int(x), int(y)), *this, connection) + Bias,LGrads.at(oy, ox) );
        else
        for (size_t y = 0, oy = 0; y < IpImage.Height() - size.y + 1; y += Stride.y, ++oy)
            for (size_t x = 0, ox = 0; x < IpImage.Width() - size.x + 1; x += Stride.x, ++ox)
                Output.at(oy, ox) = act->Function(IpImage.DotCornerAt<PConnected>({ x, y }, *this, connection) + Bias, LGrads.at(oy, ox));
    }

    inline void BackwardPass(Frame gradients, const Volume& inputs, Volume& pgrads, Volume& dW, double eta, bool* conn)
    {
        GetPGrads(gradients, pgrads, conn);
        ChangeWeights(gradients, inputs, dW, eta, conn);
    }

    inline void GetPGrads(Frame& gradients, Volume& pgrads, bool* connection)
    {
        if(Padded)
        for (size_t gy = 0; gy < gradients.Height(); ++gy)
            for (size_t gx = 0; gx < gradients.Width(); ++gx)
            {
                Vec::Loc3 s = { int(gx * Stride.x - Width() / 2) , int(gy * Stride.y - Height() / 2),0 };
                auto      e = s, is = s; e += size;

                s.x = MAX(0, s.x), e.x = MIN(int(pgrads.Width()), e.x);
                s.y = MAX(0, s.y), e.y = MIN(int(pgrads.Height()), e.y);

                double grad = gradients.at(gy, gx);
                for (size_t z = s.z; z < size_t(e.z); ++z)
                    if (!PConnected || connection[z])
                        for (size_t y = s.y; y < size_t(e.y); ++y)
                            for (size_t x = s.x; x < size_t(e.x); ++x)
                                pgrads.at(z, y, x) += grad * at(z, y - is.y, x - is.x);
                
            }
        else
        for (size_t gy = 0; gy < gradients.Height(); ++gy) 
            for (size_t gx = 0; gx < gradients.Width(); ++gx)
            { 

                double grad = gradients.at(gy, gx);
                for (size_t z = 0; z < size.z; ++z)
                    if (!PConnected || connection[z])
                        for2d(size)
                            pgrads.at(z, y + gy, x + gx) += grad * at(z, y, x);
            }
    }

    inline void ChangeWeights(Frame grads, const Volume& ipt, Volume& dW, double eta, bool* connection)
    {
        dW.Fill(0.);
        if (Padded)
        {
            for (size_t z = 0; z < size.z; ++z)
                if (!PConnected || connection[z])
                for2d(size)
                    dW.at(z, y, x) += ipt(z).DotAt(Vec::Loc(x + grads.Width() / 2 - Width() / 2, y + grads.Height() / 2 - Height() / 2), grads);
        }
        else
        {
            for (size_t z = 0; z < size.z; ++z)
                if (!PConnected || connection[z]) 
                for2d(size)
                        dW.at(z, y, x) += ipt(z).DotCornerAt({x , y}, grads);
        }

        for3d(size)
            at(z, y, x) -= eta* dW.at(z, y, x);

        double dB = 0;
        for (auto& g : grads)dB += g;
        Bias -= eta * dB;
    }

    static inline Vec::Size3 GetOpSize2(const ConvLayerDesc& desc, class Layer* prev) 
    {
        Vec::Size3 inSz = prev ? prev->GetOutput().size : desc.IpSize, stride = desc.KernelStride;
        
        if (!desc.PaddedConvolution)
            inSz.x -= desc.KernelSize.x - 1, inSz.y -= desc.KernelSize.y - 1;

        return Vec::Size3(Utils::iDivUp(inSz.x, stride.x), Utils::iDivUp(inSz.y, stride.y), desc.NumberOfKernels);
    }

    std::ostream& Print(std::ostream& stream)
    {
        //double sum = 0; for (auto& w : *this) sum += w;
        stream 
            << " |  Stride: " << Stride
            << " | Bias: " << Bias
            << "\nWeights"
            << *this;

        return stream;
    }
};

class ConvolutionLayerBase : public Layer {
protected:
    ConvolutionLayerBase(const ConvLayerDesc& desc, Layer* prev = 0) :
        Layer("ConvLayer-" + desc.Name, desc.IpSize, Kernel<true>::GetOpSize2(desc, prev), desc.Activation, prev) {}
};

template<bool PartiallyConnected>
class ConvolutionLayer : public ConvolutionLayerBase
{
    SimpleMatrix::Matrix<bool> ConnTable;
public:
    ConvolutionLayer(const ConvLayerDesc& desc, SimpleMatrix::Matrix<bool>& connTable, Layer* prev = 0) :
        ConvolutionLayerBase(desc, prev),
        ConnTable(connTable.Copy())
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
            Kernels.push_back(Kernel<PartiallyConnected>(desc, Layer::GetInput().size.z));
            dW.push_back(Volume(Kernels.back().size));
        }
    }

    virtual void ForwardPass()
    {
        for (unsigned i = 0; i < Kernels.size(); ++i)
            Kernels[i].Apply(Input, Act, Output(i), LGrads(i), PartiallyConnected ? ConnTable.data[i] : nullptr);

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
            out
                << "\n--> Summary for " << Name 
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
};

#endif
