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

#ifndef __MAX_POOLING_LAYER_INCLUDED__
#define __MAX_POOLING_LAYER_INCLUDED__

#include "Layer.hxx"

struct MaxPoolingLayerDesc
{
    std::string Name;
    std::string Activation;
    Vec::Size2  WindowSize;
    Vec::Size3  InputSize;
        
    inline MaxPoolingLayerDesc() :
        Name("<Unnamed>"),
        Activation("Sigmoid"),
        WindowSize(2, 2){}

    inline MaxPoolingLayerDesc(std::string name, std::string actName, Vec::Size2 windwSize) :
        Name(name),
        Activation(actName),
        WindowSize(windwSize){};

    inline Vec::Size3 GetOpSize(class Layer* prev) const {
        Vec::Size3 inSz = prev ? prev->GetOutput().size : InputSize;
        if (!inSz())
            throw std::invalid_argument("Input size cannot be determined for averaging layer\n");
        if (inSz() < WindowSize())
            throw std::invalid_argument("Max pooling window is larger than input size\n");

        if( inSz.x % WindowSize.x || inSz.y % WindowSize.y)
            throw std::invalid_argument("Max pooling window leaves remainder\n");

        return Vec::Size3(iDivUp(inSz.x, WindowSize.x), iDivUp(inSz.y, WindowSize.y), inSz.z);
    }

private:
    MaxPoolingLayerDesc(MaxPoolingLayerDesc&);
    MaxPoolingLayerDesc& operator=(MaxPoolingLayerDesc&);
};

class MaxPoolingLayer  : public Layer
{
    MaxPoolingLayerDesc Desc;
    SimpleMatrix::Matrix3< Vec::Vec2<unsigned char> >  MaxIndices; // indices to the max value.

public:
    MaxPoolingLayer(const MaxPoolingLayerDesc& desc, Layer* prev = 0) :
        Layer(  "MaxPooling-" + desc.Name, prev->GetOutput().size, desc.GetOpSize(prev), desc.Activation, prev),
        Desc(desc.Name, desc.Activation,desc.WindowSize),
        MaxIndices(Output.size)
    {
        if (Desc.WindowSize.x > std::numeric_limits<unsigned char>::max() ||
            Desc.WindowSize.y > std::numeric_limits<unsigned char>::max())
            throw std::runtime_error("Max pooling windows size is too large at (" 
                    + std::to_string(desc.WindowSize.x) + ", " + std::to_string(desc.WindowSize.y) +  ")" );
    }

    virtual void ForwardPass()
    {
        Vec::Size3 oLoc = { 0,0,0 };
        for (size_t z = 0; z < Input.size.z; ++z)
            for (size_t y = 0; y < Input.size.y; y += Desc.WindowSize.y)
                for (size_t x = 0; x < Input.size.x; x += Desc.WindowSize.x)
        {
            float_t m = -std::numeric_limits<float_t>::max();
            Vec::Vec2<unsigned char> mIdx = { 0,0 };
            for (unsigned char dy(0); dy < (unsigned char)(Desc.WindowSize.y); ++dy)
            {
                for (unsigned char dx (0); dx < (unsigned char)(Desc.WindowSize.x); ++dx)
                {
                    auto& a = Input.at(z, y + dy, x + dx);
                    if (a > m) m = a, mIdx = { dx,dy };
                }
            }
            oLoc = { x / Desc.WindowSize.x, y / Desc.WindowSize.y, z };
            Output.at(oLoc) = Act->Function(m, LGrads.at(oLoc));
            MaxIndices.at(oLoc) = mIdx;
        }

        if(Next) Next->ForwardPass();
    }

    virtual void BackwardPass(Volume& backError)
    {
        if (!Prev)  return;
        for3d(Grads.size) Grads.at(z, y, x) = LGrads.at(z, y, x) * backError.at(z, y, x);
        PGrads.Fill(0.);

        for3d(Output.size)
        {
            auto& mIdx = MaxIndices.at(z, y, x);
            PGrads.at(z, y * Desc.WindowSize.y + mIdx.y, x * Desc.WindowSize.x + mIdx.x) = Grads.at(z,y, x);
        }

        Prev->BackwardPass(PGrads);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Summary") != std::string::npos)
            out << "\n--> Summary for " << Name << "\t| " << Act->Name << "\t| Eta: " << Eta
                << "\nInput Size  : " << Input.size
                << "\nWindow Size : " << Desc.WindowSize
                << "\nOutput Size : " << Output.size
                << "\n";

        if(all || printList.find("full") != std::string::npos)
            Layer::Print(printList,out);

        out.flush();
    }

    virtual ~MaxPoolingLayer() { MaxIndices.Clear(); }
};

#endif
 