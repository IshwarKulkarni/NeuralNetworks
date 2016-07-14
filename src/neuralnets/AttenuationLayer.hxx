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

#ifndef __ATTENUATION_LAYER_INCLUDED__
#define __ATTENUATION_LAYER_INCLUDED__

#include <random>
#include "Network.hxx"
#include "Layer.hxx"

class Network;
class AttenuationLayer : public Layer
{
public:
    AttenuationLayer(std::string name, Vec::Size3 size, float_t mean, float_t sd, const Network* nn, Layer* prev) :
        Layer("AttenuationLayer-" + name,
            size() ? size : prev->GetOutput().size,
            size() ? size : prev->GetOutput().size,
            "Sigmoid", prev),
        distribution(mean,sd),
        Mean(mean), SD(sd),
        NN(nn)
    {
        LGrads.Clear(); Grads.Clear(); PGrads.Clear();
    }

    virtual inline void ForwardPass()
    {

        if (NN->GetCurretnStatus() == NetworkStatus::Training)
            for (size_t i = 0; i < Input.size(); ++i)
                Output[i] = distribution(generator) + Input[i];
        else
            for (size_t i = 0; i < Input.size(); ++i)
                Output[i] = Input[i];

        Next->ForwardPass();
    }

    virtual void BackwardPass(Volume& backError)
    {
        
        if (Prev) Prev->BackwardPass(backError);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Summary") != std::string::npos)
            out << "\n--> Summary for " << Name
            << "\nInput Size  : " << Input.size
            << "\nMean        : " << Mean 
            << "\nRoot Sigma  : " << SD
            << "\nOutput Size : " << Output.size
            << "\n";

        if (all || printList.find("full") != std::string::npos)
            Layer::Print(printList, out);

        out.flush();
    }
    virtual ~AttenuationLayer() {}
private:
    std::default_random_engine generator;
    std::normal_distribution<float_t> distribution;
    const float_t Mean, SD;

    const Network* NN;
};

#endif
