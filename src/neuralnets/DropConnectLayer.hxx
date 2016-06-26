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

#ifndef __DROPCONNECT_LAYER_INCLUDED__
#define __DROPCONNECT_LAYER_INCLUDED__

#include "Network.hxx"
#include "Layer.hxx"

class Network;
class DropConnectLayer : public Layer
{
public:
    DropConnectLayer(std::string name, Vec::Size3 size, double dropRate, const Network* nn, Layer* prev) :
        Layer("DropConnect-" + name, 
            size() ?size : prev->GetOutput().size,
            size() ?size : prev->GetOutput().size,  
            "Sigmoid", prev),
        DropRate(dropRate), Scale(1 / (1 - dropRate)), Mask(Input.size()), NN(nn)
    {
        LGrads.Clear(); Grads.Clear(); PGrads.Clear();
        if (dropRate > 1)  throw std::runtime_error("Drop rate is invalid: " + std::to_string(dropRate));
    }

    virtual inline void ForwardPass()
    {
       if (NN->GetCurretnStatus() == NetworkStatus::Training) 
            for (size_t i = 0; i < Mask.size();++i)
                Output[i] = ((Mask[i] = (Utils::URand(1.0) < DropRate)) * Scale) * Input[i];
       else
           for (size_t i = 0; i < Mask.size(); ++i)
               Output[i] = Input[i];

       Next->ForwardPass();
    }

    virtual void BackwardPass(Volume& backError)
    {
        for (size_t i = 0; i < Mask.size(); ++i)
            backError[i] *= Mask[i];

        if (Prev) Prev->BackwardPass(backError);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Summary") != std::string::npos)
            out << "\n--> Summary for " << Name
            << "\nInput Size  : " << Input.size
            << "\nDrop Rate   : " << DropRate
            << "\nOutput Size : " << Output.size
            << "\n";

        if (all || printList.find("full") != std::string::npos)
            Layer::Print(printList, out);

        out.flush();
    }
    virtual ~DropConnectLayer() {}
private:

    const double DropRate, Scale;
    std::vector<bool> Mask;
    const Network* NN;
};

#endif
