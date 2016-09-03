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

#ifndef __FULLY_CONNECTED_LAYER_INCLUDED__
#define __FULLY_CONNECTED_LAYER_INCLUDED__

#include "Layer.hxx"

#ifdef _DEBUG
#define NN_DEBUG 1
#define NN_PRINT 1
#else
#define NN_PRINT 0
#define NN_DEBUG 0
#endif

typedef std::vector<double> Row;
static const size_t BatchSize = 16;
struct Neuron
{
    Neuron(size_t N) : Weights(N + 1), dWeightsArray( BatchSize, Row(N+1) ), Batch(0), dWeights(N+1){}

    inline void InitWeights()
    {
        double rs = double(1 / sqrt(Weights.size()));
        for (auto& w : Weights)
            w = NN_DEBUG ? 0.1 : Utils::URand(rs, -rs);
    }

    template<typename IpArr>
    double ForwardPass(IpArr& ip, const Activation* Act, double& localGradient) {
        double out = Weights.back();
        for (size_t i = 0; i < Weights.size() - 1; ++i)
            out += (Weights[i] * (ip[i]));
        return  Act->Function(out, localGradient);
    }

    inline double operator[](unsigned wIdx)  const { return Weights[wIdx]; }
    inline size_t NumWeights() const { return Weights.size() - 1; }
    
    inline void BackwardPass(double eta, double grad, Volume& pgrads, const Volume& inputs)
    {
        for (size_t i = 0; i < Weights.size() - 1; ++i)
            pgrads[i] += Weights[i] * grad;

        ChangeWeights(eta, grad, inputs);
    }

    void ChangeWeights(double eta, double grad, const Volume& inputs)
    {
        dWeights.assign(dWeights.size(), 0.0);
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            dWeights[i] += grad * inputs[i];

        dWeights.back() += grad;

        for (unsigned i = 0; i < Weights.size(); ++i)
            Weights[i] -= dWeights[i] * eta;

        /*
        auto& dWeights =  dWeightsArray[Batch++];

        for (unsigned i = 0; i < Weights.size() - 1; ++i)
        dWeights[i] += grad * inputs[i];

        dWeights.back() += grad;

        if (Batch == dWeightsArr.size() - 1){
        for (size_t b = 0; b < BatchSize; ++b)
        {
        for (size_t i = 0; i < Weights.size(); ++i)
        Weights[i] -= dWeightsArray[b][i] * eta;
        dWeightsArray[b].assign(dWeightsArray[b].size(), 0);
        }
        Batch = 0;
        }
        */
    }


   std::ostream& Print(std::ostream& stream) const
    {
        for (unsigned i = 0; i < NumWeights(); ++i)
            stream << Weights[i] << ",\t";

        stream << " | " << Weights.back() << "\n";

        return stream;
    }

private:
    Row Weights, dWeights; // Nth index is Bias
    std::vector<Row> dWeightsArray;
    size_t Batch;
};


class FullyConnectedLayer : public Layer
{
public:

    FullyConnectedLayer(std::string name, unsigned NumInputs, unsigned NumNeurons,  std::string actName, Layer* prev = 0) :
        Layer(
            "FCLayer-" + name, 
            { NumInputs? NumInputs : prev->GetOutput().size(),1,1}, 
            { NumNeurons,1,1},
            actName,prev),
        Neurons(NumNeurons, Neuron(NumInputs ? NumInputs : prev->GetOutput().size()))
    {
        if (Prev && NumInputs > Prev->GetOutput().size())
            std::invalid_argument("NumInputs in FCLayer is larger than output of previous layer.");
        
        for (auto& n : Neurons) n.InitWeights();
    }

    virtual void ForwardPass()
    {
        for (size_t i = 0; i < Neurons.size(); i++)
            Output[i] = Neurons[i].ForwardPass(Input.data[0][0], Act, LGrads[i]);

        if (Next) Next->ForwardPass();
    }

    virtual void BackwardPass(Volume& backError)
    {
        for (size_t n = 0; n < Neurons.size(); ++n)
            Grads[n] = backError[n] * LGrads[n];

        if (Prev)
        {
            PGrads.Fill(0.0);
            for (unsigned n = 0; n < Neurons.size(); ++n)
                Neurons[n].BackwardPass(Eta, Grads[n], PGrads, Input);

            Prev->BackwardPass(PGrads);
        }
        else
            for (size_t j = 0; j < Neurons.size(); ++j)
                Neurons[j].ChangeWeights(Eta, Grads[j], Input);
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos, printed = false;
        if (all || printList.find("Summary") != std::string::npos)
        {
            out << "\n--> Summary for " << Name << "\t| " << Act->Name
                << "\nInputs : " << (Neurons.size() ? Neurons[0].NumWeights() : 0)
                << "\nOutputs: " << Neurons.size()
                << "\nEta    : " << Eta << "\n";
            printed = true;
        }
        if (all || printList.find("Neurons") != std::string::npos)
        {
            for (auto& n : Neurons) n.Print(out);
            printed = true;
        }
        if (!printed || all || printList.find("full") != std::string::npos)
            Layer::Print(printList, out);
        out.flush();
    }

protected:
    std::vector<Neuron> Neurons;
};

#endif