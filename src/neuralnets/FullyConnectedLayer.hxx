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

#ifdef CUDA_PROJECT
#include "Cuda_FullyConnectedLayer.cuh"
#include "utils/CudaSimpleMatrix.cuh"
#else
//#error "Not Cuda project"
#endif

#ifdef _DEBUG
#define NN_DEBUG 1
#define NN_PRINT 1
#else
#define NN_PRINT 0
#define NN_DEBUG 0
#endif

typedef std::vector<double> Row;

struct Neuron
{
    Neuron(size_t N) : Weights(N + 1), dWeights(N+1) {}

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
};


SimpleMatrix::Matrix<double> GetWeightMatrix(const std::vector<Neuron>& Neurons)
{
    SimpleMatrix::Matrix<double> ret({ Neurons[0].NumWeights() + 1 , Neurons.size() });
    for (size_t h = 0; h < Neurons.size(); h++)
        for (size_t i = 0; i < Neurons[0].NumWeights(); ++i)
            ret.data[h][i] = Neurons[h][i];
    return ret;
}

class FullyConnectedLayer : public Layer
{
public:

    FullyConnectedLayer(std::string name, size_t NumInputs, size_t NumNeurons,  std::string actName, Layer* prev = 0) :
        Layer(
            "FCLayer-" + name, 
            { NumInputs? NumInputs : prev->GetOutput().size(),1,1}, 
            { NumNeurons,1,1},
            actName,prev),
            Neurons(NumNeurons, Neuron(NumInputs ? NumInputs : prev->GetOutput().size()))
#ifdef CUDA_PROJECT
            , CudaNeurons(GetWeightMatrix(Neurons), actName)
#endif
    {
        if (Prev && NumInputs > Prev->GetOutput().size())
            std::invalid_argument("NumInputs in FCLayer is larger than output of previous layer.");
        
        for (auto& n : Neurons) n.InitWeights();
    }

    virtual void ForwardPass()
    {
        for (size_t i = 0; i < Neurons.size(); i++)
            Output[i] = Neurons[i].ForwardPass(Input.data[0][0], Act, LGrads[i]);

#ifdef CUDA_PROJECT
        CudaNeurons.Fire(Input.data[0][0]);
        auto res = CudaNeurons.Results.CompareDevToHost(Output.data[0][0]);
        if (res.first) throw std::runtime_error("device and host computation disagree");
#endif
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
#ifdef CUDA_PROJECT
    CudaNeuronBlock CudaNeurons;
#endif
};

#endif