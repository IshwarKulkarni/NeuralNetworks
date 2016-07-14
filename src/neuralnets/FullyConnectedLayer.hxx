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
#include "CudaLayers.cuh"
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

typedef std::vector<float_t> Row;

struct Neuron
{
    Neuron(size_t N) : Weights(N + 1), dWeights(N+1) {}

    inline void InitWeights()
    {
        float_t rs = float_t(1 / sqrt(Weights.size()));
        for (auto& w : Weights)
            w = NN_DEBUG ? float_t(0.1) : Utils::URand(rs, -rs);
    }

    template<typename IpArr>
    float_t ForwardPass(IpArr& ip, const Activation* Act, float_t& localGradient) {
        float_t out = Weights.back();
        for (size_t i = 0; i < Weights.size() - 1; ++i)
            out += (Weights[i] * (ip[i]));
        return  Act->Function(out, localGradient);
    }

    inline float_t operator[](unsigned wIdx)  const { return Weights[wIdx]; }
    inline size_t NumWeights() const { return Weights.size() - 1; }
    
    inline void BackwardPass(float_t eta, float_t grad, Volume& pgrads, const Volume& inputs)
    {
        for (size_t i = 0; i < Weights.size() - 1; ++i)
            pgrads[i] += Weights[i] * grad;

        ChangeWeights(eta, grad, inputs);
    }

    void ChangeWeights(float_t eta, float_t grad, const Volume& inputs)
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
#ifdef CUDA_PROJECT
   void Compare(float_t* dev) {
       if (!CudaUtils::DevHostCmp(Weights.begin(), Weights.end(), dev).first)
           throw std::runtime_error("Comp failed");
   }

   template <typename OutT>
   OutT* Copy(OutT* dev)
   {
	   return std::copy(Weights.begin(), Weights.end(), dev);
   }
#endif
private:
    Row Weights,  dWeights; // Nth index is Bias
};


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
			, CudaNeurons( NumInputs, NumNeurons , Act->Id)
#endif
    {
        if (Prev && NumInputs > Prev->GetOutput().size())
            std::invalid_argument("NumInputs in FCLayer is larger than output of previous layer.");
        
        for (auto& n : Neurons) n.InitWeights();
#ifdef CUDA_PROJECT
		float_t* dev = CudaNeurons.Weights.devData;
		for (auto& n : Neurons) dev = n.Copy(dev);
#endif
    }

    virtual void ForwardPass()
    {
        for (size_t i = 0; i < Neurons.size(); i++)
            Output[i] = Neurons[i].ForwardPass(Input.data[0][0], Act, LGrads[i]);

#ifdef CUDA_PROJECT
		CudaNeurons.ForwardPass(Input.data[0][0]).CompareTo(Output.begin(), Output.end(), "FCOp");
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
#ifdef CUDA_PROJECT
            CudaNeurons.BackwardPass(backError.data[0][0], Eta).CompareTo(PGrads.begin(), PGrads.end(), "\nPGrads");
            for (size_t i = 0; i < Neurons.size(); ++i)
                Neurons[i].Compare(CudaNeurons.Weights.devData + i *CudaNeurons.Weights.size.x);
#endif
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
    CudaFullyConnectedLayer CudaNeurons;
#endif
};

#endif