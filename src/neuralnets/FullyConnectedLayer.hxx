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

struct Neuron
{
    Neuron(size_t N) : Weights(N + 1) {}

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
    
    template<typename TI, typename TPG>
    inline void BackwardPass(double eta, double grad, const TI& inputs, TPG& pgrad)
    {
        GetPGrads(grad, pgrad);
        ChangeWeights(eta, grad, inputs);
    }

    template<typename TPG>
    inline void ChangeWeights(double eta, double grad, const TPG& inputs)
    {
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            Weights[i] -= eta * grad * inputs[i];

        Weights.back() += eta * grad;
    }

    template<typename TPG>
    inline void GetPGrads(double grad, TPG& pgrads)
    {
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            pgrads[i] += grad * Weights[i];
    }

    std::ostream& Print(std::ostream& stream) const
    {
        for (unsigned i = 0; i < NumWeights(); ++i)
            stream << Weights[i] << ",\t";

        stream << " | " << Weights.back() << "\n";

        return stream;
    }

private:
    Row Weights; // Nth index is Bias
};


class FullyConnectedLayer : public Layer
{
public:

    FullyConnectedLayer(std::string name, unsigned NumInputs, unsigned NumNeurons,  std::string actName, Layer* prev = 0) :
        Layer(
            "FCLayer-" + name, 
            { NumInputs? NumInputs : prev->Out().size(),1,1}, 
            { NumNeurons,1,1},
            actName,prev),
        Neurons(NumNeurons, Neuron(NumInputs ? NumInputs : prev->Out().size()))
    {
        if (Prev && NumInputs > Prev->Out().size())
            std::invalid_argument("NumInputs in FCLayer is larger than output of previous layer.");
        
        for (auto& n : Neurons) n.InitWeights();
    }

    virtual Volume& ForwardPass(Volume& input)
    {
        if (!Prev) Input = input;
        for (size_t i = 0; i < Neurons.size(); i++)
            Output[i] = Neurons[i].ForwardPass(input.data[0][0], Act, LGrads[i]);

        Print("Output");

        if (Next) return Next->ForwardPass(Output);
        return Output;
    }

    virtual void BackwardPass(Volume& backError)
    {
        for (size_t n = 0; n < Neurons.size(); ++n)
            Grads[n] = backError[n] * LGrads[n];

        if (PGrads.size())
        {
            PGrads.Fill(0.0);
            for (unsigned n = 0; n < Neurons.size(); ++n)
                Neurons[n].BackwardPass(Eta, Grads[n], Input, PGrads);
        }

        if (Prev)
            Prev->BackwardPass(PGrads);
        else
            for (size_t j = 0; j < Neurons.size(); ++j)
                Neurons[j].ChangeWeights(Eta, Grads[j], Input);
    }

    inline virtual const Vec::Size3 InputSize() const override {
        
        if (Neurons.size())
            return { Neurons[0].NumWeights(),1,1 };
        
        return 0;
    }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Summary") != std::string::npos)
        
            out
                << "\n--> Summary for " << Name
                << "\t| " << Act->Name
                << ",\tInputs : " << (Neurons.size() ? Neurons[0].NumWeights() : 0)
                << ",\tOutputs: " << Neurons.size()
                << ",\tEta    : " << Eta << "\n";
        if (all || printList.find("Neurons") != std::string::npos)
            for (auto& n : Neurons) n.Print(out);
            
        if (all || printList.find("full") != std::string::npos)
            Layer::Print(printList, out);
        out.flush();
    }

protected:
    std::vector<Neuron> Neurons;
};

#endif