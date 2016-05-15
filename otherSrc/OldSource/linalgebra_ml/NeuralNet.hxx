#ifndef __NEURAL_NET__
#define __NEURAL_NET__

#include <vector>
#include <sstream>
#include "utils/utils.hxx"
#include "utils/commandline.hxx"
#include "utils/SimpleMatrix.hpp"
#include "Activation.hxx"
#include <omp.h>

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
    Neuron(unsigned N) : Weights(N + 1){}

    inline void InitWeights()
    {
        double rs = double(1 / sqrt(Weights.size()));
        for (auto& w : Weights)
            w = NN_DEBUG ? 0.1 : Utils::Rand(rs, -rs);
    }

    template<typename IpArr>
    double ForwardPass(IpArr& ip, const Activation* Act) {

        double out = Weights.back();
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            out += (Weights[i] * (ip[i]));
        return  Act->Function(out, LocalGradient);
    }

    inline double operator[](unsigned wIdx)  const { return Weights[wIdx]; }
    inline size_t NumWeights() const { return Weights.size()-1; }
    inline double GetLocalGradient() const { return LocalGradient; }

    inline void BackwardPass(double eta, double grad, const Row& inputs, Row& pgrad)
    {
        GetPGrads(grad, pgrad);
        ChangeWeights(eta, grad, inputs);
    }

    inline void ChangeWeights(double eta, double grad, const Row& inputs)
    {
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            Weights[i] -= eta * grad * inputs[i];

        Weights.back() += eta * grad;
    }

    template<typename PGArrayType>
    inline void GetPGrads(double grad, PGArrayType& pgrads)
    {
        for (unsigned i = 0; i < Weights.size() - 1; ++i)
            pgrads[i] += grad * Weights[i];
    }

    std::ostream& Print(std::ostream& stream, const Neuron& neuron)
    {
        for (unsigned i = 0; i < neuron.NumWeights(); ++i)
            stream << neuron[i] << ",\t";
    
        stream << " | " << neuron.Weights.back();
    
        return stream;
    }

private:
    Row Weights; // Nth index is Bias
    double LocalGradient;
};

struct NeuralNet;

static unsigned NumLayers = 0;
struct LayerF
{
    friend struct NeuralNet;
    LayerF(double l, unsigned NumInputs, unsigned NumNeurons, std::string name, const Activation* act, LayerF* prev = 0) :
        Eta(l),
        Name(name + "<" + std::to_string(ThisLayerNum = ++NumLayers) + ">"),
        Inputs(0), Next(0), Prev(prev),
        Neurons(NumNeurons, Neuron(NumInputs)),
        Outputs(NumNeurons),
        Grads(NumNeurons),
        ErrorResults(NumNeurons),
        Act(act)
    {
        for (auto& n : Neurons) n.InitWeights();
        if (Prev)
        {
            prev->Next = this;
            prev->ErrorResults.resize(0);
            PGrads.resize(Prev->Outputs.size());
            THROW_IF(prev->Outputs.size() != NumInputs, LogicalOrMath,
                "Previous layer output and this layer inputs don't match.");
            
            //if (Prev->Prev) for (auto& n : Prev->Neurons) n.DropEnabled = true;
        }
        else
        {
            Inputs.resize(NumInputs);
        }

        THROW_IF(!Act, InvalidArgumentException, "Activation function not set");
    }

    template <typename T>
    inline Row&  ForwardPass(T& ip)
    {
        for (unsigned i = 0; i < Inputs.size(); i++)
            Inputs[i] = double(ip[i]);

        for (int i = 0; i < int(Neurons.size()); i++)
            Outputs[i] = Neurons[i].ForwardPass(ip, Act);

        if (NN_PRINT) Print("\nOutputs");
        if (Next) return Next->ForwardPass(Outputs);
        return Outputs;
    }

    // performs differentiation wrt errF, and calls backward pass
    template<typename Op>
    inline void BackwardPassStart(Op targets)
    {
        for (unsigned i = 0; i < Neurons.size(); ++i)
            ErrorResults[i] = (Outputs[i] - targets[i]);  // minimizing (-1/2)*(y-o)^2;
        BackwardPass(ErrorResults);
    }
   
    void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        if (printList == "All") printList = "Grads, PGrads, Neurons, Outputs";

        if (StringUtils::Contains(printList, "PGrads", false))
            Utils::PrintLinear(out, PGrads, "\nPGrads for " + Name + ",\t");
        else if (StringUtils::Contains(printList, "Grads", false))
            Utils::PrintLinear(out, Grads, "\nGrads for " + Name + ",\t");

        if (StringUtils::Contains(printList, "Neurons", false))
            SimpleMatrix::Out2d(out, Neurons, Neurons[0].NumWeights(), Neurons.size(),
                ("\nNeurons for " + Name + "\n").c_str());

        if (StringUtils::Contains(printList, "Outputs", false))
            Utils::PrintLinear(out, Outputs, "\nOutputs for " + Name + ",\t");

        if (StringUtils::Contains(printList, "Summary", false))
            out << Name << "\t| " << Act->Name
            << ",\tInputs : " << (Neurons.size() ? Neurons[0].NumWeights() : 0)
            << ",\tOutputs: " << Neurons.size()
            << ",\tEta    : " << Eta << "\n";
    }

    inline void RateDecay(double decay) { Eta *= decay; }
    inline const Row& GetOutputs() const { return Outputs; }

    template <typename T>
    inline void GetPGrads(T& pgrads) const
    {
        for (unsigned i = 0; i < PGrads.size(); ++i)
            pgrads[i] = PGrads[i];
    }

    inline void MakePGrads() { PGrads.resize(Neurons[0].NumWeights()); }

private:
    double Eta;

    std::vector<Neuron> Neurons;
    Row  ErrorResults, Grads, PGrads, Outputs, Inputs; //PGrads is total gradient till now

    unsigned    ThisLayerNum;
    std::string Name;
    LayerF      *Next, *Prev;
    const Activation  *Act;

    inline void BackwardPass(const decltype(PGrads)& backError)
    {
       for (unsigned n = 0; n < Neurons.size(); ++n)
          Grads[n] =  backError[n]*Neurons[n].GetLocalGradient();// grad = NextLayerGradient*LocalGradient

        if (NN_PRINT > 1) Print("Grads");

        const auto& ip = Prev ? Prev->Outputs : Inputs;

        if (PGrads.size())
        {
            PGrads.assign(PGrads.size(), 0.0);
            for (unsigned n = 0; n < Neurons.size(); ++n)
                Neurons[n].BackwardPass(Eta, Grads[n], ip, PGrads);
        }

        if (Prev)
            Prev->BackwardPass(PGrads);
        else
            for (int j = 0; j < int(Neurons.size()); ++j)
                Neurons[j].ChangeWeights(Eta, Grads[j], ip);
        
        if (NN_PRINT > 1) Print("Neurons");
    }
};

struct ConvNetwork;

struct NeuralNet
{
    NeuralNet(std::istream& topolFile, unsigned inSize, unsigned outSize,
        double etaFactor = 0.1f, double etaChange = 1.5f, double decay = 0.85) :
        InitEtaMultiplier(etaFactor),
        EtaChange(etaChange), 
        DecayRate(decay)
    {
        THROW_IF(!topolFile, BadFileNameOrType, "File could not be opened to read topology. Quitting\n");

        auto & nvpp = NameValuePairParser(topolFile, " : ", '\0', "#");
        auto& nameSizes = nvpp.GetPairs<unsigned>();

        WARN_IF(inSize != nameSizes[0].second || outSize != nameSizes.back().second,
            DimensionException, "Topology map's input/output does not match data\
            inSize = %d,  Listed Input Size = %d, outSize = %d, Listed output = %d\n",
            inSize, nameSizes[0].second ,  outSize, nameSizes.back().second);

        nameSizes[0].second = inSize, nameSizes.back().second = outSize;

        for (unsigned i = 0; i < nameSizes.size() - 1; ++i)
        {
            auto& splits = StringUtils::Split(nameSizes[i+1].first, ",", true);
            std::string& actvtnName = (splits.size() >1 ? splits[1] : "Sigmoid"); 
            Add(nameSizes[i].second, nameSizes[i + 1].second, splits[0],GetActivationByName(actvtnName));
        }
    }

    inline NeuralNet& Add(unsigned ipSize, unsigned opSize, std::string name, const Activation* act)
    {
        int layer = Layers.size();
        double lr = InitEtaMultiplier* pow(EtaChange, -layer) * act->Eta;

        Layers.push_back(new LayerF(lr, ipSize, opSize, name, act, layer ? Layers.back() : 0));
        return *this;
    }

    inline const std::pair<double, double>& GetOutputHiLo() const { return Layers.back()->Act->MinMax; }
    inline const ResultCmpPredicate& GetResultCmpPredicate() const { return Layers.back()->Act->ResultCmpPredicate; }
    inline std::vector<double> GetEtas() const {
        std::vector<double> res;
        for (auto& l : Layers) res.push_back(l->Eta);
        return res;
    }

    template<typename PatternIter>
    inline void RunEpoc(PatternIter dataBegin, PatternIter dataEnd)
    {
        NumTrainInEpoc = 0;

        for (auto& data = dataBegin; data != dataEnd; data++, ++NumTrainInEpoc)
            Layers.front()->ForwardPass(data->Input),
            Layers.back()->BackwardPassStart(data->Target);
        
        if (NN_PRINT > 1) Print("Neurons & Outputs after an epoch", Layers.size());
        else if (NN_PRINT > 0) Print("Outputs", 1);
        

        for (auto& l : Layers) l->RateDecay(DecayRate);
    }

    template<typename PatternIter>
    inline double Validate(PatternIter dataBegin, PatternIter dataEnd)
    {
        NumValCorrect = 0; NumVal = 0, VldnRMSE = 0;
        auto& pred = GetResultCmpPredicate();
        for (auto& data = dataBegin; data != dataEnd; data++, NumVal++)
        {
            const auto& out = Layers.front()->ForwardPass(data->Input);

            NumValCorrect += std::equal(out.begin(), out.end(), data->Target, pred);

            VldnRMSE += double(Utils::GetRMSE(out.begin(), out.end(), data->Target));
        }

        VldnRMSE /= NumVal;
        return NumValCorrect / NumVal;
    }

    double GetRMSE() { return VldnRMSE; }

    void Print(std::string printList, unsigned layersFromback = -1, std::ostream& out = Logging::Log) const
    {
        out << "Print with printlist: " << printList << "\n";
        for (auto& l : Layers) if (layersFromback--) l->Print(printList);

        out << "\n=============================================\n";
    }

    ~NeuralNet()
    {
        for (auto& l : Layers) delete l;
    }

private:
    NeuralNet() :
        InitEtaMultiplier(.1f),
        EtaChange(1.5f), 
        DecayRate(1.0f){}

    friend struct ConvNetwork;
    const double InitEtaMultiplier, EtaChange, DecayRate;

    std::vector<LayerF* > Layers;
    unsigned NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
};
#endif