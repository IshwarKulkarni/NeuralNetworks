#ifndef __NETWORK_INCLUDED__
#define __NETWORK_INCLUDED__

#include <numeric>
#include <fstream>
#include <vector>
#include <memory>

#include "Layer.hxx"
#include "utils/SimpleMatrix.hxx"
#include "data/DataSets.hxx"
#include "ErrorFunctions.hxx"

typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;

class Network : public std::vector<Layer*>
{
    Volume ErrFRes;
    double  EtaMultiplier;
    double  EtaDecayRate;
    size_t  SmallTestRate; // Perform Small Tests while training, after these many training sets
    size_t  SmallTestSize; // Size of such small tests.
    size_t  SmallTestNum;  // number of such tests done.

    bool WeightSanityCheck; 

    std::shared_ptr<ErrorFunctionType> ErrorFunction;

public:

    Network(std::string inFile);
    
    template<typename TrainIter>
    inline void Train(TrainIter begin, TrainIter end)
    {
        Volume In = { front()->InputSize(), nullptr };
        
        NumTrainInEpoc = std::distance(begin, end);

        auto iter = begin-1; size_t numTrain = 0;

        while(++iter != end)
        {
            In.data = iter->Input;
            
            ErrorFunction->Prime(front()->ForwardPass(In), iter->Target, ErrFRes);
            
            back()->BackwardPass(ErrFRes);
            if (SmallTestRate && numTrain % SmallTestRate == 0) SmallTest(begin, NumTrainInEpoc);

            numTrain++;
        }
        for (auto& l : *this) l->WeightDecay(DecayRate); 
    }

    template<typename TestIter>
    inline void SmallTest(TestIter begin, size_t numTrainInEpoc)
    {
        auto smallTestStart = begin + Utils::URand(numTrainInEpoc - SmallTestSize);

        Logging::Log << "Small test " << SmallTestNum++;
        Test(smallTestStart, smallTestStart + SmallTestSize);
        auto res = Results();
        Logging::Log << "\tAcc:\t" << res.first * 100 << "% Error:\t" << res.second << "\n" << Logging::Log.flush;
    }
    
    template<typename TestIter>
    inline double Test(TestIter begin, TestIter end)
    {
        NumValCorrect = 0; NumVal = 0; VldnRMSE = 0;
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        Volume In = { front()->InputSize(), nullptr };

        auto iter = begin; size_t numTest = 0;

        while (iter != end)
        {
            In.data = iter->Input;
            const auto& out = front()->ForwardPass(In);

            NumValCorrect += std::equal(out.begin(), out.end(), iter->Target, pred);
            ErrorFunction->Apply(out, iter->Target, ErrFRes);
            VldnRMSE += std::accumulate(ErrFRes.begin(), ErrFRes.end(), double(0));

            iter++;
            numTest++;
        }
        NumVal = numTest;
        return NumValCorrect / NumVal;
    }

    inline const std::pair<double, double>& GetOutputHiLo() const { return back()->GetAct()->MinMax; }

    inline void Print(std::string printList, std::ostream &out = Logging::Log)
    {
        out << "Printing network for printList: \"" << printList << "\"\n";
        if (printList.find("Network") != std::string::npos)
        {
            out << std::boolalpha
                << "\nNetowrk Description: "
                << "\nEtaMultiplier : " << EtaMultiplier
                << "\nErrorFunction : " << ErrorFunction->Name()
                << "\nEtaDecayRate  : " << EtaDecayRate
                << "\nSmallTestRate : " << SmallTestRate
                << "\nSmallTestSize : " << SmallTestSize
                << "\nEtaDecayRate  : " << EtaDecayRate
                << "\nWeightSanityCheck : " << WeightSanityCheck
                << "\n";
                
        }

        const Layer* l = front();
        do { l->Print(printList, out);  }while ((l = l->NextLayer()) != nullptr);
        out << "===================================================\n\n"; out.flush();
    }

    inline void Sanity()
    {
        size_t numLayers = size(), counter = 0;
        const Layer* l = front();
        do { counter++; } while ((l = l->NextLayer()) != nullptr);

        if (counter != numLayers)
            throw std::logic_error("All layers are not linked correctly");

        counter = 0; l = back();
        do { 
            counter++; 
            auto prev = l->PrevLayer();
            
            if (prev && l->InputSize()() > prev->Out().size())
            {
                Logging::Log << "This Layer: \n";
                l->Print("");
                Logging::Log << "Produces more outputs than following Layer: \n";
                prev->Print("");
                throw std::invalid_argument("This condition is an error");
            }
            l = prev;
        } while (l != nullptr);

        if (counter != numLayers)
            throw std::logic_error("All layers are not linked correctly");

    }

    inline std::pair<double, double> Results() {
        return std::make_pair(NumValCorrect / NumVal , VldnRMSE / NumVal);
    }

    ~Network() { for (auto& l : *this) delete l; }


private:

    unsigned NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
    double DecayRate;

};

#endif

