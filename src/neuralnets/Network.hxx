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
    
    template<typename TrainIter,typename InputTxType>
    inline void Train(TrainIter begin, TrainIter end, InputTxType& inputTx)
    {
        Volume In = front()->InputSize(); // implicit
        
        NumTrainInEpoc = std::distance(begin, end);

        size_t numTrain = 0;

        for (auto iter = begin; iter != end; ++iter, ++numTrain)
        {
            inputTx(iter->Input, In);
            
            ErrorFunction->Prime(front()->ForwardPass(In), iter->Target, ErrFRes);
            
            back()->BackwardPass(ErrFRes);

            if (SmallTestRate && numTrain % SmallTestRate == 0) 
                SmallTest(begin, NumTrainInEpoc, inputTx);

        }
        for (auto& l : *this) l->WeightDecay(DecayRate); 
    }

    template<typename TestIter,typename InputTxType>
    inline void SmallTest(TestIter begin, size_t numTrainInEpoc, InputTxType& inputTx)
    {
        auto smallTestStart = begin + Utils::URand(numTrainInEpoc - SmallTestSize);

        Logging::Log << "Small test " << SmallTestNum++;
        Test(smallTestStart, smallTestStart + SmallTestSize,inputTx);
        auto res = Results();
        Logging::Log << "\tAcc:\t" << res.x * 100 << "% Error:\t" << res.y << "\n" << Logging::Log.flush;
    }
    
    template<typename TestIter, typename InputTxType>
    inline double Test(TestIter begin, TestIter end, InputTxType& inputTx)
    {
        NumValCorrect = 0; NumVal = 0; VldnRMSE = 0;
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        Volume In = { front()->InputSize(), nullptr };

        size_t numTest = 0;

        for (auto iter = begin; iter != end; ++iter, ++numTest)
        {
            inputTx(iter->Input, In);
            const auto& out = front()->ForwardPass(In);

            NumValCorrect += std::equal(out.begin(), out.end(), iter->Target, pred);
            ErrorFunction->Apply(out, iter->Target, ErrFRes);
            VldnRMSE += std::accumulate(ErrFRes.begin(), ErrFRes.end(), double(0));
        }
        NumVal = numTest;
        return NumValCorrect / NumVal;
    }

    inline const Vec::Vec2<double>& GetOutputHiLo() const { return back()->GetAct()->MinMax; }

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

    inline Vec::Vec2<double> Results() {
        return{ NumValCorrect / NumVal , VldnRMSE / NumVal };
    }

    ~Network() { for (auto& l : *this) delete l; }


private:

    unsigned NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
    double DecayRate;

};

#endif

