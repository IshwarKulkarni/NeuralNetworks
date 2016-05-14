#ifndef __NETWORK_INCLUDED__
#define __NETWORK_INCLUDED__

#include <fstream>
#include <vector>

#include "Layer.hxx"
#include "utils/SimpleMatrix.hxx"
#include "data/DataSets.hxx"

typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;


class Network : public std::vector<Layer*>
{
    Volume ErrFRes;


public:

    // only implementing MSE error function; d/do ( -1/2 * (t-o)^2 ) ) = o-t
    inline Volume& GetErrFRes(const Volume& out, const double* target) 
    {
        for (size_t i = 0; i < out.size(); ++i)
            ErrFRes[i] = out[i] - target[i];

        return ErrFRes;
    }

    Network(std::string inFile);
    
    Network(unsigned outSize) {
        ErrFRes = Volume({ outSize, 1,1 });
    }

    template<typename TrainIter>
    inline void Train(TrainIter begin, TrainIter end)
    {
        Volume In = { front()->InputSize(), nullptr };
        NumTrainInEpoc = 0;
        while( begin != end && ++NumTrainInEpoc && begin->Input) 
        {
            In.data = begin->Input;
            
            GetErrFRes(front()->ForwardPass(In), begin->Target);
            
            back()->BackwardPass(ErrFRes);

            begin++;
            
            if (NumTrainInEpoc % 100 == 0)
                for (auto& l : *this) l->WeightSanity();
        }
        for (auto& l : *this) l->WeightDecay(DecayRate);
    }


    template<typename TestIter>
    inline double Test(TestIter begin, TestIter end)
    {
        NumValCorrect = 0; NumVal = 0; VldnRMSE = 0;
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        Volume In = { front()->InputSize(), nullptr };
        while (begin++ != end &&  ++NumVal)
        {
            In.data = begin->Input;
            const auto& out = front()->ForwardPass(In);

            NumValCorrect += std::equal(out.begin(), out.end(), begin->Target, pred);
        }
        return NumValCorrect / NumVal;
    }

    inline const std::pair<double, double>& GetOutputHiLo() const { return back()->GetAct()->MinMax; }

    inline void Print(std::string printList, std::ostream &out = Logging::Log)
    {
        out << "Printing network for printList: \"" << printList << "\"\n";
        const Layer* l = front();
        do { l->Print(printList, out);  }while ((l = l->NextLayer()) != nullptr);
        out << "===================================================\n"; out.flush();
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
        return std::make_pair( 0,0 );
    }

    ~Network(){ for (auto& l : *this) delete l; }


private:

    unsigned NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
    const double DecayRate = 0.95;

};

#endif

