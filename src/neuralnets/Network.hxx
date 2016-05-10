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
    Volume ErrFPrime;
public:

    // only implementing MSE error function; d/do ( -1/2 * (t-o)^2 ) ) = o-t
    inline Volume& GetErrFPrime(const Volume& out, const double* target) 
    {
        for (size_t i = 0; i < out.size(); ++i)
            ErrFPrime[i] = out[i] - target[i];

        return ErrFPrime;
    }

    Network(std::ifstream& inFile);
    Network(unsigned outSize) {
        ErrFPrime = Volume({ outSize, 1,1 });
    }

    template<typename TrainIter>
    inline void Train(TrainIter begin, TrainIter end)
    {
        Volume In = { front()->InputSize(), nullptr };
        NumTrainInEpoc = 0;
        while( begin++ != end && ++NumTrainInEpoc && begin->Input) 
        {
            In.data = begin->Input;
            back()->BackwardPass(GetErrFPrime(front()->ForwardPass(In), begin->Target));
        }
        for (auto& l : *this) l->WeightDecay(DecayRate);
    }


    template<typename TestIter>
    inline double Test(TestIter begin, TestIter end)
    {
        NumValCorrect = 0; NumVal = 0;
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

    ~Network(){ for (auto& l : *this) delete l; }

private:

    unsigned NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
    const double DecayRate = 0.95;

};

#endif

