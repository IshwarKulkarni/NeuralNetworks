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

// Network holds pointers to a nodes in a doubly linked list of layers.
class Network : public std::vector<Layer*>
{
    Volume ErrFRes;
    double  EtaMultiplier;
    double  EtaDecayRate;
    size_t  SmallTestRate; // Perform Small Tests while training, after these many training samples
    size_t  SmallTestSize; // Size of such small tests.
    size_t  SmallTestNum;  // number of such tests done.

    bool WeightSanityCheck;

    std::shared_ptr<ErrorFunctionType> ErrorFunction;

public:

    Network(std::string inFile);

    template<typename TrainIter>
    inline void Train(TrainIter begin, TrainIter end)
    {
        NumTrainInEpoc = std::distance(begin, end);

        size_t numTrain = 0;
        Layer *f = front(), *b = back();
        for (auto iter = begin; iter != end; ++iter, ++numTrain)
        {
            iter->GetInput(f->GetInput());
            f->ForwardPass();

            ErrorFunction->Prime(b->GetOutput(), iter->Target, ErrFRes);

            b->BackwardPass(ErrFRes);

#ifdef VALGRIND
            if (numTrain > 2) break;
#endif
            if (numTrain && SmallTestRate && numTrain % SmallTestRate == 0)
                SmallTest(begin, NumTrainInEpoc);
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
        Logging::Log << "\tAcc:\t" << res.x * 100 << "% Error:\t" << res.y << "\n" << Logging::Log.flush;
    }

    template<typename TestIter>
    inline double Test(TestIter begin, TestIter end)
    {
        NumValCorrect = 0; NumVal = 0; VldnRMSE = 0;
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        size_t numTest = 0;
        Layer *f = front(), *b = back();
        for (auto iter = begin; iter != end; ++iter, ++numTest)
        {
            iter->GetInput(f->GetInput());
            f->ForwardPass();

            auto& out = b->GetOutput();
            NumValCorrect += std::equal(out.begin(), out.end(), iter->Target, pred);
            ErrorFunction->Apply(out, iter->Target, ErrFRes);
            VldnRMSE += std::accumulate(ErrFRes.begin(), ErrFRes.end(), double(0));

#ifdef VALGRIND
            if (numTest > 2) break;
#endif
        }

        NumVal = numTest;

        return NumValCorrect / NumVal;
    }

    inline const Vec::Vec2<double>& GetOutputHiLo() const { return back()->GetAct()->MinMax; }

    inline void Print(std::string printList, std::ostream &out = Logging::Log)
    {
        out << "\nPrinting network for printList: \"" << printList << "\"\n";
        if (printList.find("Network") != std::string::npos)
        {
            out << std::boolalpha
                << "\nConfig Source : " << this->ConfigSource
                << "\nNetowrk Description: "
                << "\nEtaMultiplier : " << EtaMultiplier
                << "\nErrorFunction : " << ErrorFunction->Name()
                << "\nSmallTestRate : " << SmallTestRate
                << "\nSmallTestSize : " << SmallTestSize
                << "\nEtaDecayRate  : " << EtaDecayRate
                << "\nWeightSanityCheck : " << WeightSanityCheck
                << "\n";

        }

        Layer* l = front();
        do { l->Print(printList, out);  } while ((l = l->NextLayer()) != nullptr);
        out << "===================================================\n\n"; out.flush();
    }

    inline void Sanity()
    {
        size_t numLayers = size(), counter = 0;
        Layer* l = front();
        do { counter++; } while ((l = l->NextLayer()) != nullptr);

        if (counter != numLayers)
            throw std::logic_error("All layers are not linked correctly");

        counter = 0; l = back();
        do {
            counter++;
            auto prev = l->PrevLayer();

            if (prev && l->GetInput().size() > prev->GetOutput().size())
            {
                Logging::Log << "This Layer: \n";
                l->Print("Summary");
                Logging::Log << "Produces more outputs than following Layer: \n";
                prev->Print("Summary");
                throw std::invalid_argument("This condition is an error"); // wow! what a helpful message
            }
            l = prev;
        } while (l != nullptr);

        if (counter != numLayers)
            throw std::logic_error("All layers are not linked correctly");

    }

    inline Vec::Vec2<double> Results() {
        return{ NumValCorrect / NumVal , VldnRMSE / NumVal };
    }

    ~Network() {
        for (auto& l : *this) delete l;
        ErrFRes.Clear();
    }


private:

    size_t NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
    double DecayRate;
    std::string ConfigSource;

};

#endif

