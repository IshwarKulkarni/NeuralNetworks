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
#include <bitset>

#include "Layer.hxx"
#include "utils/SimpleMatrix.hxx"
#include "data/DataSets.hxx"
#include "ErrorFunctions.hxx"


typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;

enum NetworkStatus
{
    Building,
    Training,
    Testing,
    None
};


// Network holds pointers to a nodes in a doubly linked list of layers.
class Network : public std::vector<Layer*> // yeah, thou shalt not inherit from STL containers! Sue me!
{
    Volume ErrFRes;
    double  EtaMultiplier;
    double  EtaDecayRate;

    bool WeightSanityCheck;
    
    std::shared_ptr<ErrorFunctionType> ErrorFunction;

    Layer *f, *b;

    size_t NumVal;
    double NumValCorrect, VldnRMSE;
    std::string ConfigSource;

    void push_back(Layer* in)
    {
        if (!in) throw std::runtime_error("Pushed null Layer\n");
        if (!f) f = in;
        b = in;
        std::vector<Layer*>::push_back(in);
        if (in) b->Print("Summary");
    }

public: // exposed types :

    struct TestNumErr
    {
        double Err;
        size_t TestOffset;   
        operator double() const { return Err; }
    };

    struct TrainEpocStatus
    {
        std::chrono::high_resolution_clock::time_point  TrainStart;
        size_t NumTrainInEpoc;
        size_t SamplesDone;
        size_t TotNumPasses;
        size_t EpochNum;
        static const size_t PassWinSize = 100;
        std::bitset<PassWinSize> LastPasses;
        TrainEpocStatus(size_t numTrains, size_t e) : 
            TrainStart(std::chrono::high_resolution_clock::now()),
            NumTrainInEpoc(numTrains), SamplesDone(0), TotNumPasses(0), EpochNum(e) {}
    };

private:
    
    std::vector<TrainEpocStatus*> TrainEpocStatuses;

    NetworkStatus CurrentStatus;

public: 

    NetworkStatus GetCurretnStatus() const { return CurrentStatus; }
    
    TrainEpocStatus* GetCurrentTrainStatus() const { 
        return (TrainEpocStatuses.size() && (CurrentStatus == Training)) ? TrainEpocStatuses.back() : nullptr;
    }
        
    Network(std::string inFile);

    template<typename TrainIter>
    inline void Train(TrainIter begin, TrainIter end)
    {
        CurrentStatus = Training;
        TrainEpocStatus* stat;
        TrainEpocStatuses.push_back(stat = new TrainEpocStatus(std::distance(begin, end), TrainEpocStatuses.size()));
        
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        for (auto iter = begin; iter != end; ++iter, stat->SamplesDone++)
        {
            iter->GetInput(f->GetInput());
            f->ForwardPass();

            auto& out = b->GetOutput();
            ErrorFunction->Prime(out, iter->Target, ErrFRes);

            bool pass = std::equal(out.begin(), out.end(), iter->Target, pred);

            stat->TotNumPasses += pass;
            stat->LastPasses[stat->SamplesDone % stat->PassWinSize] = pass;
            b->BackwardPass(ErrFRes);
            Sanity();
        }
        for (auto& l : *this) l->WeightDecay(EtaDecayRate);

        CurrentStatus = None;
    }


    template<typename TestIter>
    inline double Test(TestIter begin, TestIter end, Utils::TopN<TestNumErr>* topNFails = nullptr)
    {
        CurrentStatus = Testing;

        if (topNFails) topNFails->clear();
        NumValCorrect = 0; NumVal = 0; VldnRMSE = 0;
        auto& pred = back()->GetAct()->ResultCmpPredicate;
        size_t numTest = 0;

        for (auto iter = begin; iter != end; ++iter, ++numTest)
        {
            iter->GetInput(f->GetInput());
            f->ForwardPass();

            auto& out = b->GetOutput();
            NumValCorrect += std::equal(out.begin(), out.end(), iter->Target, pred);
            ErrorFunction->Apply(out, iter->Target, ErrFRes);
            double thisRmse = std::accumulate(ErrFRes.begin(), ErrFRes.end(), double(0));
            VldnRMSE += thisRmse;
            
            if (topNFails) topNFails->insert({ thisRmse, numTest });
        }

        NumVal = numTest;
        
        CurrentStatus = None;

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
                << "\nEtaDecayRate  : " << EtaDecayRate
                << "\nWeightSanityCheck : " << WeightSanityCheck
                << "\n";

        }
        out.flush();
        if (!size()) return;
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
        for (auto& s : TrainEpocStatuses) delete s;
    }
};

#endif

