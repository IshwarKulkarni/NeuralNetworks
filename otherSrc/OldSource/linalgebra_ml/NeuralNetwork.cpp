#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif  // _DEBUG

#include "linalgebra_ml/ConvNeuralNet.hxx"
#include "linalgebra_ml/NeuralNet.hxx"

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include "segment.h"
#include "utils/SimpleMatrix.hpp"
#include "utils/StringUtils.hxx"
#include "DataSets.hxx"
#include "utils/utils.hxx"
#include "utils/Commandline.hxx"
#include "CIFAR-ImageReader.hxx"

using namespace std;
using namespace Utils;
using namespace Logging;
using namespace StringUtils;
using namespace SimpleMatrix;


bool HittingPlateu(vector<double>& SuccesssRates, double sr, unsigned plateuLen) {

    SuccesssRates.push_back(sr);

    if (SuccesssRates.size() < plateuLen + 1) return false;

    auto& it = SuccesssRates.rbegin();
    auto next = it; ++next;

    for (; it != SuccesssRates.rbegin() + plateuLen; ++it, ++next)
        if (fabs(*next - *it) > 0.5)
            return false;
    return true;
};

int NN(int argc, char** argv)
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    Log.ResetIntervealLogging(false);
    Log.Tee(cout);


    unsigned NumEpocs = 50;
    double stoppingPc = 98, etaMultiplier = 1., etaChangeExp = 1.5, decay = 0.85;

    NameValuePairParser nvpp(argc, argv);
    nvpp.Get("epocs",  NumEpocs);
    nvpp.Get("pcs",    stoppingPc);
    nvpp.Get("ilr",    etaMultiplier);
    nvpp.Get("lrc",    etaChangeExp);
    nvpp.Get("dec",    decay);
    
    unsigned InputSize, OutputSize;
    auto& data = LoadMnistData(InputSize, OutputSize);

#ifndef _DEBUG
    srand(int(time(0)));
    data.ShuffleAll();
#endif   
    NeuralNet network( 
        ifstream(DATA_LOCATION "MNISTTopology.txt"),
        InputSize, OutputSize,
        etaMultiplier, etaChangeExp,
        decay);
    auto& highLow = network.GetOutputHiLo();
    data.ResetHighLow(highLow.first, highLow.second);

    data.Summarize(Logging::Log, false); 
    network.Print("Summary");
    Timer timer("Classify");

    Log << setfill('0') << std::right << fixed << showpoint << left
        << "\nRunning until " << NumEpocs << " epochs or until validation test hits " 
        << setprecision(3) << stoppingPc  << "% success\n" << LogEndl;

    double prevSr = 0.f, time = timer.TimeFromLastCheck(), totTime = 0;
    vector<double> times;
    typedef std::remove_reference<decltype(data)>::type::Iter& PatternIterType;

    auto RunValidation = [&](string s, PatternIterType begin, PatternIterType end) {
        
        times.push_back(timer.TimeFromLastCheck());
        double sr = network.Validate(begin, end) * 100;
        
        totTime = std::accumulate(times.begin(), times.end(), totTime) / times.size();
        const auto& etas = network.GetEtas();
        string etaStr; for (auto& e : etas) etaStr += (to_string(e*decay) + ", ");
        
        Log << s << ",  Prediction Rate: " << setw(6) << sr
            << "%, [" << showpos << setw(4) << sr - prevSr << noshowpos << "]\t"
            << " Took [" << times.back() << "s|" << totTime << "s]"
            << " Etas: " << etaStr
            << ".\n";
        
        totTime = 0;
        prevSr = sr; 
    };    

    
    for (unsigned e = 0; e < NumEpocs && prevSr < stoppingPc; ++e)
    {
        network.RunEpoc(data.TrainBegin(), data.TrainEnd());
        RunValidation("Epoch " + std::to_string(e), data.VldnBegin(), data.VldnEnd());
    }

    RunValidation("TestSet", data.TestBegin(), data.TestEnd());
    data.Clear();

    return 0;
}

int main(int argc, char** argv)
{
	NN(argc, argv);
	_CrtDumpMemoryLeaks();
	exit(0);
}
