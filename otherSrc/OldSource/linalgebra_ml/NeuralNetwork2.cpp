
#include "linalgebra_ml/ConvNeuralNet.hxx"
#include "linalgebra_ml/NeuralNet.hxx"

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include "segment.h"
#include "graphics/geometry/Vec23.hxx"
#include "DataSets.hxx"
#include "utils/utils.hxx"
#include "utils/Commandline.hxx"
#include<conio.h>

using namespace std;
using namespace Vec;
using namespace Utils;
using namespace Logging;
using namespace StringUtils;

typedef std::vector<Vec::Size2> Size2Vec;

int main(int argc, char** argv)
{
    Log.ResetIntervealLogging(false);
    Log.Tee(cout);
    Log << setw(7) << setprecision(5);

    srand(int(time(0)));

    unsigned NumEpocs = 50;
    float stoppingPc = 90.f;

    Size3 ipSize;
    unsigned outSize;
    //auto& data = LoadCIFAR10(ipSize, outSize,10);
    auto& data = LoadMnistData2(ipSize, outSize
#ifdef _DEBUG
       ,1000
#endif 
       );
	typedef PatternSet<float***>::Iter PatternIter;
	//data.ShuffleAll();
	data.Summarize(Log);
#if 1
    std::vector<ConvLayerDesc> descs = {
        { ipSize ,    "IpLayer", "Sigmoid", 6, Size2(3,3),    Size2(1,1) },
        { ipSize ,    "IpLayer", "TanH",    3, Size2(5,5),    Size2(2,2) },
        { Zeroes3(),  "Mid"    , "Sigmoid", 1, Size2(5,5),    Size2(3,3) },
    };
#else
    std::vector<ConvLayerDesc> descs = {
        { ipSize ,    "IpLayer", "Sigmoid", 7, Size2(3,3),    Size2(2,2) },
        { Zeroes3(),  "Mid1"   , "Sigmoid", 5,  Size2(5,5),    Size2(1,1) },
        { Zeroes3(),  "Mid2"   , "Sigmoid", 3,  Size2(5,5),    Size2(2,2) },
        { Zeroes3(),  "Out"    , "Sigmoid", 1,  Size2(3,3),    Size2(1,1) },
    };
#endif
    ConvNetwork network(outSize, descs);
  
    network.Print("Summary");

    Log << setfill('0') << std::right << fixed << showpoint << left
        << "\nRunning until " << NumEpocs << " epochs or until validation test hits "
        << setprecision(3) << stoppingPc << "% success\n" << LogEndl;

    Timer timer("Classify");
    double prevSr = 0.f, time = timer.TimeFromLastCheck(), totTime = 0;
    vector<double> times;
    typedef std::remove_reference<decltype(data)>::type::Iter& PatternIterType;


    auto RunValidation = [&](string s, PatternIterType begin, PatternIterType end) {

        times.push_back(timer.TimeFromLastCheck());
        double sr = network.Validate(begin, end) * 100;

        totTime = std::accumulate(times.begin(), times.end(), totTime) / times.size();

        Log << s << ",  Prediction Rate: " << setw(6) << sr
            << "%, [" << showpos << setw(4) << sr - prevSr << noshowpos << "]\t"
            << ", Took " << times.back() << "s | " << totTime << "s.\n";

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

    getch();
    return 0;
}
