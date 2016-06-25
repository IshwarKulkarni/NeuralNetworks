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

#include <atomic>
#include <thread>
#include <chrono>

#include "Network.hxx"
#include "utils/CommandLine.hxx"

using namespace std;
using namespace Logging;
using namespace SimpleMatrix;
using namespace std::chrono;

struct NeuralNetRunParams_t
{
    string DataLoc, ConfigFile;
    double  VldnFrac, TestFrac;

    unsigned NumSamples, MaxEpocs, TargetAcc, RunTest, TopNFailIms;

    NeuralNetRunParams_t() :
        DataLoc(DATA_LOCATION), ConfigFile("MNIST_LeNet-5.config"),
        VldnFrac(0.05), TestFrac(0.05),NumSamples(-1),   MaxEpocs(2), TargetAcc(98),
        RunTest(1), TopNFailIms(10)
    {
        GLOBAL_OF_TYPE_WNAME(NumSamples);
        GLOBAL_OF_TYPE_WNAME(DataLoc);
        GLOBAL_OF_TYPE_WNAME(ConfigFile);
        GLOBAL_OF_TYPE_WNAME(VldnFrac);
        GLOBAL_OF_TYPE_WNAME(TestFrac);
        GLOBAL_OF_TYPE_WNAME(MaxEpocs);
        GLOBAL_OF_TYPE_WNAME(TargetAcc);
        GLOBAL_OF_TYPE_WNAME(RunTest);
        GLOBAL_OF_TYPE_WNAME(TopNFailIms);
    }
};

void statusMonitor(const Network* nn, std::atomic<size_t>& currentEpoch, bool& monitor)
{
    string wheel("-\\|/-\\|/"); size_t checkNum = 0;
    
    while (monitor)
    {
        auto stat = nn->GetCurrentTrainStatus();

        if (!stat || !stat->NumTrainDone) { std::this_thread::yield(); continue; }

        if (stat->NumEpoc != currentEpoch) continue;

        auto nextCheck = system_clock::now() + milliseconds(1500);

        auto done = (100. * stat->NumTrainDone) / stat->NumTrainInEpoc,
            passRate = (100. * stat->LastPasses.count()) / stat->PassWinSize,
            //cumPassRate = double(stat->NumPasses) / double(stat->NumTrainDone),
            imRate = double(stat->NumTrainDone) / Utils::TimeSince(stat->TrainStart);

        cout 
            << setw(4) << setprecision(4) 
            << "\rEpoch " << currentEpoch << "> "
            << wheel[(checkNum++) % wheel.length()]  << " "
            << "Complete : " << done <<  "% "
            << "\tPass : " << passRate << "% "
            << "\tRate : " << imRate << " smpls/s.         ";

        std::this_thread::sleep_until(nextCheck);
    }
    cout << "\r";
}

int main(int argc, char** argv)
{
    NameValuePairParser::MakeGlobalNVPP(argc, argv);
    NeuralNetRunParams_t rParam;

    cout << "\nBuilding network & data... ";
    Network nn(rParam.DataLoc + rParam.ConfigFile);
    
    SetVldnTestFractions(rParam.VldnFrac, rParam.TestFrac);    
    Vec::Size3 in; unsigned out;
    auto data = LoadMnistData2(in, out, nn.GetOutputHiLo(), rParam.NumSamples);
    data.Summarize(Log, false);

    cout << "\nTraining... " << endl;
    Timer  epochTime("ClassifierTime");
    
    std::atomic<size_t> currentEpoch = ATOMIC_FLAG_INIT; bool monitor = true;
    std::thread statMonitorThread(statusMonitor, &nn, std::ref(currentEpoch), std::ref(monitor));    

    while( currentEpoch < rParam.MaxEpocs)
    {
        epochTime.TimeFromLastCheck();

        nn.Train(data.TrainBegin(), data.TrainEnd());
        double lastCheck = epochTime.TimeFromLastCheck();

        currentEpoch++; // stops the monitor from printing

        nn.Test(data.VldnBegin(), data.VldnEnd());
        auto res = nn.Results();

        cout << "\rEpoch " << currentEpoch << "> ["
            << lastCheck << "s]:\tAccuracy: "  << res.first * 100  << "%,"
            << "\t Mean Error: " << res.second << "\n";

        if (res.first * 100 > rParam.TargetAcc )  break;
    }

    if (rParam.RunTest)
    {
        Utils::TopN<Network::TestNumErr>* topNFails = nullptr;
        if (rParam.TopNFailIms)
            topNFails = new Utils::TopN<Network::TestNumErr>(rParam.TopNFailIms);

        Log << "\nRunning test.. ";
        double acc = nn.Test(data.TestBegin(), data.TestEnd(), topNFails);
        Log << "Accuracy: " << acc * 100 << "%\n";

        if (topNFails)
        {
            size_t i = 0;
            cout << "\nWriting top " << topNFails->size() << " fails as images\n";
            for (auto& f : *topNFails)
                PPMIO::Write("Fail_" + to_string(i++), MNISTReader::ImW, MNISTReader::ImH, 1,
                (data.TestBegin() + f.TestOffset)->Input[0][0]);

            cout << "Done";
        }
    }
    monitor = false;
    statMonitorThread.join();

    data.Clear();
    cin.get();
    return 0;
}
