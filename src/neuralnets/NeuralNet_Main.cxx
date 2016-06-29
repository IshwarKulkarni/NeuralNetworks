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
        DataLoc(DATA_LOCATION), ConfigFile("CIFAR_Network.config"),
        VldnFrac(0.05), TestFrac(0.05), NumSamples(-1), MaxEpocs(4), TargetAcc(98),
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

void statusMonitor(const Network* nn, bool& monitor)
{
    size_t checkNum = 0;
    static const string wheel = { char(176), char(177), char(178), char(219) , char(178),char(177) };
    while (monitor)
    {
        auto stat = nn->GetCurrentTrainStatus();

        if (!(stat && stat->SamplesDone)) { std::this_thread::yield(); continue; }

        auto nextCheck = system_clock::now() + milliseconds(400);

        auto done = (100. * stat->SamplesDone) / stat->NumTrainInEpoc,
            passRate = (100. * stat->LastPasses.count()) / stat->PassWinSize,
            cumPassRate = double(stat->TotNumPasses) / double(stat->SamplesDone),
            timeSpent = Utils::TimeSince(stat->TrainStart),
            imRate = double(stat->SamplesDone) / timeSpent;

        cout
            << setw(4) << setprecision(4)
            << "\rEpoch " << stat->EpochNum << "> [" << timeSpent << "s]"
            //<< wheel[(checkNum++) % wheel.length()]
            << " Complete : " << done
            << "%\tPass<" << stat->PassWinSize << ">: " << passRate
            << "%\tTotal Pass: " << cumPassRate * 100
            << "%\tRate : " << imRate << " smpls/s.         ";

        //Log << stat->EpochNum << Utils::TimeSince(stat->TrainStart) 
        //    << '\t' <<  done << '\t' << passRate << '\t' << cumPassRate << '\n';

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

    auto data = LoadCifarData10(in, out, nn.GetOutputHiLo(), rParam.NumSamples);
    data.Summarize(Log, false);

    cout << "\nTraining... " << endl;
    Timer  epochTime("ClassifierTime");

    auto monitor = true;
    std::thread statMonitorThread(statusMonitor, &nn, std::ref(monitor));

    for (size_t epoch = 0; epoch < rParam.MaxEpocs; ++epoch)
    {
        epochTime.TimeFromLastCheck();
        nn.Train(data.TrainBegin(), data.TrainEnd());
        double lastCheck = epochTime.TimeFromLastCheck();

        nn.Test(data.VldnBegin(), data.VldnEnd());

        cout << "\rEpoch " << epoch << "> [" << lastCheck << "s]" 
            << ":\t(Accuracy, RMSE): " << nn.Results() << string(80, ' ') << "\n";

        Log << "\rEpoch " << epoch << "> [" << lastCheck << "s]"
            << ":\t(Accuracy, RMSE): " << nn.Results() << "\n";

        if (nn.Results().first * 100 > rParam.TargetAcc)  break;
    }
    
    monitor = false;
    statMonitorThread.join();

    if (rParam.RunTest)
    {
        Utils::TopN<Network::TestNumErr>* topNFails = nullptr;
        if (rParam.TopNFailIms)
            topNFails = new Utils::TopN<Network::TestNumErr>(rParam.TopNFailIms);

        cout << "\nRunning test.. ";
        nn.Test(data.TestBegin(), data.TestEnd(), topNFails);

        Log <<  "\nTest (Accuracy, RMSE): " << nn.Results() << '\n';
        cout << "\nTest (Accuracy, RMSE): " << nn.Results() << string(80, ' ') << '\n';

        if (topNFails)
        {
            size_t i = 0;
            cout << "\nWriting top " << topNFails->size() << " fails as images.. ";
            for (auto& f : *topNFails)
                PPMIO::Write("Fail_" + to_string(i++), MNISTReader::ImW, MNISTReader::ImH, 1,
                    (data.TestBegin() + f.TestOffset)->Input[0][0]);

            cout << "Done\n";
        }
    }

    data.Clear();
    cin.get();
    return 0;
}
