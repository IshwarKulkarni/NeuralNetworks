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

#include "Network.hxx"
#include "utils\CommandLine.hxx"

using namespace std;
using namespace Logging;
using namespace SimpleMatrix;

int main(int argc, char** argv)
{
    //try
    //{

    cout.rdbuf(cerr.rdbuf());

    string dataLoc(DATA_LOCATION), configFile("MNIST_LeNet-5.config");

    NameValuePairParser nvpp(argc, argv);
    nvpp.Get("DataLocation", dataLoc);
    nvpp.Get("ConfigFile", configFile);

    cout << "Building network ...";
    Network nn(dataLoc + configFile);
    Vec::Size3 in;; unsigned out;
    cout << " Done \nReading data...";

    auto data = LoadMnistData2(in, out, nn.GetOutputHiLo());
    data.Summarize(Log, false);

    cout << endl << "Done \nStarting training... " << endl;

    Timer  epochTime("ClassifierTime");

    unsigned maxEpochs = 2;
    double   targetAcc = 0.95, acc = 0.;

    for (size_t i = 0; i < maxEpochs && acc < targetAcc; i++)
    {
        epochTime.TimeFromLastCheck();

        nn.Train(data.TrainBegin(), data.TrainEnd());
        double lastCheck = epochTime.TimeFromLastCheck();

        nn.Test(data.VldnBegin(), data.VldnEnd());
        auto res = nn.Results(); acc = res.x;

        cout << "Train Epoch " << i << "> [" << lastCheck << "s]:\tAccuracy: "
            << res.first * 100 << "%,\t Mean Error: " << res.second << endl;
        data.ShuffleTrnVldn();
    }

    cout << "Running test.. ";
    acc = nn.Test(data.TrainBegin(), data.TrainEnd());
    cout << "Accuracy: " << acc*100 << "%" << endl;



    //}
    //catch (std::exception e)
    //{
    //    std::cout << e.what() << endl;
    //    throw e;
    //}

    data.Clear();
    return 0;
}

