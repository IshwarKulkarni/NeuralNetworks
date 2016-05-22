#include "Network.hxx"
#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"

using namespace std;
using namespace Logging;
using namespace SimpleMatrix;

int main()
{

    //try
    //{
        cout << "Building network..." ;
        Network nn(DATA_LOCATION "MNIST_Network.config");
        cout << " done!" << endl;
        Vec::Size3 in;; unsigned out;
        cout << "\nReading data...";
        
        auto data = LoadMnistData2(in, out, nn.GetOutputHiLo());
        cout << " done!" << endl;
        data.Summarize(Log, false);

        cout << endl << "Starting training... " << endl;

        Timer  epochTime("ClassifierTime");

        unsigned maxEpochs = 20;
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
        }
    //}
    //catch (std::exception e)
    //{
    //    std::cerr << e.what() << endl;
    //    throw e;
    //}

    return 0;
}
   
