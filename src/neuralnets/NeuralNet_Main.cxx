#include "Network.hxx"
#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"

using namespace std;
using namespace Logging;

int main()
{
    try
    {
        Vec::Size3 in;; unsigned out;
        cout << "Reading data..." << endl;
        auto data = LoadMnistData2(in, out);
        data.Summarize(Log, false);

        cout << "Building network..." << endl;
        Network nn(DATA_LOCATION "MNIST_Network.config");
        //Network nn(DATA_LOCATION "comparetcnn1.txt");
        auto& highLow = nn.GetOutputHiLo();
        data.ResetHighLow(highLow.first, highLow.second);

        cout << endl << "Starting training... " << endl;

        Timer  epochTime("ClassifierTime");

        unsigned maxEpochs = 20;
        double   targetAcc = 0.95, acc = 0.;
        for (size_t i = 0; i < maxEpochs && acc < targetAcc; i++)
        {
            epochTime.TimeFromLastCheck();
            nn.Train(data.TrainBegin(), data.TrainEnd());
            double lastCheck = epochTime.TimeFromLastCheck();
            cout << "Train Epoch " << i << ">  "
                << "[" << lastCheck << "s]:\t"
                << "Validation Accuracy : " << (acc = nn.Test(data.VldnBegin(), data.VldnEnd())) * 100 << "%"
                << endl;
        }

    }
    catch (std::exception e)
    {
        std::cerr << e.what() << endl;
        throw e;
    }

    return 0;
}
   
