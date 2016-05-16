#include "Network.hxx"
#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"

using namespace std;
using namespace Logging;
using namespace SimpleMatrix;

int main()
{
    try
    {
        Vec::Size3 in;; unsigned out;
        cout << "Reading data..." << endl;
        
        cout << "Building network..." << endl;
        Network nn(DATA_LOCATION "CIFAR_Network.config");
        
        auto data = LoadCifarData10(in, out, nn.GetOutputHiLo());
        data.Summarize(Log, false);

        cout << endl << "Starting training... " << endl;

        Timer  epochTime("ClassifierTime");

        unsigned maxEpochs = 20;
        double   targetAcc = 0.95, acc = 0.;
        
        auto char255ToDouble = [&](unsigned char*** in, Matrix3<double> out){
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = double(in[0][0][i]) / 255;
        };


        for (size_t i = 0; i < maxEpochs && acc < targetAcc; i++)
        {
            epochTime.TimeFromLastCheck();
            
            nn.Train(data.TrainBegin(), data.TrainEnd() , char255ToDouble );
            double lastCheck = epochTime.TimeFromLastCheck();
            
            nn.Test(data.VldnBegin(), data.VldnEnd(), char255ToDouble);
            auto res = nn.Results(); acc = res.x;

            cout << "Train Epoch " << i << "> [" << lastCheck << "s]:\tAccuracy: "
                 << res.x * 100 << "%,\t Mean Error: " << res.y << endl;
        }

    }
    catch (std::exception e)
    {
        std::cerr << e.what() << endl;
        throw e;
    }

    return 0;
}
   
