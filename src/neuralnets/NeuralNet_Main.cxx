#include "Network.hxx"
#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"

using namespace std;
using namespace Logging;

void ManualInitNetwork(Network& nn)
{
    Vec::Size3 MNISTDataSize = { MNISTReader::ImW, MNISTReader::ImH, 1 };
#if 1
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("Input",    "Sigmoid", MNISTDataSize, { 5, 5 }, { 1,1 }, 4), nullptr));
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("Middle",   "Sigmoid", Vec::Zeroes3, { 5, 5 }, { 2,2 }, 2), nn.back()));
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("LastFC",   "Sigmoid", Vec::Zeroes3, { 5, 5 }, { 3,3 }, 1), nn.back()));

    nn.push_back(new FullyConnectedLayer("FCIn", 0, 25, "Sigmoid", nn.back()));
#else

    nn.push_back(new FullyConnectedLayer("FCIn", MNISTDataSize(), 100, "Sigmoid", nullptr));
    nn.push_back(new FullyConnectedLayer("FCIn", 50, 25, "Sigmoid", nn.back()));
#endif 

    nn.push_back(new FullyConnectedLayer("Out",  25, 10, "Sigmoid", nn.back()));
}

int main()
{
    //Network nn(std::ifstream("MNIST_Network.config"));
    Network nn(10); ManualInitNetwork(nn);
    
    Vec::Size3 in;; unsigned out;
    auto data = LoadMnistData2(in, out);
    auto& highLow = nn.GetOutputHiLo();
    data.ResetHighLow(highLow.first, highLow.second);

    data.Summarize(Log, false);
    nn.Print("Summary");

    Log << "\n Starting trainging... " << endl;

    Timer  epochTime("ClassifierTime");

    unsigned maxEpochs = 20;
    double   targetAcc = 0.95, acc = 0.;
    for (size_t i = 0; i < maxEpochs && acc < targetAcc; i++)
    {
        epochTime.TimeFromLastCheck();
        nn.Train(data.TrainBegin(), data.TrainEnd());
        double lastCheck = epochTime.TimeFromLastCheck();
        Log << "Train Epoch " << i << ">  "
            << "[" <<  lastCheck<< "s]:\t"
            << "Validation Accuracy : " << (acc = nn.Test(data.VldnBegin(), data.VldnEnd()) ) * 100 << "%"
            << endl;
    }
    

    return 0;
}
