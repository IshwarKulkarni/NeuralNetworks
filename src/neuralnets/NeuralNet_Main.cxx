#include "Network.hxx"
#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"

using namespace std;
using namespace Logging;

void ManualInitNetwork(Network& nn)
{
    Vec::Size3 MNISTDataSize = { MNISTReader::ImW, MNISTReader::ImH, 1 };
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("Input",    "Sigmoid", MNISTDataSize, { 5, 5 }, { 1,1 }, 4), nullptr));
    nn.push_back(new AveragePoolingLayer(AvgPooLayerDesc("Avg1", "Sigmoid", { 2, 2 }), nn.back()));
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("Middle",   "Sigmoid", Vec::Zeroes3, { 5, 5 }, { 2,2 }, 2), nn.back()));
    nn.push_back(new ConvolutionLayer(ConvLayerDesc("LastFC",   "Sigmoid", Vec::Zeroes3, { 5, 5 }, { 3,3 }, 1), nn.back()));

    nn.push_back(new FullyConnectedLayer("FCIn", 0, 25, "Sigmoid", nn.back()));
    nn.push_back(new FullyConnectedLayer("FCIn", 50, 25, "Sigmoid", nn.back()));

    nn.push_back(new FullyConnectedLayer("Out",  25, 10, "Sigmoid", nn.back()));
}

void main1()
{
    Vec::Size3 in;; unsigned out;
    cout << "Reading data..." << endl;
    auto data = LoadMnistData2(in, out);
    data.Summarize(Log, false);

    cout << "Building network..." << endl;
    Network nn(DATA_LOCATION "MNIST_Network.config");
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
            << "[" <<  lastCheck<< "s]:\t"
            << "Validation Accuracy : " << (acc = nn.Test(data.VldnBegin(), data.VldnEnd()) ) * 100 << "%"
            << endl;
    }
}
   
int main()
{
    try
    {
        main1();
    }
    catch (std::exception e)
    {
        cout << e.what() << endl;
        cin.get();
        return -1;
    }
    return 0;
}
