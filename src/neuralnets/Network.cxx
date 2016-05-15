#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"
#include "utils/CommandLine.hxx"
#include "ConvolutionLayer.hxx"
#include "Network.hxx"


using namespace std;

Network::Network(std::string mnistLoc) :
    EtaMultiplier(1.0),
    EtaDecayRate(1.0),
    SmallTestRate(0),
    SmallTestSize(0),
    SmallTestNum(0), 
    WeightSanityCheck(false), 
    ErrorFunction(GetErrofFunctionByName("MeanSquareError"))
{
    std::ifstream inFile(mnistLoc.c_str(), ios::in | ios::binary);
    if (!inFile.good())
        throw std::invalid_argument(("File to read network config from is unavailable to read from: " + mnistLoc).c_str());

    bool networkDescribed = false;
    char buffer[256];
    while (inFile && inFile.getline(buffer, 256))
    {
        std::string line(buffer);
        StringUtils::StrTrim(line);
        if (line.length() == 0 || line[0] == '#')
            continue;

        if (StringUtils::beginsWith(line, "->NetworkDescription"))
        {
            if (networkDescribed)
                throw std::invalid_argument("You can describe network only once");

            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndNetworkDescription");

            nvpp.Get("EtaMultiplier", EtaMultiplier);
            nvpp.Get("EtaDecayRate",  EtaDecayRate);
            nvpp.Get("SmallTestRate", SmallTestRate);
            nvpp.Get("SmallTestSize", SmallTestSize);
            nvpp.Get("SmallTestNum",  SmallTestNum);
            nvpp.Get("WeightSanityCheck", WeightSanityCheck);

            std::string errfName = "MeanSquareError";
            nvpp.Get("ErrorFunction", errfName);
            if ((ErrorFunction = GetErrofFunctionByName(errfName)) == nullptr)
                throw std::invalid_argument("Cannot find Error function by name: " + errfName);

            networkDescribed = true;
        }
        else if (StringUtils::beginsWith(line, "->ConvLayer"))
        {
            std::string last;
            while (inFile && last != "true")
            {
                ConvLayerDesc desc;
                NameValuePairParser nvpp(inFile, ":", '\0', "#", "->ConvDescEnd");
                nvpp.Get("Last", last);

                nvpp.Get("Name", desc.Name);
                nvpp.Get("IpSize", desc.IpSize);
                nvpp.Get("Activation", desc.Activation);
                nvpp.Get("NumKernels", desc.NumberOfKernels);
                nvpp.Get("KernelSize", desc.KernelSize);
                nvpp.Get("KernelStride", desc.KernelStride);
                nvpp.Get("NumOutputs", desc.NumOuputs);

                if (!size() && desc.IpSize() == 0)
                    throw std::invalid_argument("First convolution layer description should have a valid input size");

                if (desc.Name.length() &&
                    GetActivationByName(desc.Activation) &&
                    desc.NumberOfKernels &&
                    desc.KernelSize() &&
                    desc.KernelStride())
                {
                    push_back(new ConvolutionLayer(desc, size() ? back() : nullptr));
                }
                else
                    throw std::invalid_argument("Convolution layer descriptor is ill formed");
            }
        }
        else if (StringUtils::beginsWith(line, "->AveragePoolingLayer"))
        {
            if (!size())
                throw std::invalid_argument("Average pooling layer cannot be first layer");

            if (dynamic_cast<AveragePoolingLayer*>(back()))
                Logging::Log << "Two consecutive average layer? Bravo! ";

            AvgPooLayerDesc desc;
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndAveragePoolingLayer");
            
            nvpp.Get("Name", desc.Name);
            nvpp.Get("Activation", desc.Activation);
            nvpp.Get("WindowSize", desc.WindowSize);
            
            if (desc.Name.length() && GetActivationByName(desc.Activation) )
                push_back(new AveragePoolingLayer(desc, back()));
            else
                throw std::invalid_argument("Average pooling layer descriptor is ill formed");
        }
        else if (StringUtils::beginsWith(line, "->FullyConnectedLayers"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndFullyConnectedLayers");
            const auto& nameSizes = nvpp.GetPairs<unsigned>();

            for (unsigned i = 0; i < nameSizes.size() - 1; ++i)
            {
                const auto& splits = StringUtils::Split(nameSizes[i + 1].first, ",", true);

                const std::string& actName = StringUtils::StrTrim(splits.size() >1 ? splits[1] : "Sigmoid");
                const std::string& layerName = splits[0];

                unsigned  inSize = nameSizes[i].second, outSize = nameSizes[i + 1].second;

                if (!inSize) inSize = back()->Out().size();

                push_back(new FullyConnectedLayer(layerName, inSize, outSize, actName, back()));
            }
        }
    }
    
    if (size())
    {
        ErrFRes = Volume(back()->Out().size);
        for (auto* l : *this) l->GetEta() *= EtaMultiplier;
    }
    else
        throw std::invalid_argument("File could not be read!");

    Sanity();

    Print("Network & Summary");
}
