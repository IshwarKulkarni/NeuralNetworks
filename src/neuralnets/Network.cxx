#include "ConvolutionLayer.hxx"
#include "FullyConnectedLayer.hxx"
#include "utils/CommandLine.hxx"
#include "Network.hxx"


using namespace std;

Network::Network(std::string mnistLoc)
{
    std::ifstream inFile(mnistLoc.c_str(), ios::in | ios::binary);
    if (!inFile.good())
        throw std::invalid_argument(("File to read network config from is unavailable to read at: " + mnistLoc).c_str());

    char buffer[256];
    while (inFile && inFile.getline(buffer, 256))
    {
        std::string line(buffer);
        StringUtils::StrTrim(line);
        if (line.length() == 0 || line[0] == '#')
            continue;

        if (StringUtils::beginsWith(line, "->ConvLayer"))
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
        else if (StringUtils::beginsWith(line, "->FullyConnectedLayers"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->FullyConnectedLayersEnd");
            const auto& nameSizes = nvpp.GetPairs<unsigned>();

            for (unsigned i = 0; i < nameSizes.size() - 1; ++i)
            {
                const auto& splits = StringUtils::Split(nameSizes[i + 1].first, ",", true);

                const std::string& actName = StringUtils::StrTrim(splits.size() >1 ? splits[1] : "Sigmoid");
                const std::string& layerName = splits[0];

                size_t  inSize = nameSizes[i].second, outSize = nameSizes[i + 1].second;

                if (!inSize) inSize = back()->Out().size();

                push_back(new FullyConnectedLayer(layerName, inSize, outSize, actName, back()));
            }
        }
    }
    
    if(size())
        ErrFPrime = Volume(back()->Out().size);
    else
        throw std::invalid_argument("File could not be read!");

    Sanity();

    Print("Summary");
}
