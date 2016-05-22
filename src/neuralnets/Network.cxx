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

#include "FullyConnectedLayer.hxx"
#include "AveragePoolingLayer.hxx"
#include "utils/CommandLine.hxx"
#include "ConvolutionLayer.hxx"
#include "Network.hxx"


using namespace std;

Network::Network(std::string configFile) :
    EtaMultiplier(1.0),
    EtaDecayRate(1.0),
    SmallTestRate(0),
    SmallTestSize(0),
    SmallTestNum(0),
    WeightSanityCheck(false),
    ErrorFunction(GetErrofFunctionByName("MeanSquareError"))
{
    std::ifstream inFile(configFile.c_str(), ios::in | ios::binary);
    if (!inFile.good())
        throw std::invalid_argument(("File to read network config from is unavailable to read from: " + configFile).c_str());

    this->ConfigSource = configFile;

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
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndNetworkDescription");

            nvpp.Get("EtaMultiplier", EtaMultiplier);
            nvpp.Get("EtaDecayRate", EtaDecayRate);
            nvpp.Get("SmallTestRate", SmallTestRate);
            nvpp.Get("SmallTestSize", SmallTestSize);
            nvpp.Get("SmallTestNum", SmallTestNum);
            nvpp.Get("WeightSanityCheck", WeightSanityCheck);

            std::string errfName = "MeanSquareError";
            nvpp.Get("ErrorFunction", errfName);
            if ((ErrorFunction = GetErrofFunctionByName(errfName)) == nullptr)
                throw std::invalid_argument("Cannot find Error function by name: " + errfName);

            networkDescribed = true;
        }
        else if (StringUtils::beginsWith(line, "->ConvLayer"))
        {

            ConvLayerDesc desc; desc.KernelStride = { 1, 1 };
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndConvLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndConvLayer");

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
        else if (StringUtils::beginsWith(line, "->AveragePoolingLayer"))
        {
            if (!size())
                throw std::invalid_argument("Average pooling layer cannot be first layer");

            if (dynamic_cast<AveragePoolingLayer*>(back()))
                Logging::Log << "Two consecutive average layer? Bravo! ";

            AvgPooLayerDesc desc;
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndAveragePoolingLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndAveragePoolingLayer");

            nvpp.Get("Name", desc.Name);
            nvpp.Get("Activation", desc.Activation);
            nvpp.Get("WindowSize", desc.WindowSize);

            if (desc.Name.length() && GetActivationByName(desc.Activation))
                push_back(new AveragePoolingLayer(desc, back()));
            else
                throw std::invalid_argument("Average pooling layer descriptor is ill formed");
        }
        else if (StringUtils::beginsWith(line, "->FullyConnectedLayers"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndFullyConnectedLayers");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndFullyConnectedLayers");

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
