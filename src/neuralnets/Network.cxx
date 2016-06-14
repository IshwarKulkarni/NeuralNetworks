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
#include "MaxPoolingLayer.hxx"
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

    std::string line;

    SimpleMatrix::Matrix<bool> ConnectionTable (Vec::Size2( 0, 0 ),nullptr); // just a name;
    
    while (inFile && std::getline(inFile,line))
    {
        StringUtils::StrTrim(line);
        if (line.length() == 0 || line[0] == '#')
            continue;

        if (StringUtils::beginsWith(line, "->NetworkDescription"))
        {
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

            if (desc.NumberOfKernels == 0)
                desc.NumberOfKernels = ConnectionTable.Width();
            else if (ConnectionTable.Height() && ConnectionTable.Height() != desc.NumberOfKernels)
            {
                std::cerr << "Connection Table last described dictates a different number "
                    << " of outputs (" << ConnectionTable.Width() << ") than described in conv layer descriptor\n";
                throw std::runtime_error("Bad Conv layer descriptor");
            }

            if (!size() && desc.IpSize() == 0)
                throw std::invalid_argument("First convolution layer description should have a valid input size");

            if (desc.Name.length() &&
                GetActivationByName(desc.Activation) &&
                desc.NumberOfKernels &&
                desc.KernelSize() &&
                desc.KernelStride())
            {
             
                if (ConnectionTable.size())
                {
                    push_back(new ConvolutionLayer<true>(desc, ConnectionTable, size() ? back() : nullptr));
                    ConnectionTable.Clear();
                }
                else
                    push_back(new ConvolutionLayer <false>(desc, ConnectionTable, size() ? back() : nullptr));
            }
            else
                throw std::invalid_argument("Convolution layer descriptor is ill formed");

        }
        else if (StringUtils::beginsWith(line, "->ConnectionTable"))
        {
            stringstream csvStrm;
            string lineFull;
            while (inFile && std::getline(inFile, line) )
            {
                StringUtils::StrTrim(line);
                if (line.length() == 0 || line[0] == '#') continue;
                if (StringUtils::beginsWith(line, "->EndConnectionTable")) break;
                csvStrm << line << "\n";
            }

            if (!inFile)
                throw std::runtime_error("Cannot create connection table unless there's a convolution layer after");

            ConnectionTable = SimpleMatrix::ReadCSV<bool>(csvStrm);
        }
        else if (StringUtils::beginsWith(line, "->AveragePoolingLayer"))
        {
            if (dynamic_cast<AveragePoolingLayer*>(back()))
                std::cerr << "Two consecutive average layer? Bravo! ";

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
        else if (StringUtils::beginsWith(line, "->MaxPoolingLayer"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndMaxPoolingLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndMaxPoolingLayer");

            MaxPoolingLayerDesc desc;
            nvpp.Get("Name", desc.Name);
            nvpp.Get("Activation", desc.Activation);
            nvpp.Get("WindowSize", desc.WindowSize);

            if (desc.Name.length() && GetActivationByName(desc.Activation))
                push_back(new MaxPoolingLayer(desc, back()));
            else
                throw std::invalid_argument("Average pooling layer descriptor is ill formed");

        }
        else if (StringUtils::beginsWith(line, "->FullyConnectedLayerGroup"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndFullyConnectedLayerGroup");
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

                push_back(new FullyConnectedLayer(layerName, inSize, outSize, actName, size() ? back() : nullptr));
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
