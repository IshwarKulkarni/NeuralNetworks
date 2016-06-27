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
#include "ConvolutionLayer.hxx"
#include "AveragePoolingLayer.hxx"
#include "MaxPoolingLayer.hxx"
#include "DropConnectLayer.hxx"
#include "AttenuationLayer.hxx"
#include "Network.hxx"

#include "utils/CommandLine.hxx"
#include <deque>

using namespace std;

Network::Network(std::string configFile) :
    EtaMultiplier(1.0),
    EtaDecayRate(1.0),
    WeightSanityCheck(false),
    ErrorFunction(GetErrofFunctionByName("MeanSquareError")),
    f(nullptr), 
    b(nullptr),
    CurrentStatus(Building)
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

            NVPP_GET_TYPE_WNAME(nvpp,EtaMultiplier);
            NVPP_GET_TYPE_WNAME(nvpp, EtaMultiplier);
            NVPP_GET_TYPE_WNAME(nvpp, EtaDecayRate);
            NVPP_GET_TYPE_WNAME(nvpp, WeightSanityCheck);

            std::string ErrFName = "MeanSquareError";
            NVPP_GET_TYPE_WNAME(nvpp, ErrFName);
            if ((ErrorFunction = GetErrofFunctionByName(ErrFName)) == nullptr)
                throw std::invalid_argument("Cannot find Error function by name: " + ErrFName);

            Print("Network");
        }
        else if (StringUtils::beginsWith(line, "->FullyConnectedLayerGroup"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndFullyConnectedLayerGroup");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndFullyConnectedLayerGroup");

            Vec::Size3 inSize = b ? b->GetInput().size : Vec::Size3(0, 1, 1);
            nvpp.Get("InputSize", inSize.x);
            if (!inSize) throw std::runtime_error("Input size cannot be zero on first input");

            auto nameVals = nvpp.GetPairs<size_t>();
            for (size_t i = 0; i < nameVals.size(); ++i)
            {
                inSize = b ? b->GetOutput().size : inSize; 

                if (StringUtils::beginsWith(nameVals[i].first, "InputSize")) continue;

                const auto& names = StringUtils::Split(nameVals[i].first, ",:", true);

                if (StringUtils::beginsWith(nameVals[i].first, "DropConnect"))
                    push_back(new DropConnectLayer(names[0], inSize, nvpp.Get<double>(nameVals[i].first)/100, this, b));
                else if (names.size() < 2)
                    throw std::runtime_error("Need to have a <LayerName, Activation> in line: "  + nameVals[i].first);
                else
                    push_back(new FullyConnectedLayer(names[0], inSize(), nameVals[i].second, StringUtils::StrTrim(names[1]), b));
            }
        }
        else if (StringUtils::beginsWith(line, "->ConvLayer"))
        {
            ConvLayerDesc desc; desc.KernelStride = { 1, 1 };
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndConvLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndConvLayer");

            nvpp.Get("Name",        desc.Name);
            nvpp.Get("IpSize",      desc.IpSize);
            nvpp.Get("Activation",  desc.Activation);
            nvpp.Get("NumKernels",  desc.NumberOfKernels);
            nvpp.Get("KernelSize",  desc.KernelSize);
            nvpp.Get("KernelStride",desc.KernelStride);
            nvpp.Get("Padded",      desc.PaddedConvolution);
            
            if (desc.NumberOfKernels == 0)
                desc.NumberOfKernels = ConnectionTable.Width();
            else if (ConnectionTable.Height() && ConnectionTable.Height() != desc.NumberOfKernels)
            {
                std::cerr << "Connection Table last described dictates a different number "
                    << " of outputs (" << ConnectionTable.Width() << ") than described in conv layer descriptor ("
                    <<  desc.NumberOfKernels << ")\n";
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
                    push_back(new ConvolutionLayer<true>(desc, ConnectionTable, b));
                    ConnectionTable.Clear();
                }
                else
                    push_back(new ConvolutionLayer <false>(desc, ConnectionTable, b));
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
            continue;
        }
        else if (StringUtils::beginsWith(line, "->AveragePoolingLayer"))
        {
            if (dynamic_cast<AveragePoolingLayer*>(b))
                std::cerr << "Two consecutive average layer? Bravo! ";

            AvgPooLayerDesc desc;
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndAveragePoolingLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndAveragePoolingLayer");

            nvpp.Get("Name", desc.Name);
            nvpp.Get("Activation", desc.Activation);
            nvpp.Get("WindowSize", desc.WindowSize);

            if (desc.Name.length() && GetActivationByName(desc.Activation))
                push_back(new AveragePoolingLayer(desc, b));
            else
                throw std::invalid_argument("Average pooling layer descriptor is ill formed");
        }
        else if (StringUtils::beginsWith(line, "->MaxPoolingLayer"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndMaxPoolingLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndMaxPoolingLayer");

            MaxPoolingLayerDesc desc;
            nvpp.Get("Name",        desc.Name);
            nvpp.Get("Activation",  desc.Activation);
            nvpp.Get("WindowSize",  desc.WindowSize);

            if (desc.Name.length() && GetActivationByName(desc.Activation))
                push_back(new MaxPoolingLayer(desc, b));
            else
                throw std::invalid_argument("Average pooling layer descriptor is ill formed");
        }
        else if (StringUtils::beginsWith(line, "->DropConnectLayer"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndDropConnect");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndDropConnect");

            double dropRate = 50; string name = "DropConn"; 
            Vec::Size3 inSz = b ? b->GetOutput().size : Vec::Size3(0, 0, 0);
            nvpp.Get("DropRate",    dropRate);
            nvpp.Get("Name",        name);
            nvpp.Get("InputSize",   inSz);
            dropRate /= 100;

            push_back(new DropConnectLayer(name, inSz, dropRate, this, b));

        }
        else if (StringUtils::beginsWith(line, "->AttenuationLayer"))
        {
            NameValuePairParser nvpp(inFile, ":", '\0', "#", "->EndAttenuationLayer");
            if (!nvpp.IsLastLineRead())
                throw std::runtime_error("End of file found matching ->EndAttenuationLayer");

            string name = "Attenuation";
            Vec::Size3 inSz = b ? b->GetOutput().size : Vec::Size3(0, 0, 0);
            
            double mean = 0.0, SD = .5;
            nvpp.Get("Name", name);
            nvpp.Get("InputSize", inSz);
            nvpp.Get("Mean", mean);
            nvpp.Get("SD", SD);
            push_back(new AttenuationLayer(name, inSz, mean, SD, this, b));

        }

        else if (StringUtils::beginsWith(line, "->"))
            throw std::runtime_error("Unknown layer description at: " + line);
    }

    if (size())
    {
        ErrFRes = Volume(b->GetOutput().size);
        for (auto* l : *this) l->GetEta() *= EtaMultiplier;
    }
    else
        throw std::invalid_argument("File could not be read!");

    Sanity();

    CurrentStatus = None;

    //Print("Network & Summary");
}
