#include <vector>
#include <sstream>
#include "utils/utils.hxx"
#include "graphics/geometry/Vec23.hxx"
#include "linalgebra_ml/NeuralNet.hxx"
#include "utils/simplematrix.hpp"
#include "utils/exceptions.hxx"
#include "utils/CommandLine.hxx"
#include "utils/Exceptions.hxx"
#include <iomanip>
#include <vector>


#ifdef _DEBUG
#define CNN_DEBUG 1
#define CNN_PRINT 1
#else
#define CNN_PRINT 0
#define CNN_DEBUG 0
#endif

typedef std::vector<Vec::Size2> Size2Vec;

typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;

struct ConvLayerDesc
{
    Vec::Size3  IpSize;
    std::string Name;
    std::string Activation;
    size_t      NumberOfKernels;
    Vec::Size2  KernelSize;
    Vec::Size2  KernelStride;
};

struct Kernel : public Volume
{
    const uint Id;
    const Vec::Size2 Stride;  // when stride is 1, max overlap; i.e. windows moves 1pixel 
    double Bias;

    inline Kernel(uint id, const ConvLayerDesc& desc, uint ipSizeZ) :
        Volume(Vec::Size3(desc.KernelSize.x | 1, desc.KernelSize.y | 1, ipSizeZ)), // odd sizes only
        Id(id),
        Stride(desc.KernelStride)
    {
        double rs = 1. / sqrt(size());

        Bias = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
        for (auto& d : *this) d = CNN_DEBUG ? 0.1 : Utils::URand(-rs, rs);
    }

    inline Vec::Size2 GetOutputSize2(Vec::Size3 inSz) {
        return Vec::Size2(iDivUp(inSz.x, Stride.x), iDivUp(inSz.y, Stride.y));
    }

    inline void Apply(const Volume& IpImage, const Activation* act, Frame& Output, Frame& LGrads)
    {
        for (uint y = 0, oy = 0; y < IpImage.size.y; y += Stride.y, ++oy)
            for (uint x = 0, ox = 0; x < IpImage.size.x; x += Stride.x, ++ox)
               Output.at(oy, ox) = act->Function(IpImage.DotAt(Vec::Loc(x, y), *this) + Bias,LGrads.at(oy, ox));
    }

    inline void BackwardPass(Frame& gradients, Volume& inputs, Volume& pgrads, double eta)
    {
        GetPGrads(gradients, pgrads);
        ChangeWeights(gradients, inputs, eta);
    }

    inline void GetPGrads(Frame& gradients, Volume& pgrads)
    {
        for (uint gy = 0; gy < gradients.size.y; ++gy)
            for (uint gx = 0; gx < gradients.size.x; ++gx)
            {
                Vec::Loc3 s = { int(gx * Stride.x - size.x / 2) , int(gy * Stride.y - size.y / 2),0 };
                auto      e = s, is = s; e += size;

                s.x = MAX(0, s.x), e.x = MIN(int(pgrads.size.x), e.x);
                s.y = MAX(0, s.y), e.y = MIN(int(pgrads.size.y), e.y);

                double grad = gradients.at(gy,gx);
                for3d2(s, e)
                    pgrads.at(z, y, x) += grad * at(z, y - is.y, x - is.x);
            }
    }

    inline void ChangeWeights(Frame& grads, Volume& ipt, double eta)
    {
        for3d(size)
            at(z, y, x) -= 
            ipt(z).DotAt(Vec::Loc(x + grads.size.x / 2 - size.x / 2, y + grads.size.y / 2 - size.y / 2), grads);
        
        double dB = 0;
        for (auto& g : grads)dB += g; 
        Bias -= eta * dB;
    }
};

std::ostream& operator << (std::ostream& stream, const Kernel& kernel)
{
    double sum = 0; for (auto& w : kernel) sum += w;
    stream
        << "Kernel ID: " << kernel.Id
        << " |  Stride: " << kernel.Stride 
        << " | Bias: " << kernel.Bias
        //<< " | Avg : " << sum / size();
        << "\n" << Volume(kernel);

    return stream;
}

static uint NumLayers2 = 0;
struct ConvLayer
{
    uint    ThisLayerNum;
    std::string Name;
    Vec::Size3  InputSize;
    ConvLayer   *Prev, *Next;
    double Eta = 0.425;
    const Activation* Act;

    std::vector<Kernel>  Kernels;

    Volume    Output, Input, LGrads, Grads, PGrads; // declare after InputSize

    ConvLayer(ConvLayerDesc& desc, ConvLayer* prev = 0) :
        Name(desc.Name + "<" + std::to_string(ThisLayerNum = ++NumLayers2) + ">"),
        InputSize(prev ? prev->Output.size : desc.IpSize),
        Prev(prev), Next(0),
        PGrads(prev ? Volume(prev->Output.size) : Vec::Size3(0, 0, 0)),
        Act(GetActivationByName(desc.Activation))
    {
        THROW_IF(!InputSize() || !desc.NumberOfKernels, DataException, "Input size or #kernels cannot be zero");

        desc.IpSize = InputSize;
        for (uint i = 0; i < desc.NumberOfKernels; ++i)
            Kernels.push_back(Kernel(i, desc, InputSize.z));

        Vec::Size3 opSize = Kernels[0].GetOutputSize2(InputSize);
        opSize.z    = Kernels.size();
        Output      = opSize;
        Grads       = opSize;
        LGrads      = opSize;

        if (Prev) prev->Next = this;
    }

    inline Volume& ForwardPass(Volume& input)
    {
        if (!Prev) Input = input;
        if (CNN_PRINT > 1) Print("Inputs , Kernels");

        for (unsigned i = 0; i < Kernels.size(); ++i)
            Kernels[i].Apply(input, Act, Output(i), LGrads(i));
        
        if (CNN_PRINT > 1) Print("Outputs");

        if (Next) return Next->ForwardPass(Output);
                
        return Output;
    }

    template<typename BackError>
    void BackwardPass(BackError& backError)
    {
        if (CNN_PRINT > 1) Log << "\nBackError at " << Name << ":\n" << backError;

        for3d(Grads.size) Grads.at(z, y, x) = LGrads.at(z, y, x) * backError.at(z, y, x);

        if (CNN_PRINT > 1) Print("Grads");

        if (Prev)
        {
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].BackwardPass(Grads(i), Prev->Output, PGrads, Eta);
            
            Prev->BackwardPass(PGrads);
            PGrads.Fill(0.f);
            if (CNN_PRINT) Print("PGrads");
            
        }
        else
            for (unsigned i = 0; i < Kernels.size(); ++i)
                Kernels[i].ChangeWeights(Grads(i), Input, Eta);

        if (CNN_PRINT > 1) Print("Kernels");
        
    }

    void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = StringUtils::Contains(printList, "all", false);

        if (all || StringUtils::Contains(printList, "Inputs", false))
            if (Input.size())
                out << "\nInputs for " << Name << ":\n" << Input;
            else if (Prev && Prev->Output.size())
                out << "\nInputs (=Prev Output) for  " << Name << "\n";
            else
                out << "\nInputs is not set for " << Name << "\n";

            if (all || StringUtils::Contains(printList, "Kernels", false))
            {
                out << "\nKernels for " << Name;
                for (uint i = 0; i < Kernels.size(); ++i)
                    out << "\nKernel " << i << " : " << Kernels[i];
            }
            if (all || StringUtils::Contains(printList, "Outputs", false))
                out << "\nOutputs for " << Name << Output;

            if (all || StringUtils::Contains(printList, "PGrads", false))
                out << "\nPGradients for " + Name << PGrads << "\n";
            else if (all || StringUtils::Contains(printList, "Grads", false))
                out << "\nGradients for " + Name << Grads << "\n"
                << "\nLGradients for " + Name << LGrads << "\n";

            if (StringUtils::Contains(printList, "Summary", false))
            {
                out << "\n--> Summary for " << Name
                    << "\nInput Size  : " << InputSize
                    << "\nNum Kernels : " << Kernels.size()
                    << "\nKernel Size : " << Kernels[0].size << " | Kernel Stride: " << Kernels[0].Stride
                    << "\nOutput Size : " << Output.size
                    << "\n";
            }
    }

    ~ConvLayer() {
        LGrads.Clear();
        Output.Clear();
        Grads.Clear();
        PGrads.Clear();
        for (auto& k : Kernels) k.Clear();
    }
};

struct ConvNetwork
{
    std::vector<ConvLayer*> Layers;
    NeuralNet FCNet;
    Volume FCNetPGrads;

    inline ConvNetwork(unsigned opSize, std::vector<ConvLayerDesc>& layerDescs)
    {
        Layers.push_back(new ConvLayer(layerDescs.front(), 0));
        for (uint i = 1; i < layerDescs.size(); ++i)
            Layers.push_back(new ConvLayer(layerDescs[i], Layers[i - 1]));
        
        auto CnnOPSizeLin = Layers.back()->Output.size();
        
        FCNet.Add(CnnOPSizeLin, opSize, "FC", GetActivationByName(layerDescs.front().Activation));

        FCNet.Layers.front()->MakePGrads();

        FCNetPGrads = Layers.back()->Output.size;
    }

    void Print(std::string printList, uint layersFromback = -1, std::ostream& out = Logging::Log) const
    {
        out << "Printing for list: " << printList << "\n";
        bool all = StringUtils::Contains(printList, "All", false);

        for (auto& l : Layers) if (layersFromback--) l->Print(printList);

        if (all || StringUtils::Contains(printList, "FC", false) || 
                    StringUtils::Contains(printList, "Summary", false))
        {
            out << " Fully connected layers: \n";
            FCNet.Print(printList);
        }
        else
            out << "\n=============================================\n";
    }

    template<typename PatternIter>
    inline void RunEpoc(PatternIter dataBegin, PatternIter dataEnd)
    {
        NumTrainInEpoc = 0;

        Volume Ip; Ip.size = Layers[0]->InputSize;
        for (auto& data = dataBegin; data != dataEnd; data++, ++NumTrainInEpoc)
        {
            Ip.data = data->Input;
            FCNet.Layers.front()->ForwardPass(Layers.front()->ForwardPass(Ip)[0][0]);

            FCNet.Layers.back()->BackwardPassStart(data->Target);
            FCNet.Layers.front()->GetPGrads(FCNetPGrads[0][0]);

            Layers.back()->BackwardPass(FCNetPGrads);
        }
        for (auto& cl : Layers) cl->Eta *= 0.9;
    }

    template<typename PatternIter>
    inline double Validate(PatternIter dataBegin, PatternIter dataEnd)
    {
        NumValCorrect = 0; NumVal = 0, VldnRMSE = 0;
        Volume Ip; Ip.size = Layers[0]->InputSize;

        auto& pred = FCNet.GetResultCmpPredicate();
        for (auto& data = dataBegin; data != dataEnd; data++, NumVal++)
        {
            Ip.data = data->Input;
            const auto& out = FCNet.Layers.front()->ForwardPass(
                this->Layers.front()->ForwardPass(Ip)[0][0]
                );
            NumValCorrect += std::equal(out.begin(), out.end(), data->Target, pred);

            VldnRMSE += double(Utils::GetRMSE(out.begin(), out.end(), data->Target));
        }

        VldnRMSE /= NumVal;
        return NumValCorrect / NumVal;
    }

    ~ConvNetwork() {
        for (auto& c : Layers) delete c;
        FCNetPGrads.Clear();
    }

private:

    uint NumVal, NumTrainInEpoc;
    double NumValCorrect, VldnRMSE;
};

