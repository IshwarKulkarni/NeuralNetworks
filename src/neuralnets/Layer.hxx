#ifndef __LAYER_INCLUDED__
#define __LAYER_INCLUDED__

#include "utils/Vec23.hxx"
#include "utils/SimpleMatrix.hxx"
#include "Activation.hxx"

#ifdef _DEBUG
#define CNN_DEBUG 1
#define CNN_PRINT 1
#else
#define CNN_PRINT 0
#define CNN_DEBUG 0
#endif

typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;

static unsigned NumLayers2 = 0;

class Layer
{
public:
    Layer(std::string name, Vec::Size3 inSize, Vec::Size3 outSize, std::string actName, Layer* prev) :
        ThisLayerNum(++NumLayers2),
        Name(name + "<" + std::to_string(ThisLayerNum) + ">"),
        Prev(prev),
        Output(outSize), LGrads(outSize), Grads(outSize),
        PGrads(prev ? Volume(prev->Output.size) : Vec::Size3(0, 0, 0)),
        Input(prev ? prev->Output : inSize),        
        Act(GetActivationByName(actName)),
        Eta(Act->Eta)
    {
        if(!Input.size() && !Prev) throw std::invalid_argument("Input size cannot be zero for a hidden/output layer");

        if (Prev) prev->Next = this;
        
        else Input.Clear(); // if we alloc'ed Input (i.e. Prev != null), we only need size
    }

    inline const Vec::Size3& InputSize() { return Input.size; }

    virtual Volume& ForwardPass(Volume& input) = 0;

    virtual void BackwardPass(Volume& backError) = 0;

    virtual void WeightDecay(double decayRate) { Eta *= decayRate; }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        out << "\nInputs for "      + Name << Input
        ;out    << "\nPGradients for "  + Name << PGrads << "\n"
        ;out    << "\nGradients for "   + Name << Grads << "\n"
        ;out    << "\nLGradients for "  + Name << LGrads << "\n";
    }

    const Activation* GetAct() const { return Act; }

    const inline Layer* NextLayer() const { return Next; }
    const inline Layer* PrevLayer() const { return Prev; }
    
    const Volume& Out() { return Output; }
    
    virtual ~Layer() {
        Output.Clear();
        LGrads.Clear();
        Grads.Clear();
        PGrads.Clear();
    }  

protected:
    const unsigned    ThisLayerNum;
    const std::string Name;
    Layer             *Prev, *Next;

    Volume      Output, LGrads, Grads, PGrads;
    Volume      Input;
    Activation* Act;
    double      Eta;

};

#endif
