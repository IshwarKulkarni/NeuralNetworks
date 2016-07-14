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

#ifndef __LAYER_INCLUDED__
#define __LAYER_INCLUDED__

#include "utils/Vec23.hxx"
#include "utils/SimpleMatrix.hxx"
#include "Activation.hxx"

#include <algorithm>

#ifdef _DEBUG
#define CNN_DEBUG 1
#define CNN_PRINT 1
#else
#define CNN_PRINT 0
#define CNN_DEBUG 0
#endif

typedef SimpleMatrix::Matrix3<float_t> Volume;
typedef SimpleMatrix::Matrix<float_t> Frame;

static unsigned NumLayers2 = 0;

class Layer
{
public:
    Layer(std::string name, Vec::Size3 inSize, Vec::Size3 outSize, std::string actName, Layer* prev) :
        ThisLayerNum(++NumLayers2),
        Name(name + "<" + std::to_string(ThisLayerNum) + ">"),
        Prev(prev),
        Next(nullptr),
        Output(outSize), LGrads(outSize), Grads(outSize),
        PGrads(prev ? Volume(prev->Output.size) : Vec::Size3(0, 0, 0)), // I want to eliminate this as = prev->Grad
        Input(prev ? prev->Output : inSize),        
        Act(GetActivationByName(actName)),
        Eta(Act->Eta)
    {
        if(!Input.size() && !Prev) throw std::invalid_argument("Input size cannot be zero for a hidden/output layer");

        if (Prev) prev->Next = this;

        if(!outSize()) throw std::runtime_error("Output size cannot be zero size in Layer " + Name);
        
        if(Prev && !PGrads.size()) 
            throw std::runtime_error("PGrads cannot have zero size in Layer " + Name);
        
        if (!Prev && !Input.size()) 
            throw std::runtime_error("Input cannot have zero size in Layer " + Name);

        if(Act == nullptr) throw std::runtime_error("Activation function not found for Name " + actName);
        if (Eta <= 0) throw std::runtime_error("Eta cannot be less than or equal to zero. Is  " + std::to_string(Eta));

#ifdef _DEBUG
        Eta *= 20;
#endif

    }

    virtual void ForwardPass() = 0;

    virtual void BackwardPass(Volume& backError) = 0;

    virtual void WeightDecay(float_t decayRate) { Eta *= decayRate; }
    virtual float_t& GetEta() { return Eta; }

    virtual void Print(std::string printList, std::ostream& out = Logging::Log) const
    {
        bool all = printList.find("all") != std::string::npos;
        if (all || printList.find("Inputs") != std::string::npos)
            out << "\nInputs for " << Name << Input;
        if (all || printList.find("LGradients") != std::string::npos)
            out << "\nLGradients for " << Name << LGrads;
        else if (all || printList.find("PGradients") != std::string::npos)
            out << "\nPGradients for " << Name << PGrads;
        else if (all || printList.find("Gradients") != std::string::npos)
            out << "\nGradients for " << Name << Grads;
        if (all || printList.find("Outputs") != std::string::npos)
            out << "\nOutputs for " << Name << Output;
    }

    const Activation* GetAct() const { return Act; }

    inline Layer* NextLayer() { return Next; }
    inline Layer* PrevLayer() { return Prev; }
    
    const Volume& GetOutput() const { return Output; }

    inline SimpleMatrix::Matrix3<float_t>& GetInput()  { return Input; }
    
    void WeightSanity()    {
        
        const auto& checkIsMessedUp = [&](Volume& toCheck, std::string name)
        {
            for (auto* d = toCheck.begin(); d != toCheck.end(); ++d)
            {
                if (isnan(*d) || isinf(*d))
                {
                    Logging::Log
                        << Name << "Infinity or NaN found in  : " << toCheck << " at " << std::distance(toCheck.begin(), d)
                        << " 3d Idx: " << SimpleMatrix::IndexTo3dIdx(std::distance(toCheck.begin(), d), toCheck.size);
                    throw std::invalid_argument("Infinity in " + name);
                }
            }
        };
                   
        checkIsMessedUp(Grads, "Gradients for " + Name);
        checkIsMessedUp(Output, "Outputs for " + Name);
    }

    virtual ~Layer() {
        Output.Clear();
        LGrads.Clear();
        Grads.Clear();
        PGrads.Clear();
        if (!Prev) Input.Clear();
    }  

protected:
    const unsigned    ThisLayerNum;
    const std::string Name;
    Layer             *Prev, *Next;

    // TODO Eliminate PGrads ( = prev->Grad )
    Volume      Output, LGrads, Grads, PGrads;
    Volume      Input;
    Activation* Act;
    float_t      Eta;

};

#endif
