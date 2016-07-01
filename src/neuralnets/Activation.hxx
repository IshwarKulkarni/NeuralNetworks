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

#ifndef __ACTIVATION_HXX__
#define __ACTIVATION_HXX__

#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#include "utils/Utils.hxx"

#if defined( CUDA_PROJECT) && defined(__CUDA_ARCH__)
#define DEVICE_AND_HOST __device__ __host__
#else 
#define DEVICE_AND_HOST 
#endif

DEVICE_AND_HOST
double inline SigmoidActivation(double p, double& grad)
{
    p = 1.f / (1.f + exp(-p));
    grad = p * (1 - p);
    return p;
}

DEVICE_AND_HOST
double inline TanHActivation(double p, double& grad) 
{
    p = tanh(p);
    grad = 1 - p*p;
    return p;
}

DEVICE_AND_HOST
double inline RELUActivation(double p, double& grad)
{
    if (p > 0) grad = 1;
    else grad = p = 0;
    return p;
}

typedef decltype(&SigmoidActivation)        ActivationFunction;
typedef decltype(&Utils::RoundedCompare)    ResultCmpPredicateType;

struct Activation
{
    std::string Name;
    ActivationFunction Function;
    ResultCmpPredicateType ResultCmpPredicate;
    double Eta;
    Vec::Vec2<double> MinMax;
};


static Activation List[] = {
    { "Sigmoid", SigmoidActivation,  Utils::RoundedCompare,  0.01,{ 0, 1 } },
    { "TanH",    TanHActivation,     Utils::SameSign,        0.01,{ -.9, .9 } },
    { "RELU",    RELUActivation,     Utils::RoundedCompare,  0.01,{ 0.1, .9 } },
};

inline void MultiplyEta(double etaMul)
{
    for (unsigned i = 0; i < ARRAY_LENGTH(List); ++i)
        List[i].Eta *= etaMul;
}

inline Activation* GetActivationByName(std::string name)
{
    for (unsigned i = 0; i < ARRAY_LENGTH(List); ++i)
        if (List[i].Name == name) return &(List[i]);
    
    throw std::invalid_argument( "Bad Activation name as argument: " + name);
}


inline ActivationFunction GetCudaActivationFunction(std::string name)
{
    if (name == "Sigmoid")  return SigmoidActivation;
    if (name == "TanH")     return TanHActivation;
    if (name == "RELU")     return RELUActivation;

    throw std::runtime_error("No activation function with name : " + name);
}

#endif
