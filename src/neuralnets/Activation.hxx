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

#if defined(CUDA_PROJECT) && defined(__CUDA_ARCH__)

#define DEVICE_ __device__
#else
#define DEVICE_
#endif

DEVICE_
double inline SigmoidActivation(double p, double& grad)
{
    p = 1.f / (1.f + exp(-p));
    grad = p * (1 - p);
    return p;
}

DEVICE_
double inline TanHActivation(double p, double& grad) 
{
    p = tanh(p);
    grad = 1 - p*p;
    return p;
}

DEVICE_
double inline RELUActivation(double p, double& grad)
{
    if (p > 0) grad = 1;
    else grad = p = 0;
    return p;
}


typedef double(*ActivationFunction)(double p, double& grad);
typedef bool(*ResultCmpPredicateType)(double p1, double p2);

enum ActivationId
{
	SigmoidActivationId,
	TanHActivationId,
	RELUActivationId
};

struct Activation
{
	ActivationId Id;
    std::string Name;
    ActivationFunction Function;
    ResultCmpPredicateType ResultCmpPredicate;
    double Eta;
    Vec::Vec2<double> MinMax;
};


DEVICE_ inline double Activate(ActivationId activationId, double p) // redesign this
{
    if (activationId == SigmoidActivationId)  return 1.f / (1.f + exp(-p));
    if (activationId == TanHActivationId)	  return tanh(p);
    if (activationId == RELUActivationId)	  return (p > 0 ? p : 0);
	else
		return -1;
}

DEVICE_ inline double ActivatePrime(ActivationId activationId, double p) // redesign this
{
    if (activationId == SigmoidActivationId)  return p * (1 - p);
    if (activationId == TanHActivationId)	  return 1 - p*p;
    if (activationId == RELUActivationId)	  return (p > 0 ? 1 : 0);
    else
        return -1;
}


static Activation List[] = {
	{ SigmoidActivationId,	"Sigmoid",	SigmoidActivation,  Utils::RoundedCompare, 0.01, { 0, 1 } },
	{ TanHActivationId,		"TanH",		TanHActivation,		Utils::SameSign,       0.01, { -.9, .9 } },
	{ RELUActivationId,		"RELU",		RELUActivation,		Utils::RoundedCompare, 0.01, { 0.1, .9 } },
};

inline double GetEta(ActivationId activationId)
{
	for (unsigned i = 0; i < ARRAY_LENGTH(List); ++i)
		if (List[i].Id == activationId) return List[i].Eta;

	return -1;
}

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

#endif
