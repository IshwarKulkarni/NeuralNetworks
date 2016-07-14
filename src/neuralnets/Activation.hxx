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
float_t inline SigmoidActivation(float_t p, float_t& grad)
{
    p = 1.f / (1.f + exp(-p));
    grad = p * (1 - p);
    return p;
}

DEVICE_
float_t inline TanHActivation(float_t p, float_t& grad) 
{
    p = tanh(p);
    grad = 1 - p*p;
    return p;
}

DEVICE_
float_t inline RELUActivation(float_t p, float_t& grad)
{
    if (p > 0) grad = 1;
    else grad = p = 0;
    return p;
}


typedef float_t(*ActivationFunction)(float_t p, float_t& grad);
typedef bool(*ResultCmpPredicateType)(float_t p1, float_t p2);

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
    float_t Eta;
    Vec::Vec2<float_t> MinMax;
};


DEVICE_ inline float_t Activate(ActivationId activationId, float_t p) // redesign this
{
    if (activationId == SigmoidActivationId)  return 1.f / (1.f + exp(-p));
    if (activationId == TanHActivationId)	  return tanh(p);
    if (activationId == RELUActivationId)	  return (p > 0 ? p : 0);
	else
		return -1;
}

DEVICE_ inline float_t ActivatePrime(ActivationId activationId, float_t p) // redesign this
{
    if (activationId == SigmoidActivationId)  return float_t(p * (1 - p));
    if (activationId == TanHActivationId)	  return float_t(1 - p*p);
    if (activationId == RELUActivationId)	  return float_t((p > 0 ? 1 : 0));
    else
        return -1;
}


static Activation List[] = {
	{ SigmoidActivationId,	"Sigmoid",	SigmoidActivation,  Utils::RoundedCompare, float_t(0.01), { float_t(  0), float_t( 1) } },
	{ TanHActivationId,		"TanH",		TanHActivation,		Utils::SameSign,       float_t(0.01), { float_t(-.9), float_t(.9) } },
	{ RELUActivationId,		"RELU",		RELUActivation,		Utils::RoundedCompare, float_t(0.01), { float_t(0.1), float_t(.9) } },
};

inline float_t GetEta(ActivationId activationId)
{
	for (unsigned i = 0; i < ARRAY_LENGTH(List); ++i)
		if (List[i].Id == activationId) return List[i].Eta;

	return -1;
}

inline void MultiplyEta(float_t etaMul)
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
