
Neural Networks Project

The repo is organized as 

	/
	|- data
	    |- Scripts to fetch CIFAR and MNSIT data
		|- Network configuration file(s)

	|- src  < source >
		|- neuralnets  <files concerning neural networks >
	    |- data <reading and writing data from above dir >
	    
		|- utils < other sundry utilities >

	|-NeuralNetworks
		| - VS <Visual Studio 2015/2013 build files>
		| - GCC <GCC 5.3 build files>


common:
cd data;
./GetMNISTDATA.sh ;
		
GCC: 
cd NeuralNetworks/GCC 
make # or
make # debug

Visual Studio:
Open NeuralNetworks/VS015
<double click to open the solution file>	

		
Currently supports:
	1. Configurable Layers 
	 	a. Fully connected layer
		b. Convolution layer : can have partial connection ^1
		c. Average pooling layer
		d. Max pooling layer
	2. Read network config from file with simple syntax
	3. Configurable activation function (Sigmoid or TanH)

^1: 
This achieves some 97%+ accuracy in classifying MNIST data (found at DATA_LOCATION macro in GCC and VS), but I have only found FCLayer-only configs that can achieve it.

I have deliberately left out several features specifically to keep this implementation as simple as possible; such features include parallelization and drop connect. 

My aim has been to keep the implementation as close to, say, a textbook description of neural network.

I learnt a lot from these two sources: MIT OCW: https://www.youtube.com/watch?v=uXt8qF2Zzfo, and http://neuralnetworksanddeeplearning.com/ 

The main motication behind this project is to add CUDA support and use CPU implementation as an architectural template and verification base for CUDA.

Ultimately I want to classify Leeds Butterfly set on GPU. 

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
