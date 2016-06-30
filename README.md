
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

	|-build
		| - VS2015 <Visual Studio 2015/2013 build files>
		| - GCC <GCC 5.3 build files>


    Common Steps:
    cd data;
    ./GetMNISTDATA.sh ;
		
    Linux/GCC: 
    cd NeuralNetworks/GCC 
    make # or
    make debug

    Windows/Visual Studio 2015:
    ->Open NeuralNetworks/VS015
    ->double click to open the sln file
    ->Build and run 
    
    Windows/Visual Studio 2013:
    ->Replace "v140" to "v120" in NeuralNetworks.vcxproj
    ->double click to open the sln file
    ->Build and run 
    
		
Currently supports:
	1. Configurable Layers 
	 	a. Fully connected layer
		b. Convolution layer : can have partial connection ^1
		c. Average pooling layer
		d. Max pooling layer
        e. Drop Connect layer
        f. Attenuation layer
	2. Read network config from file with simple syntax
	3. Configurable activation function (Sigmoid/TanH/RELU)
    
^1: 
This achieves some 97%+ accuracy in classifying MNIST data (found at DATA_LOCATION macro in GCC and VS) in test1.config.

I have deliberately left out several features specifically to keep this implementation as simple as possible; such features include parallelization.

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
