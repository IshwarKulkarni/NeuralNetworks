
Neural Networks Project

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
