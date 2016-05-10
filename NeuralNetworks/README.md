
Neural Networks Project

Currently supports:
	1. Configurable Layers 
	 	a. Fully connected layer
		b. Convolution layer
	2. Read network config from file with simple syntax
	3. Configurable activation function (Sigmoid or TanH)

This achieves some 97%+ accuracy in classifying MNIST data (found at DATA_LOCATION macro in GCC and VS), but I have only found FCLayer-only configs that can achieve it, will implement LeNet-5 when other layers are sported. 

I have deliberately left out several features specifically to keep this implementation as simple as possible; such features include parallelization and drop connect. My aim has been to keep the implementation as close to, say, a textbook description of neural network.

I learnt a lot from these two sources: MIT OCW: https://www.youtube.com/watch?v=uXt8qF2Zzfo, and http://neuralnetworksanddeeplearning.com/ 

I will be adding more layer types, the main idea behind this project is to add CUDA support, ultimately I want to classify Leeds Butterfly set on GPU. 
