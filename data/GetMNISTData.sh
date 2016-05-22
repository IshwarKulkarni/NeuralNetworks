#!/bin/bash

mkdir MNIST;

curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > MNIST/train-images-idx3-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > MNIST/train-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  > MNIST/t10k-images-idx3-ubyte.gz 
curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz  > MNIST/t10k-labels-idx1-ubyte.gz 


cd MNIST;

gunzip *.gz
if [ $? == 0 ] ; then
	echo "Removing temp archive file"
	rm -rf *.gz
else
	echo "Extraction failed"
fi