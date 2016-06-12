#!/bin/bash

curl https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz > .temp.tar.gz;
if [ $? == 0 ] ; then
	echo "Fetched file!\n"
fi
tar -xvf .temp.tar.gz 
if [ $? == 0 ] ; then
	echo "Removing temp archive file"
	rm -rf .temp.tar.gz
else
	echo "Extraction failed"
fi
