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
