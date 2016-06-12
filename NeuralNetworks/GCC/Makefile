#Make file for NeuralNetwork project

CPP      = g++ -std=c++11 -pedantic
CPPFLAGS = -Wall -g -I../../src -DDATA_LOCATION="\"../../data/\"" 


SOURCES	:= $(shell find ../../src/data/ -name *.cxx)
SOURCES	+= $(shell find ../../src/utils/ -name *.cxx)
SOURCES	+= $(shell find ../../src/neuralnets -name *.cxx)

OBJECTS    := ${SOURCES:.cxx=.o}

release: 	CPPFLAGS += -DNDEBUG -O3 
debug: 		CPPFLAGS += -D_DEBUG -Og -O0

release: 	all
debug: 		all
neuralnet: 	all

all:
	@if $(CPP) $(SOURCES) $(CPPFLAGS) -o NeuralNet ; then \
	echo "Build completed, binary: NeuralNet" ; else \
	echo "Build Failed!" ; fi

clean:
	rm NeuralNet
