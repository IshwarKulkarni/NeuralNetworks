#Make file for NeuralNetwork project

VALGRIND ?= "false"

CPP      = g++ -std=c++11 -pedantic
CPPFLAGS = -Wall -g -I../../src -DDATA_LOCATION="\"../../data/\"" 

SOURCES	:= $(shell find ../../src/data/ -name *.cxx)
SOURCES	+= $(shell find ../../src/utils/ -name *.cxx)
SOURCES	+= $(shell find ../../src/neuralnets -name *.cxx)

ifeq "$(VALGRIND)" "true"
CPPFLAGS += -DVALGRIND
endif

VALGRINDEXE = valgrind --leak-check=full  --track-origins=yes --show-leak-kinds=all 

OBJECTS    := ${SOURCES:.cxx=.o}

release: 	CPPFLAGS += -DNDEBUG -O3 
debug: 		CPPFLAGS += -D_DEBUG -Og -O0 -ggdb -DGLIBCXX_FORCE_NEW

release: 	all
debug: 		all
neuralnet: 	all

all:
	@if $(CPP) $(SOURCES) $(CPPFLAGS) -o NeuralNet ; then \
	echo "Build completed, binary: NeuralNet" ; \
    if [ "$(VALGRIND)" = "true" ] ; then \
    echo "Running Valgrind; Check file \".valgrind\" for results" ; \
    $(VALGRINDEXE) ./NeuralNet  >& .valgrind; \
    fi  \
    else \
	echo "Build Failed!" ; fi

clean:
	rm NeuralNet
