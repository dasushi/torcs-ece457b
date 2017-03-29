# Project: client
# Makefile created by Dev-C++ 4.9.9.2

CPP  = g++.exe
CC   = gcc.exe
WINDRES = windres.exe
RES  = 
OBJ  = BPNeuron.o BPNeuralNetwork.o CarState.o client.o NeuralDriver.o SimpleParser.o WrapperBaseDriver.o CarControl.o $(RES)
LINKOBJ  = BPNeuron.o BPNeuralNetwork.o CarState.o client.o NeuralDriver.o SimpleParser.o WrapperBaseDriver.o CarControl.o $(RES)
LIBS =  -L WS2_32.Lib -lwsock32
INCS =  -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0" 
CXXINCS =  -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0/include"  -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0/include/c++/backward"   
BIN  = client.exe
CXXFLAGS = $(CXXINCS) -fexceptions
CFLAGS = $(INCS) -mwin32
RM = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before client.exe all-after


clean: clean-custom
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "client.exe" $(LIBS)

stdafx.h.pch: stdafx.h
	$(CPP) -c stdafx.h -o stdafx.h.pch $(CXXFLAGS)
	
BPNeuron.o: BPNeuron.cpp
	$(CPP) -c BPNeuron.cpp -o BPNeuron.o $(CXXFLAGS)

BPNeuralNetwork.o: BPNeuralNetwork.cpp
	$(CPP) -c BPNeuralNetwork.cpp -o BPNeuralNetwork.o $(CXXFLAGS)

CarState.o: CarState.cpp
	$(CPP) -c CarState.cpp -o CarState.o $(CXXFLAGS)
	
NeuralDriver.o: NeuralDriver.cpp
	$(CPP) -c NeuralDriver.cpp -o NeuralDriver.o $(CXXFLAGS)

client.o: client.cpp
	$(CPP) -c client.cpp -o client.o $(CXXFLAGS)

SimpleParser.o: SimpleParser.cpp
	$(CPP) -c SimpleParser.cpp -o SimpleParser.o $(CXXFLAGS)

WrapperBaseDriver.o: WrapperBaseDriver.cpp
	$(CPP) -c WrapperBaseDriver.cpp -o WrapperBaseDriver.o $(CXXFLAGS)

CarControl.o: CarControl.cpp
	$(CPP) -c CarControl.cpp -o CarControl.o $(CXXFLAGS)
