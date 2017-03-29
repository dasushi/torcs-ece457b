# Project: client
# Makefile created by Dev-C++ 4.9.9.2

CPP  = g++.exe
CC   = gcc.exe
WINDRES = windres.exe
RES  = 
OBJ  = CarState.o client.o SimpleDriver.o NNDataLog.o SimpleParser.o WrapperBaseDriver.o CarControl.o $(RES)
LINKOBJ  = CarState.o client.o SimpleDriver.o NNDataLog.o SimpleParser.o WrapperBaseDriver.o CarControl.o $(RES)
LIBS =  -L WS2_32.Lib -lwsock32
INCS =  -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0" -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0" 
CXXINCS =  -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0/include" -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0/include/c++/backward" -I"C:/cygwin64/lib/gcc/x86_64-pc-cygwin/5.4.0" 
BIN  = client.exe
CXXFLAGS = $(CXXINCS)   -Wall
CFLAGS = $(INCS)   -Wall
RM = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before client.exe all-after


clean: clean-custom
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "client.exe" $(LIBS)

CarState.o: CarState.cpp
	$(CPP) -c CarState.cpp -o CarState.o $(CXXFLAGS)

client.o: client.cpp
	$(CPP) -c client.cpp -o client.o $(CXXFLAGS)
	
NNDataLog.o: NNDataLog.cpp
	$(CPP) -c NNDataLog.cpp -o NNDataLog.o $(CXXFLAGS)

SimpleDriver.o: SimpleDriver.cpp
	$(CPP) -c SimpleDriver.cpp -o SimpleDriver.o $(CXXFLAGS)

SimpleParser.o: SimpleParser.cpp
	$(CPP) -c SimpleParser.cpp -o SimpleParser.o $(CXXFLAGS)

WrapperBaseDriver.o: WrapperBaseDriver.cpp
	$(CPP) -c WrapperBaseDriver.cpp -o WrapperBaseDriver.o $(CXXFLAGS)

CarControl.o: CarControl.cpp
	$(CPP) -c CarControl.cpp -o CarControl.o $(CXXFLAGS)
