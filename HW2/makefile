# Example makefile for using the mat libraries
BIN=nn

# what you want to name your tar/zip file:
TARNAME=mat31C
CXX=g++

CXXFLAGS=-O3 -Wall   # optimize
CXXFLAGS=-g -Wall    # debug
LIBS = -lm

EXAMPLES=

EXTRAS=\
randf.cpp\
randmt.cpp\
testData

SRCS=\
nnoneof.cpp\
$(BIN).cpp\
mat.cpp\
rand.cpp

HDRS=\
rand.h\
mat.h

OBJS=\
$(BIN).o\
mat.o\
rand.o

all: $(BIN) nnoneof

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $(BIN)

nnoneof: nnoneof.o mat.o rand.o
	$(CXX) $(CXXFLAGS) nnoneof.o mat.o rand.o $(LIBS) -o nnoneof

clean:
	/bin/rm -f *.o $(BIN)*.tar *~ core gmon.out a.out

tar:
	tar -cvf $(TARNAME).tar makefile $(EXAMPLES) $(SRCS) $(HDRS) $(EXTRAS)
	ls -l $(TARNAME).tar

zip:
	zip $(TARNAME).zip makefile $(EXAMPLES) $(SRCS) $(HDRS) $(EXTRAS)
	ls -l $(TARNAME).zip
