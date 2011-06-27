CXX           = mpic++
LINK          = mpic++
CXXFLAGS      = -pipe -O2 -Wall -W -Werror
LFLAGS        = -pipe
INCPATH       = -I.
OBJECTS       = main.o \
		car.o \
		typemaps.o \
		mech.o
TARGET        = demo

.SUFFIXES: .o .cpp .cc

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

all: $(TARGET)

$(TARGET): $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
