DIRS := . MinCost GEOM 

SOURCES := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS := $(patsubst %.cpp, %.o, $(SOURCES))

CFLAGS := -O3 -D_NDEBUG -fPIC
CXX ?= c++
LIBS := 
INCLUDES := 
LIBDIR := 

# Add librt if the target platform is not Darwin (OS X)
ifneq ($(shell uname -s),Darwin)
    LIBS += -lrt
endif

#all: blossom5
all: libpypm.so

#blossom5: ${OBJS}
#	$(CXX) $(CFLAGS) ${LIBDIR} -o $@ ${OBJS} ${LIBS}

libpypm.so: ${OBJS}
	$(CXX) -shared ${OBJS} -o $@  ${LIBS}

.cpp.o:
	$(CXX) $(CFLAGS) ${INCLUDES} $< -c -o $@

clean:
#	rm -f ${OBJS} blossom5
	rm -f ${OBJS} libpypm.so
