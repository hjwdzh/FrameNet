export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64

export OPENCV_INCLUDE_DIR=/orions4-zfs/projects/jingweih/opencv/include
export OPENCV_LIBRARY_DIR=/orions4-zfs/projects/jingweih/opencv/lib

export CFLAGS="-I/data/Taskonomy/opencv/include -I$OPENCV_INCLUDE_DIR"
export DFLAGS="-L/data/Taskonomy/opencv/lib -L$OPENCV_LIBRARY_DIR -lopencv_core -lopencv_highgui"

g++ -std=c++11 -c main.cpp $CFLAGS -O2 -o main.o -fPIC
g++ -std=c++11 main.o $CFLAGS $DFLAGS -O2 -o libRender.so -shared -fPIC

#g++ -std=c++11 main.o buffer.o loader.o render.o $CFLAGS $DFLAGS -o render -lcudart

#rm *.o
