SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_DIR=$SCRIPT_PATH/../build
DEPS_DIR=$SCRIPT_PATH/../deps
if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
fi

## Build deps
# Build unordered_dense
cd $DEPS_DIR/unordered_dense
if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake ..
cmake --build .
# Build lgmath
cd $DEPS_DIR/lgmath
if [ ! -d "build" ]; then
  mkdir build
fi
cd build
cmake ..
cmake --build .

## Build ba
cd $BUILD_DIR
cmake .. -DCMAKE_CXX_FLAGS="-fopenmp -O3 -march=native -Wall -pedantic"
cmake --build .