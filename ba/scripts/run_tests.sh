SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_DIR=$SCRIPT_PATH/../build

cd $BUILD_DIR
ctest --output-on-failure