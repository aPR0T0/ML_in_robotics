# Compiler and flags
CXX = g++
CXXFLAGS = -O2 -std=c++17

# Directories
SRC_DIR = src
DATA_DIR = data
BUILD_DIR = build

# Files
TARGET = $(BUILD_DIR)/mnist_split
SOURCES = $(SRC_DIR)/main.cpp

# MNIST Dataset URLs
MNIST_URL = http://yann.lecun.com/exdb/mnist
FILES = train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
        t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz

# Default target
all: download_data $(TARGET)

# Rule to compile the C++ code
$(TARGET): $(SOURCES)
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to download MNIST data
download_data:
	mkdir -p $(DATA_DIR)
	cd $(DATA_DIR) && for file in $(FILES); do \
	    if [ ! -f $$file ]; then \
	        wget $(MNIST_URL)/$$file; \
	        gunzip -f $$file; \
	    fi \
	done

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(DATA_DIR)/*.ubyte

.PHONY: all download_data clean
