# Specifying which compiler to use
CC = g++							

# Here Include specifies which directories can be added in the include path of the main.cpp
CFLAGS = -std=c++17	 -Iinclude/ 
# What libraries to be imported here it is opencv 
LIBS = -I/usr/include/eigen3
# Similar to how we used to link ad make object files in the terminal
# the command make build SRC=<FILENAME> should build the executable in the subfolder
.PHONY: build
ifeq ($(SRC), $(link), $(OBJ))
build:
	$(error "SRC is not set")
else
build:
	@echo "Building..."
	@mkdir -p output
	@$(CC) ./examples/$(SRC) ./src/$(link) -o ./output/$(OBJ) $(CFLAGS)
endif	

# if folder is not set, clean all build files all subfolders
.PHONY: clean
clean:
	@echo "Cleaning..."
	@rm -rf $(PROJECT)
