MPICC = mpicc

C_FLAGS = -Wall
LD_FLAGS = -lm

SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.c)
EXECUTABLE = $(BUILD_DIR)/poisson

.PHONY: all debug clean 

all: $(EXECUTABLE)

openmp: C_FLAGS += -fopenmp
openmp: $(EXECUTABLE)

$(EXECUTABLE): $(SRCS) $(BUILD_DIR)
	$(MPICC) $(C_FLAGS) $(SRCS) $(LD_FLAGS) -o $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)
