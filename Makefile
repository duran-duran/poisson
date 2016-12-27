MPICC = mpicc

C_FLAGS = -Wall

SRC_DIR = src
BUILD_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLE = $(BUILD_DIR)/poisson

.PHONY: all debug clean 

all: $(EXECUTABLE)

debug: CPP_FLAGS += -DDEBUG
debug: $(EXECUTABLE)

$(EXECUTABLE): $(SRCS) $(BUILD_DIR)
	$(MPICC) $(C_FLAGS) $(SRCS) -o $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -r $(BUILD_DIR)