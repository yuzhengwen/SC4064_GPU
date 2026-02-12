NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80

TARGET = matrix_add
SOURCE = matrix_add.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
