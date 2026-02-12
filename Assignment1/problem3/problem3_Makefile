NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80

TARGET = matrix_mul
SOURCE = matrix_mul.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
