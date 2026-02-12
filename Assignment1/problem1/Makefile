NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80

TARGET = vector_add
SOURCE = vector_add.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
