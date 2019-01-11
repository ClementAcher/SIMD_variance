GCC=gcc
OBJECTS=project.o
GCC_FLAGS=-mavx -mavx2 -mfma -lm -pthread

main:
	$(GCC) -o project project.c $(GCC_FLAGS)
	./project


debug:
	$(GCC) -o project project.c $(GCC_FLAGS) -g
	gdb ./project

.PHONY: clean
clean:
	rm $(OBJECTS)
