% : src/%/main.cu
	nvcc -ccbin /usr/bin/gcc-13 -lstdc++ -Isrc/common -o build/$@ $^

%_run : build/%
	build/$* < src/$*/input.txt

clean :
	rm build/*
