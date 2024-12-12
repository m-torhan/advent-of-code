% : src/%.cu
	nvcc -O3 --gpu-architecture compute_86 --expt-relaxed-constexpr -ccbin /usr/bin/clang++ -lstdc++ -Isrc/common -o build/$@ $^

%p1_run : build/%p1
	build/$*p1 < input/$*

%p2_run : build/%p2
	build/$*p2 < input/$*

clean :
	rm build/*
