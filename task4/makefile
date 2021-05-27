
.PHONY: test


main: main.cpp logic_gate.h
	mpicxx -std=c++11 -o main main.cpp -fopenmp


tobin: tobin.cpp
	g++ -o tobin tobin.cpp

totxt: totxt.cpp
	g++ -o totxt totxt.cpp

hadamar: hadamar.cpp logic_gate.h
	mpicxx -std=c++11 -o hadamar hadamar.cpp -fopenmp

N_hadamar: N_hadamar.cpp logic_gate.h
	mpicxx -std=c++11 -o N_hadamar N_hadamar.cpp -fopenmp

Not: Not.cpp logic_gate.h
	mpicxx -std=c++11 -o Not Not.cpp -fopenmp

C_not: C_not.cpp logic_gate.h
	mpicxx -std=c++11 -o C_not C_not.cpp -fopenmp

ROT: ROT.cpp logic_gate.h
	mpicxx -std=c++11 -o ROT ROT.cpp -fopenmp


C_ROT: C_ROT.cpp logic_gate.h
	mpicxx -std=c++11 -o C_ROT C_ROT.cpp -fopenmp

compare: compare.cpp
	g++ -o compare compare.cpp

testhadamar: hadamar compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./hadamar 6 1 1 1 1 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	mpirun --oversubscribe -n 2 ./hadamar 6 1 1 1 1 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	mpirun --oversubscribe -n 4 ./hadamar 6 1 1 1 1 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	mpirun --oversubscribe -n 4 ./hadamar 6 1 1 1 4 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	mpirun --oversubscribe -n 4 ./hadamar 6 1 1 1 8 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	mpirun --oversubscribe -n 8 ./hadamar 6 1 1 1 1 hadamar.bin
	@./compare test/hadamar.bin hadamar.bin
	@rm hadamar.bin
	@rm in.bin
	@echo "hadamar test PASSED"

testN_hadamar: N_hadamar compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./N_hadamar 6 1 1 1 1 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	mpirun --oversubscribe -n 2 ./N_hadamar 6 1 1 1 1 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	mpirun --oversubscribe -n 4 ./N_hadamar 6 1 1 1 1 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	mpirun --oversubscribe -n 4 ./N_hadamar 6 1 1 1 4 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	mpirun --oversubscribe -n 4 ./N_hadamar 6 1 1 1 8 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	mpirun --oversubscribe -n 8 ./N_hadamar 6 1 1 1 1 N_hadamar.bin
	@./compare test/N_hadamar.bin N_hadamar.bin
	@rm N_hadamar.bin
	@rm in.bin
	@echo "N_hadamar test PASSED"

testNot: Not compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./Not 6 1 1 1 1 Not.bin
	@./compare test/Not.bin Not.bin
	mpirun --oversubscribe -n 2 ./Not 6 1 1 1 1 Not.bin
	@./compare test/Not.bin Not.bin
	mpirun --oversubscribe -n 4 ./Not 6 1 1 1 1 Not.bin
	@./compare test/Not.bin Not.bin
	mpirun --oversubscribe -n 4 ./Not 6 1 1 1 4 Not.bin
	@./compare test/Not.bin Not.bin
	mpirun --oversubscribe -n 4 ./Not 6 1 1 1 8 Not.bin
	@./compare test/Not.bin Not.bin
	mpirun --oversubscribe -n 8 ./Not 6 1 1 1 1 Not.bin
	@./compare test/Not.bin Not.bin
	@rm Not.bin
	@rm in.bin
	@echo "Not test PASSED"

testROT: ROT compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./ROT 6 1 1 1 1 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	mpirun --oversubscribe -n 2 ./ROT 6 1 1 1 1 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	mpirun --oversubscribe -n 4 ./ROT 6 1 1 1 1 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	mpirun --oversubscribe -n 4 ./ROT 6 1 1 1 4 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	mpirun --oversubscribe -n 4 ./ROT 6 1 1 1 8 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	mpirun --oversubscribe -n 8 ./ROT 6 1 1 1 1 ROT.bin 3.1415
	@./compare test/ROT.bin ROT.bin
	@rm ROT.bin
	@rm in.bin
	@echo "ROT test PASSED"

testC_not: C_not compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./C_not 6 6 1 1 1 C_not.bin
	@./compare test/C_not.bin C_not.bin
	mpirun --oversubscribe -n 2 ./C_not 6 6 1 1 1 C_not.bin
	@./compare test/C_not.bin C_not.bin
	mpirun --oversubscribe -n 4 ./C_not 6 6 1 1 1 C_not.bin
	@./compare test/C_not.bin C_not.bin
	mpirun --oversubscribe -n 4 ./C_not 6 6 1 1 4 C_not.bin
	@./compare test/C_not.bin C_not.bin
	mpirun --oversubscribe -n 4 ./C_not 6 6 1 1 8 C_not.bin
	@./compare test/C_not.bin C_not.bin
	mpirun --oversubscribe -n 8 ./C_not 6 6 1 1 1 C_not.bin
	@./compare test/C_not.bin C_not.bin
	@rm C_not.bin
	@rm in.bin
	@echo "C_not test PASSED"

testC_ROT: C_ROT compare
	@cp ./test/in.bin .
	mpirun --oversubscribe -n 1 ./C_ROT 6 1 6 1 1 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	mpirun --oversubscribe -n 2 ./C_ROT 6 1 6 1 1 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	mpirun --oversubscribe -n 4 ./C_ROT 6 1 6 1 1 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	mpirun --oversubscribe -n 4 ./C_ROT 6 1 6 1 4 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	mpirun --oversubscribe -n 4 ./C_ROT 6 1 6 1 8 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	mpirun --oversubscribe -n 8 ./C_ROT 6 1 6 1 1 C_ROT.bin 3.1415
	@./compare test/C_ROT.bin C_ROT.bin
	@rm C_ROT.bin
	@rm in.bin
	@echo "C_ROT test PASSED"

test: testhadamar testN_hadamar testNot testROT testC_not testC_ROT


clean :
	rm -f *.txt *.bin main compare totxt tobin hadamar N_hadamar Not ROT C_not C_ROT
