main: main.cpp
	mpicxx -std=c++11 -O3 -o main main.cpp -fopenmp


tobin: tobin.cpp
	g++ -o tobin tobin.cpp

totxt: totxt.cpp
	g++ -o totxt totxt.cpp


clean:
	rm -f *.txt *.bin main tobin totxt a.out
