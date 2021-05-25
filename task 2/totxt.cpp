#include <fstream>
#include <complex>
#include <iostream>


typedef std::complex<double> complexd;

using namespace std;

int main(int argc, char **argv) {
	if (argc != 3) {
		cout << "Input: <input.bin> <output.txt>" << endl;
		return 0;
	}

	int n;
	fstream fin(argv[1], ios::in | ios::binary),
			fout(argv[2], ios::out | ios::trunc);

	fin.read((char *) &n, sizeof(n));
	fout << n << endl;

	long vec_length = 1 << n;
	for (long i = 0; i < vec_length; ++i) {
		double re, im;
		fin.read((char *) &re, sizeof(re));
		fin.read((char *) &im, sizeof(im));
		fout << "(" << re << "," << im << ")" << endl;
	}

	fout.close();
	fin.close();

	return 0;
}
