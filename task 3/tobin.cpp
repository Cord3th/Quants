#include <iostream>
#include <fstream>
#include <complex>

typedef std::complex<double> complexd;

using namespace std;

int main(int argc, char **argv) {
	if (argc != 3) {
		cout << "Input: <in.txt> <in.bin>" << endl;
		return 0;
	}

	int n;
	fstream fin(argv[1]),
			fout(argv[2], ios::out | ios::binary | ios::trunc);

	fin >> n;
	fout.write((char *) &n, sizeof(n));

	long vec_length = 1 << n;
	for (long i = 0; i < vec_length; ++i) {
		complexd temp;
		fin >> temp;
		fout.write((char *)	&temp, sizeof(double));
		fout.write((char *) &temp + sizeof(double), sizeof(double));
	}

	fout.close();

	return 0;
}
