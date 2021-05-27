#include <fstream>
#include <complex>
#include <iostream>


typedef std::complex<double> complexd;

using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "Format: in1.bin in2.bin" << endl;
        return 0;
    }

    fstream fin1(argv[1], ios::in | ios::binary),
            fin2(argv[2], ios::in | ios::binary);
    int n1, n2;
    fin1.read(reinterpret_cast<char *> (&n1), sizeof(n1));
    fin2.read(reinterpret_cast<char *> (&n2), sizeof(n2));
    if (n1 != n2) {
        cout << "Qubit amount diff" << endl;
        fin1.close();
        fin2.close();
        return -1;
    }
    int n = n1;
    long vec_length = 1 << n;
	double norm_length_ = 0;
    for (int i = 0; i < vec_length; ++i) {
        double re1, im1, re2, im2;
        fin1.read(reinterpret_cast<char *> (&re1), sizeof(re1));
        fin1.read(reinterpret_cast<char *> (&im1), sizeof(im1));
        fin2.read(reinterpret_cast<char *> (&re2), sizeof(re2));
        fin2.read(reinterpret_cast<char *> (&im2), sizeof(im2));
        norm_length_ += re2 * re2 + im2 * im2;
        if (re1 != re2 || im1 != im2) {
            cout << "Diff in qubit number " << i << endl
                 << re1 << ' ' << im1 << ' '
                 << re2 << ' ' << im2 << ' ' << endl;
            fin1.close();
            fin2.close();
            return -1;
        }
    }
    cout << "Canonization test passed" << endl;
	if (norm_length_ < 1.000001 && norm_length_ > 0.999999) {
        cout << "Blackbox test passed" << endl;
    } else {
        cout << "Blackbox test failed, norm = " << norm_length_ << endl;
    }
    fin1.close();
    fin2.close();
    return 0;
}
