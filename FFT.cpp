#include <iostream>
#include <complex>
#include <cmath>
#include <Eigen/Sparse>

double pi = M_PI;

int reverse(int n, int size) {
    int rev = 0;
    for (int i = 0; i<size; i++) {
        rev <<= 1;
        if ((n & 1) == 1) {
            rev ^= 1;
        }
        n >>= 1;
    }
    return rev;
}
void diag_ones(Eigen::SparseMatrix<std::complex<double>>* matrix, int row, int column, int size) {
    for (int i = 0; i<size; i++) {
        matrix->insert(row+i,column+i) = 1;
    }
}
void diag_omega(Eigen::SparseMatrix<std::complex<double>>* matrix, std::complex<double>* roots, int row, int column, int size, int num_points, bool negative = false) {
    for (int i = 0; i<size; i++) {
        int spacing = num_points/size/2;
        matrix->insert(row+i,column+i) = pow(-1,negative)*roots[spacing*i];
    }
}
void M_block(Eigen::SparseMatrix<std::complex<double>>* matrix, std::complex<double>* roots, int row, int column, int size, int num_points) {
    int length = size/2;
    diag_ones(matrix, row, column, length);
    diag_ones(matrix, row+length, column, length);
    diag_omega(matrix, roots, row, column+length, length, num_points);
    diag_omega(matrix, roots, row+length, column+length, length, num_points, true);
}
Eigen::SparseMatrix<std::complex<double>> M(std::complex<double>* roots, int size, int blocks) {
    Eigen::SparseMatrix<std::complex<double>> mat(size,size);
    int block_size = size/blocks;

    for (int i = 0; i<blocks; i++) {
        M_block(&mat, roots, i*block_size, i*block_size, block_size, size);
    }
    return mat;
}
Eigen::SparseMatrix<std::complex<double>> F(int size) {
    Eigen::SparseMatrix<std::complex<double>> matrix(size,size);
    for (int i = 0; i<size; i+=2) {
        matrix.insert(i,i) = 1;
        matrix.insert(i,i+1) = 1;
        matrix.insert(i+1,i) = 1;
        matrix.insert(i+1,i+1) = -1;
    }
    return matrix;
}
Eigen::VectorXcd fft(std::complex<double> x[], int size) {
    std::complex<double> omega(cos(2*pi/size),-sin(2*pi/size));
    int nearest_power = ceil(log2(size));
    int padded_size = pow(2,nearest_power);
    Eigen::VectorXcd transform(padded_size);
    for (int i=0; i<padded_size; i++) {
        int rev = reverse(i, nearest_power);
        if (rev<size) {
            transform[i] = x[rev];
        } else {
            transform[i] = 0;
        }
    }
    transform = F(padded_size)*transform;

    std::complex<double> roots[size];
    for (int i = 0; i<size; i++) {
        roots[i] = pow(omega,i);
    }
    for (int i = padded_size/4; i >= 1; i /= 2) {
        transform = M(roots, padded_size, i)*transform;
    }
    return transform;
}

int main() {
    int size = 400;
    double a = 0;
    double b = 1;
    double interval = b-a;
    double f = 120;
    std::complex<double> x[size];
    for (int i = 0; i < size; i++) {
        x[i] = cos(2*pi*f*(a+i*interval/(size-1)));
    }
    Eigen::VectorXcd transform = fft(x, size);
    std::complex<double> max = 0;
    int max_k = 0;
    for (int i = 0; i < size/2; i++) {
        if (abs(transform[i]) > abs(max)) {
            max = transform[i];
            max_k = i;
        }
    }
    std::cout << "The most significant component has a frequency of " << max_k*(size/pow(2,ceil(log2(size))))/(b-a) << std::endl;
    return 0;
}
