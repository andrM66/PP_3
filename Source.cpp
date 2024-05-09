#include <iostream>
#include <vector>
#include<fstream>
#include <random>
#include <time.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
int ProcNum, ProcRank;
int* first = nullptr;
int* second = nullptr;
int* res = nullptr;
class SquareMatrix
{
	vector<vector<int>> _data;
	size_t _size;
public:
	friend ostream& operator<<(ostream& s, SquareMatrix mat);
	SquareMatrix(size_t size, bool null_flag)
	{
		_size = size;
		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<> dist(-100, 100);
		for (size_t i = 0; i < size; ++i)
		{
			vector<int> tmp;
			for (size_t j = 0; j < size; ++j)
			{
				if (!null_flag)
				{
					tmp.push_back(dist(gen));
				}
				else
				{
					tmp.push_back(0);
				}
			}
			_data.push_back(tmp);
		}
	}
	SquareMatrix(const SquareMatrix& mat)
	{
		_size = mat._size;
		for (size_t i = 0; i < _size; ++i)
		{
			vector<int> tmp;
			for (size_t j = 0; j < _size; ++j)
			{
				tmp.push_back(mat._data[i][j]);
			}
			_data.push_back(tmp);
		}
	}
	SquareMatrix& operator=(const SquareMatrix& mat)
	{
		_data.resize(mat._size);
		_size = mat._size;
		if (_size == 0)
		{
			return *this;
		}
		for (size_t i = 0; i < _size; ++i)
		{
			for (size_t j = 0; j < _size; ++j)
			{
				_data[i][j] = mat._data[i][j];
			}
		}
		return *this;
	}
	/*static vector<SquareMatrix> read_matrixes(size_t size, size_t number_of_mat, string path)
	{
		vector<SquareMatrix> result;
		ifstream fin;
		fin.open(path);
		if (!fin)
		{
			throw(logic_error("file does not exist"));
		}
		for (size_t k = 0; k < number_of_mat; ++k)
		{
			SquareMatrix tmp(size, true);
			for (size_t i = 0; i < tmp._size; ++i)
			{
				for (size_t j = 0; j < tmp._size; ++j)
				{
					fin >> tmp._data[i][j];
				}
			}
			result.push_back(tmp);
		}
		fin.close();
		return result;
	}*/
	static void write_matrixes(const vector<SquareMatrix>& arr, string path)
	{
		ofstream of;
		of.open(path);
		for (size_t k = 0; k < arr.size(); ++k)
		{
			for (size_t i = 0; i < arr[k]._size; ++i)
			{
				for (size_t j = 0; j < arr[k]._size; ++j)
				{
					of << arr[k]._data[i][j];
					if (j == arr[k]._size - 1)
					{
						of << "\n";
					}
					else
					{
						of << " ";
					}
				}
			}
		}
		of.close();
	}
	static vector<SquareMatrix> make_array(size_t size, size_t number)
	{
		vector<SquareMatrix> result;
		for (size_t i = 0; i < number; ++i)
		{
			SquareMatrix tmp(size, false);
			result.push_back(tmp);
		}
		return result;
	}
	int* matrix_to_array(int* arr) const
	{
		if (arr)
		{
			delete[]arr;
			arr = nullptr;
		}
		arr = new int[_size * _size];
		for (size_t i = 0; i < _size; ++i)
		{
			for (size_t j = 0; j < _size; ++j)
			{
				arr[j + i * _size] = _data[i][j];
			}
		}
		return arr;
	}
	static SquareMatrix array_to_matrix(int* arr, int size)
	{
		SquareMatrix tmp(size, 1);
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t j = 0; j < size; ++j)
			{
				tmp._data[i][j] = arr[j + i * size];
			}
		}
		delete[]arr;
		return tmp;
	}
};
ostream& operator<<(ostream& s, SquareMatrix mat)
{
	for (size_t i = 0; i < mat._size; ++i)
	{
		for (size_t j = 0; j < mat._size; ++j)
		{
			s << mat._data[i][j] << " ";
		}
		s << "\n";
	}
	return s;
}
void init_process()
{
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
}
void write_time_stat(vector<double> arr, string path)
{
	ofstream of;
	of.open(path);
	for (size_t i = 0; i < arr.size(); ++i)
	{
		of << arr[i] << " ";
	}
	of.close();
}
void Flip(int*& B, int dim)
{
	int temp = 0;
	for (int i = 0; i < dim; i++) {
		for (int j = i + 1; j < dim; j++) {
			temp = B[i * dim + j];
			B[i * dim + j] = B[j * dim + i];
			B[j * dim + i] = temp;
		}
	}
}
void MatrixMultiplicationMPI(int*& A, int*& B, int*& C, int& Size)
{
	int dim = Size;
	int i, j, k, p, ind;
	int temp;
	MPI_Status Status;
	int ProcPartSize = dim / ProcNum;
	int ProcPartElem = ProcPartSize * dim;
	int* bufA = new int[dim * ProcPartSize];
	int* bufB = new int[dim * ProcPartSize];
	int* bufC = new int[dim * ProcPartSize];
	int ProcPart = dim / ProcNum, part = ProcPart * dim;
	if (ProcRank == 0) {
		Flip(B, Size);
	}

	MPI_Scatter(A, part, MPI_INT, bufA, part, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, part, MPI_INT, bufB, part, MPI_INT, 0, MPI_COMM_WORLD);

	temp = 0;
	for (i = 0; i < ProcPartSize; i++) {
		for (j = 0; j < ProcPartSize; j++) {
			for (k = 0; k < dim; k++) temp += bufA[i * dim + k] * bufB[j * dim + k];
			bufC[i * dim + j + ProcPartSize * ProcRank] = temp; temp = 0.0;
		}
	}

	int NextProc; int PrevProc;
	for (p = 1; p < ProcNum; p++) {
		NextProc = ProcRank + 1;
		if (ProcRank == ProcNum - 1) NextProc = 0;
		PrevProc = ProcRank - 1;
		if (ProcRank == 0) PrevProc = ProcNum - 1;
		MPI_Sendrecv_replace(bufB, part, MPI_INT, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
		temp = 0.0;
		for (i = 0; i < ProcPartSize; i++) {
			for (j = 0; j < ProcPartSize; j++) {
				for (k = 0; k < dim; k++) {
					temp += bufA[i * dim + k] * bufB[j * dim + k];
				}
				if (ProcRank - p >= 0)
					ind = ProcRank - p;
				else ind = (ProcNum - p + ProcRank);
				bufC[i * dim + j + ind * ProcPartSize] = temp;
				temp = 0.0;
			}
		}
	}
	MPI_Gather(bufC, ProcPartElem, MPI_INT, C, ProcPartElem, MPI_INT, 0, MPI_COMM_WORLD);

	delete[]bufA;
	delete[]bufB;
	delete[]bufC;
}
vector<int> A = { 100, 200, 400, 1000 };
size_t NUMBER = 100;
vector<string> PATH = { "matrix_list_100.txt", "matrix_list_200.txt", "matrix_list_400.txt", "matrix_list_1000.txt" };
vector<string> SAVE_PATH = { "result100.txt", "result200.txt", "result400.txt", "result1000.txt" };
vector<string> TIME_RESULT = { "time_res100.txt", "time_res200.txt", "time_res400.txt", "time_res1000.txt" };
int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	init_process();
	vector<vector<double>> time;
	for (size_t i = 0; i < A.size(); ++i)
	{
		vector<SquareMatrix> arr;
		arr = SquareMatrix::make_array(A[i], NUMBER);
		vector<SquareMatrix> result;
		time.push_back(vector<double> {});
		for (size_t j = 0; j < arr.size() - 1; ++j)
		{
			first = arr[j].matrix_to_array(first);
			second = arr[j + 1].matrix_to_array(second);
			res = new int[A[i] * A[i]];
			clock_t start = clock();
			MatrixMultiplicationMPI(first, second, res, A[i]);
			clock_t end = clock();
			result.push_back(SquareMatrix::array_to_matrix(res, A[i]));
			double seconds = (double)(end - start) / CLOCKS_PER_SEC;
			time[i].push_back(seconds);
		}
		if (ProcRank ==0)
		{
			SquareMatrix::write_matrixes(result, SAVE_PATH[i]);
			write_time_stat(time[i], TIME_RESULT[i]);
		}
	}
	MPI_Finalize();
	delete[]first;
	delete[]second;
	delete[]res;
	return 0;
}