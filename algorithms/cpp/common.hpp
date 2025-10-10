#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <assert.h>
#include <iostream>
namespace CS {
	template<typename T = double>
	struct Matrix {
		using value_type = T;
		size_t rows_ = 0;
		size_t cols_ = 0;
		std::vector<T> data_;
		Matrix() = default;
		Matrix(size_t rows, size_t cols, T init_val = T(0)){
			resize(rows, cols, init_val);
		}

	
		Matrix(size_t rows, size_t cols, const std::vector<T>& vec) {
			assert(vec.size() == rows * cols);
			rows_ = rows; cols_ = cols;
			data_ = vec;
		}
		inline size_t rows() const noexcept { return rows_; }
		inline size_t cols() const noexcept { return cols_; }
		inline const T* data() const noexcept { return data_.data(); }
		inline T* data() noexcept { return data_.data(); }

		inline bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

		inline void resize(size_t rows, size_t cols, T init_val = T(0)){
			rows_ = rows;
			cols_ = cols;
			data_.assign(rows * cols, init_val);
		}
		inline void fill(T v){
			std::fill(data_.begin(), data_.end(), v);
		}
		inline void setZero(){
			fill(T(0));
		}
		inline T& at(size_t i, size_t j) {
			assert(i < rows_ && j < cols_);
			return data_[i * cols_ + j];
		}
		T& operator()(size_t i, size_t j) {
    		return data_[i * cols_ + j];
		}
		const T& operator()(size_t i, size_t j) const {
    		return data_[i * cols_ + j];
		}		
		void printMatrix() {
			std::cout << "Matrix(" << rows_ << "x" << cols_ << ")" << std::endl;
			for (size_t i = 0; i < rows_; ++i) {
				for (size_t j = 0; j < cols_; ++j) {
					std::cout << at(i,j) << "\t";
				}
				std::cout << "\n";
			}
		}
		static Matrix<T> randomGaussian(size_t rows, size_t cols,
		unsigned long seed = std::random_device{}(), T sigma = 1, 
		bool normalize_by_sqrt_rows = false){
			Matrix<T> M(rows, cols);
			std::mt19937_64 rng(seed);
			std::normal_distribution<T> nd(T(0), sigma);
			for (size_t i = 0; i < rows; i++) {
		        for (size_t j = 0; j < cols; j++) {
		            M.at(i, j) = nd(rng);
		        }
    		}
			for(size_t i = 0; i < rows * cols; i++){
				if(normalize_by_sqrt_rows && rows > 0){
					T s= std::sqrt((T)rows);
					for(auto &v: M.data_) v /= s;
				}
			}
		return M;

		}

	};
}