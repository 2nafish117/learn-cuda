#pragma once

#include <string_view>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class ScopedTimer final {
public:
	ScopedTimer(std::string_view name);
	~ScopedTimer();

	inline double Elapsed() {
		using namespace std::chrono;
		using namespace std::chrono_literals;

		m_stop = high_resolution_clock::now();
		duration<double> duration = m_stop - m_start;
		return duration.count();
	}
	
private:
	std::string_view m_name;
	std::chrono::high_resolution_clock::time_point m_start{};
	std::chrono::high_resolution_clock::time_point m_stop{};
};

#define SCOPED_TIMER(name) ScopedTimer hidden_scopedTimer(name)

#define ASSET_PATH(path) ("../../../assets/" ## path)

void cudaErrorPrint(cudaError_t err);

#define CUDA_CHECK(expr) {			\
	cudaError_t err = (expr);		\
	cudaErrorPrint(err);			\
	cudaDeviceSynchronize();		\
}