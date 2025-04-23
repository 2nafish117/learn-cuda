#pragma once

#include <string_view>
#include <chrono>

class ScopedTimer final {
public:
	ScopedTimer(std::string_view name);
	~ScopedTimer();

private:
	inline double Elapsed() {
		using namespace std::chrono;
		using namespace std::chrono_literals;

		duration<double> duration = m_stop - m_start;
		return duration.count();
	}

	std::string_view m_name;
	std::chrono::high_resolution_clock::time_point m_start{};
	std::chrono::high_resolution_clock::time_point m_stop{};
};

#define SCOPED_TIMER(name) ScopedTimer hidden_scopedTimer(name)

#define ASSET_PATH(path) ("../../../assets/" ## path)

#if __CUDACC__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void cudaErrorPrint(cudaError_t err);

#define CUDA_CHECK(expr) {			\
	cudaError_t err = (expr);		\
	cudaErrorPrint(err);			\
}
#endif