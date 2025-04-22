#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <random>
#include <cassert>

#include <clog/log.h>
#include <stb/stb_image.h>

class ScopedTimer final {
public:
	ScopedTimer(std::string_view name) 
		: m_name(name)
	{
		using namespace std::chrono;
		using namespace std::chrono_literals;

		m_start = high_resolution_clock::now();
	}

	~ScopedTimer() {
		using namespace std::chrono;
		using namespace std::chrono_literals;

		m_stop = high_resolution_clock::now();
		log_info("%s elapsed %f\n", m_name.data(), Elapsed());
	}

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

void cudaErrorPrint(cudaError_t err) {
	if(err != cudaSuccess) {
		const char* errStr = cudaGetErrorString(err);
		const char* errName = cudaGetErrorName(err);
		log_error("[cuda error %d] %s %s\n", err, errName, errStr);
	}
}

#define CUDA_CHECK(expr) {			\
	cudaError_t err = (expr);		\
	cudaErrorPrint(err);			\
}

int main() {
	return 0;
}