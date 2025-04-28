#include "common.h"

ScopedTimer::ScopedTimer(std::string_view name) 
	: m_name(name)
{
	using namespace std::chrono;
	using namespace std::chrono_literals;

	m_start = high_resolution_clock::now();
}

ScopedTimer::~ScopedTimer() {
	using namespace std::chrono;
	using namespace std::chrono_literals;

	std::printf("%s elapsed %.3fms\n", m_name.data(), 1000 * Elapsed());
}

void cudaErrorPrint(cudaError_t err) {
	if(err != cudaSuccess) {
		const char* errStr = cudaGetErrorString(err);
		const char* errName = cudaGetErrorName(err);
		std::fprintf(stderr, "[cuda error %d] %s %s\n", (uint32_t) err, errName, errStr);
		std::fflush(stdout);
		std::fflush(stderr);
		cudaDeviceSynchronize();
	}
}