#include "common.ch"

// #include <spdlog/spdlog.h>

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

	m_stop = high_resolution_clock::now();
	std::printf("%s elapsed %f", m_name.data(), Elapsed());
	// spdlog::info("{} elapsed {}", m_name, Elapsed());
}

void cudaErrorPrint(cudaError_t err) {
	if(err != cudaSuccess) {
		const char* errStr = cudaGetErrorString(err);
		const char* errName = cudaGetErrorName(err);
		std::printf("[cuda error %d] %s %s", (uint32_t) err, errName, errStr);
		// spdlog::error("[cuda error {}] {} {}", (uint32_t) err, errName, errStr);
	}
}