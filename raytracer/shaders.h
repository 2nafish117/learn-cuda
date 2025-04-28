#pragma once

#include <iostream>

#define NOMINMAX
#include <d3d11_1.h>
#include <wrl.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3dcommon.h>
#include <directxmath.h>

template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

extern ComPtr<ID3D11Device> device;
extern ComPtr<ID3D11DeviceContext> deviceContext;

namespace Shaders {

struct WindowQuad {
	bool compile();
	
	ComPtr<ID3D11VertexShader> vertexShader = nullptr;
	ComPtr<ID3DBlob> vertexShaderBytecode = nullptr;
	ComPtr<ID3D11PixelShader> pixelShader = nullptr;

	ComPtr<ID3D11Buffer> windowQuadDataBuffer = nullptr;

	struct alignas(16) WindowQuadData {
		float imageWidth;
		float imageHeight;
		float screenWidth;
		float screenHeight;
	};

	void setWindowQuadData(const WindowQuadData& data);

private:
	const char* const getShaderSource();
};
extern WindowQuad windowQuad;

} // namespace shaders