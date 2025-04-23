#include <common/common.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define NOMINMAX
#include <d3d11_1.h>
#include <wrl.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3dcommon.h>
#include <directxmath.h>

#include "effects.h"
#include "shaders.h"

#include <iostream>
#include <chrono>
#include <random>
#include <cassert>
 
template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

int windowWidth = WINDOW_WIDTH;
int windowHeight = WINDOW_HEIGHT;

// d3d handles and states
ComPtr<IDXGISwapChain> swapchain;

ComPtr<ID3D11Device> device;
ComPtr<ID3D11DeviceContext> deviceContext;

ComPtr<ID3D11RenderTargetView> renderTargetView;

ComPtr<ID3D11InputLayout> inputLayout;
ComPtr<ID3D11RasterizerState> rasterState;

ComPtr<ID3D11Texture2D> outputTexture;
ComPtr<ID3D11ShaderResourceView> outputTextureView;
ComPtr<ID3D11SamplerState> samplerState;

void obtainSwapchainResources();
void setGraphicsPipelineState();
void createOutputTexture(int width, int height);

int main(void)
{
	if (!glfwInit()) {
		return -1;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "image effects with cuda", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	DXGI_SWAP_CHAIN_DESC swapchainDesc = {
		.BufferDesc = DXGI_MODE_DESC {
			.Width = (UINT) windowWidth,
			.Height = (UINT) windowHeight,
			.RefreshRate = DXGI_RATIONAL {
				.Numerator = 0,
				.Denominator = 0,
			},
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
			.Scaling = DXGI_MODE_SCALING_UNSPECIFIED,
		},
		.SampleDesc = DXGI_SAMPLE_DESC {
			.Count = 1,
			.Quality = 0,
		},
		.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
		.BufferCount = 3,
		.OutputWindow = glfwGetWin32Window(window),
		.Windowed = true,
		.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
		.Flags = 0,
	};

	if(FAILED(D3D11CreateDeviceAndSwapChain(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE, 
		nullptr,
		D3D11_CREATE_DEVICE_DEBUG, 
		nullptr, 0, D3D11_SDK_VERSION, &swapchainDesc, &swapchain, &device, nullptr, &deviceContext))) 
	{
		std::fprintf(stderr, "D3D11CreateDeviceAndSwapChain failed\n");
	}

	bool res = Shaders::windowQuad.compile();
	if(!res) {
		__debugbreak();
	}

	obtainSwapchainResources();
	setGraphicsPipelineState();

	createOutputTexture(windowWidth, windowHeight);

	while (!glfwWindowShouldClose(window))
	{
		const FLOAT clearColor[] = {0.0f, 1.0f, 0.0f, 1.0f};

		deviceContext->ClearRenderTargetView(renderTargetView.Get(), clearColor);

		// draw window quad
		auto vertShader = Shaders::windowQuad.vertexShader.Get();

		deviceContext->VSSetShader(Shaders::windowQuad.vertexShader.Get(), nullptr, 0);

		deviceContext->PSSetShader(Shaders::windowQuad.pixelShader.Get(), nullptr, 0);
		deviceContext->PSSetShaderResources(0, 1, outputTextureView.GetAddressOf());
		deviceContext->PSSetSamplers(0, 1, samplerState.GetAddressOf());
		
		deviceContext->OMSetRenderTargets(1, renderTargetView.GetAddressOf(), nullptr);
		
		deviceContext->Draw(6, 0);

		/* Swap front and back buffers */
		if(FAILED(swapchain->Present(1, 0))) {
			std::fprintf(stderr, "failed swapchain present\n");
		}

		#if 0
		DXGI_FRAME_STATISTICS stats{};
		if(FAILED(swapchain->GetFrameStatistics(&stats))) {
			std::fprintf(stderr, "GetFrameStatistics failed\n");
		}

		std::printf("frame stats: \n");
		std::printf("present count: %d, present refresh count: %d, sync refresh count: %d\n", 
			stats.PresentCount, stats.PresentRefreshCount, stats.SyncRefreshCount);
		std::printf("sync qpc time: %lld, sync gpu time: %lld\n", stats.SyncQPCTime.QuadPart, stats.SyncGPUTime.QuadPart);
		#endif

		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

// int main() {
// 	int width, height, channels;

// 	const int desired_channels = 3;
// 	uint8_t* data = stbi_load(ASSET_PATH("/guy_behind_car.jpg"), &width, &height, &channels, desired_channels);

// 	Effects::blur_img(data, width, height, desired_channels, 32);
// 	stbi_write_jpg(ASSET_PATH("/blurred_guy_behind_car.jpg"), width, height, desired_channels, data, 100);

// 	stbi_image_free(data);

// 	return 0;
// }

void obtainSwapchainResources() {
	ComPtr<ID3D11Texture2D> backBuffer;
	if (FAILED(swapchain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
		std::fprintf(stderr, "failed GetBuffer on swapchain\n");
	}
	assert(backBuffer.Get());

	std::printf("obtained buffer from swapchain\n");

	D3D11_RENDER_TARGET_VIEW_DESC rtDesc = {
		.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
		.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D,
		.Texture2D = {
			.MipSlice = 0,
		}
	};

	if (FAILED(device->CreateRenderTargetView(backBuffer.Get(), &rtDesc, &renderTargetView))) {
		std::fprintf(stderr, "failed CreateRenderTargetView on swapchain buffer\n");
	}

	std::printf("created render target view\n");
}

void setGraphicsPipelineState() {
	
	D3D11_INPUT_ELEMENT_DESC layoutDescs[] = {
		{
			.SemanticName = "SV_Position",
			.SemanticIndex = 0,
			.Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
			.InputSlot = 0,
			.AlignedByteOffset = 0,
			.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA,
			.InstanceDataStepRate = 0,
		},
	};

	const auto& bytecode = Shaders::windowQuad.vertexShaderBytecode;
	if(FAILED(device->CreateInputLayout(layoutDescs, 1, bytecode->GetBufferPointer(), bytecode->GetBufferSize(), &inputLayout))) {
		std::fprintf(stderr, "CreateInputLayout failed\n");
	}

	deviceContext->IASetInputLayout(inputLayout.Get());
	deviceContext->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	D3D11_RASTERIZER_DESC rasterDesc = {
		.FillMode = D3D11_FILL_SOLID,
		.CullMode = D3D11_CULL_BACK,
		.FrontCounterClockwise = true,
		.DepthBias = 0,
		.DepthBiasClamp = 0,
		.SlopeScaledDepthBias = 0,
		.DepthClipEnable = true,
		.ScissorEnable = false,
		.MultisampleEnable = false,
		.AntialiasedLineEnable = false,
	};
	
	if(FAILED(device->CreateRasterizerState(&rasterDesc, &rasterState))) {
		std::fprintf(stderr, "CreateRasterizerState failed\n");
	}

	// we dont need to use these states 
	// D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
	// D3D11_BLEND_DESC blendDesc;
	// device->CreateDepthStencilState()
	// device->CreateBlendState()

	deviceContext->RSSetState(rasterState.Get());
	// const ID3D11RenderTargetView* renderTargetViews[] = {
	// 	(const ID3D11RenderTargetView* const) renderTargetView.Get(),
	// };

	D3D11_VIEWPORT viewport = {
		.TopLeftX = 0,
		.TopLeftY = 0,
		.Width = (float) windowWidth,
		.Height = (float) windowHeight,
		.MinDepth = 0,
		.MaxDepth = 1,
	};
	deviceContext->RSSetViewports(1, &viewport);
}

void createOutputTexture(int width, int height) {
	D3D11_TEXTURE2D_DESC textureDesc = {
		.Width = (UINT) width,
		.Height = (UINT) height,
		.MipLevels = 1,
		.ArraySize = 1,
		.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
		.SampleDesc = DXGI_SAMPLE_DESC {
			.Count = 1,
			.Quality = 0,
		},
		.Usage = D3D11_USAGE_DEFAULT,
		.BindFlags = D3D11_BIND_SHADER_RESOURCE,
		.CPUAccessFlags = 0,
		.MiscFlags = 0,
	};

	if(FAILED(device->CreateTexture2D(&textureDesc, nullptr, &outputTexture))) {
		std::fprintf(stderr, "CreateTexture2D failed");
	}
	
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {
		.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
		.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D,
		.Texture2D = D3D11_TEX2D_SRV {
			.MostDetailedMip = 0,
			.MipLevels = 1,
		},
	};
	
	if(FAILED(device->CreateShaderResourceView(outputTexture.Get(), &srvDesc, &outputTextureView))) {
		std::fprintf(stderr, "CreateShaderResourceView failed");
	} 

	D3D11_SAMPLER_DESC samplerDesc = {
		.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT,
		.AddressU = D3D11_TEXTURE_ADDRESS_WRAP,
		.AddressV = D3D11_TEXTURE_ADDRESS_WRAP,
		.AddressW = D3D11_TEXTURE_ADDRESS_WRAP,
		.MipLODBias = 0,
		.MaxAnisotropy = 1,
		.ComparisonFunc = D3D11_COMPARISON_NEVER,
		.BorderColor = {0, 0, 0, 0},
		.MinLOD = 0,
		.MaxLOD = D3D11_FLOAT32_MAX,
	};

	if(FAILED(device->CreateSamplerState(&samplerDesc, &samplerState))) {
		std::fprintf(stderr, "CreateSamplerState failed");
	}
}