#include <common/common.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <imgui.h>
#include <backends/imgui_impl_dx11.h>
#include <backends/imgui_impl_glfw.h>

#define NOMINMAX
#include <d3d11_1.h>
#include <wrl.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <d3dcommon.h>
#include <directxmath.h>

#include "shaders.h"

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
 
template<typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

int windowWidth = WINDOW_WIDTH;
int windowHeight = WINDOW_HEIGHT;

char pickedGpuDescription[128]{};
LARGE_INTEGER pickedGpuDriverVersion{};

// d3d handles and states
ComPtr<IDXGIFactory1> factory;

ComPtr<IDXGISwapChain> swapchain;

ComPtr<ID3D11Device> device;
ComPtr<ID3D11DeviceContext> deviceContext;

ComPtr<ID3D11RenderTargetView> renderTargetView;

ComPtr<ID3D11InputLayout> inputLayout;
ComPtr<ID3D11RasterizerState> rasterState;

ComPtr<ID3D11SamplerState> samplerState;

ComPtr<ID3D11Texture2D> outputTexture;
ComPtr<ID3D11ShaderResourceView> outputTextureView;

// cuda mapped d3d11 resources 
struct CudaImage {
	// handle to the graphics resource
	cudaGraphicsResource_t texture;
	// intermediate to write into
	uint8_t* intermediate;
	int width, height;
};

CudaImage cudaOutputImage{};

void obtainSwapchainResources();
void enumAdapters(std::vector<ComPtr<IDXGIAdapter>>& outAdapters);
ComPtr<IDXGIAdapter> pickAdapter(const std::vector<ComPtr<IDXGIAdapter>>& adapters);
void setGraphicsPipelineState();

void createTextureAndView(int width, int height, ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11ShaderResourceView>& srv, bool isInput);
void releaseTextureAndView(ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11ShaderResourceView>& srv);

void dropCallback(GLFWwindow* window, int path_count, const char* paths[]) {
	std::printf("drop callback\n");
	
	for(int i = 0; i < path_count; ++i) {
		const char* path = paths[i];
		std::printf("%s\n", path);
	}

	if(path_count > 0) {
        
	}
}

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

	glfwSetDropCallback(window, dropCallback);

	if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
		std::fprintf(stderr, "CreateDXGIFactory1 failed\n");
		return -2;
	}

	std::vector<ComPtr<IDXGIAdapter>> adapters;
	enumAdapters(adapters);
	ComPtr<IDXGIAdapter> pickedAdapter = pickAdapter(adapters);
	if(pickedAdapter == nullptr) {
		return -3;
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

	UINT createFlags = D3D11_CREATE_DEVICE_DEBUG;

	if(FAILED(D3D11CreateDeviceAndSwapChain(
		pickedAdapter.Get(),
		D3D_DRIVER_TYPE_UNKNOWN, 
		nullptr,
		createFlags, 
		nullptr, 0, D3D11_SDK_VERSION, &swapchainDesc, &swapchain, &device, nullptr, &deviceContext))) 
	{
		std::fprintf(stderr, "D3D11CreateDeviceAndSwapChain failed\n");
		return -4;
	}

	bool res = Shaders::windowQuad.compile();
	if(!res) {
		__debugbreak();
	}

	obtainSwapchainResources();
	setGraphicsPipelineState();

	D3D11_SAMPLER_DESC samplerDesc = {
		.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT,
		.AddressU = D3D11_TEXTURE_ADDRESS_BORDER,
		.AddressV = D3D11_TEXTURE_ADDRESS_BORDER,
		.AddressW = D3D11_TEXTURE_ADDRESS_BORDER,
		.MipLODBias = 0,
		.MaxAnisotropy = 1,
		.ComparisonFunc = D3D11_COMPARISON_NEVER,
		.BorderColor = {0.2, 0.2, 0.2, 1},
		.MinLOD = 0,
		.MaxLOD = D3D11_FLOAT32_MAX,
	};

	if(FAILED(device->CreateSamplerState(&samplerDesc, &samplerState))) {
		std::fprintf(stderr, "CreateSamplerState failed");
	}

	createTextureAndView(windowWidth, windowHeight, outputTexture, outputTextureView, false);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // IF using Docking Branch

	// imgui font
	{
		ImFontConfig font_config = {};

		font_config.FontDataOwnedByAtlas = true;
		font_config.OversampleH = 6;
		font_config.OversampleV = 6;
		font_config.GlyphMaxAdvanceX = FLT_MAX;
		font_config.RasterizerMultiply = 1.4f;
		font_config.RasterizerDensity = 1.0f;
		font_config.EllipsisChar = UINT16_MAX;

		font_config.PixelSnapH = false;
		font_config.GlyphOffset = ImVec2{0.0, -1.0};

		io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 16.0f, &font_config);

		font_config.MergeMode = true;

		const uint16_t ICON_MIN_FA = 0xe005;
		const uint16_t ICON_MAX_FA = 0xf8ff;

		static uint16_t FA_RANGES[3] = {ICON_MIN_FA, ICON_MAX_FA, 0};

		font_config.RasterizerMultiply = 1.0;
		font_config.GlyphOffset = ImVec2{0.0, -1.0};

		font_config.MergeMode = false;
	}

	ImGui_ImplGlfw_InitForOther(window, true);
	ImGui_ImplDX11_Init(device.Get(), deviceContext.Get());

	while (!glfwWindowShouldClose(window))
	{
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
		ImGui::ShowDemoWindow();

        // raytrace here

		// 1. raytace into scene (bunch of primitives) from camera (dynamic camera)
		// 2. write into a cudaTexture maybe, or just in a buffer
		// 3. copy buffer into d3d11 output texture when we need to 

		const FLOAT clearColor[] = {0.0f, 1.0f, 0.0f, 1.0f};

		deviceContext->ClearRenderTargetView(renderTargetView.Get(), clearColor);

		// draw window quad
		auto vertShader = Shaders::windowQuad.vertexShader.Get();

		deviceContext->VSSetShader(Shaders::windowQuad.vertexShader.Get(), nullptr, 0);

		deviceContext->PSSetShader(Shaders::windowQuad.pixelShader.Get(), nullptr, 0);
		deviceContext->PSSetConstantBuffers(0, 1, Shaders::windowQuad.windowQuadDataBuffer.GetAddressOf());
		deviceContext->PSSetShaderResources(0, 1, outputTextureView.GetAddressOf());
		deviceContext->PSSetSamplers(0, 1, samplerState.GetAddressOf());
		
		deviceContext->OMSetRenderTargets(1, renderTargetView.GetAddressOf(), nullptr);
		
		deviceContext->Draw(6, 0);

		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

		// present swapchain
		if(FAILED(swapchain->Present(1, 0))) {
			std::fprintf(stderr, "failed swapchain present\n");
		}

		glfwPollEvents();
	}

	ImGui_ImplDX11_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

void enumAdapters(std::vector<ComPtr<IDXGIAdapter>>& outAdapters) {
	// hard upper bound 512, who has more than 512 gpus?
	for (int i = 0; i < 512; ++i)
	{
		ComPtr<IDXGIAdapter> adapter;
		HRESULT res = factory->EnumAdapters(i, &adapter);

		if (res == DXGI_ERROR_NOT_FOUND) {
			break;
		}

		outAdapters.push_back(adapter);
	}
}

ComPtr<IDXGIAdapter> pickAdapter(const std::vector<ComPtr<IDXGIAdapter>>& adapters) {
	std::printf("trying to pick a cuda compatible adapter\n");

	assert(adapters.size() > 0 && "");

	bool foundCudaCompatibleDevice = false;
	int cudaCompatibleDevice{};
	ComPtr<IDXGIAdapter> cudaCompatibleAdapter;
	
	for (auto& adapter : adapters) {
		DXGI_ADAPTER_DESC desc{};
		if (FAILED(adapter->GetDesc(&desc))) {
			std::fprintf(stderr, "error GetDesc on adapter\n");
		}

		memset(pickedGpuDescription, 0, 128);
		wcstombs_s(nullptr, pickedGpuDescription, desc.Description, 128);

		std::printf(
			"[DXGI_ADAPTER_DESC1 Description=%s VendorId=%d DeviceId=%d SubSysId=%d Revision=%d DedicatedVideoMemory=%lld DedicatedSystemMemory=%lld SharedSystemMemory=%lld]\n",
			pickedGpuDescription,
			desc.VendorId,
			desc.DeviceId,
			desc.SubSysId,
			desc.Revision,
			desc.DedicatedVideoMemory,
			desc.DedicatedSystemMemory,
			desc.SharedSystemMemory
		);

		int device{};
		if(cudaError_t err = cudaD3D11GetDevice(&device, adapter.Get()); err == cudaSuccess) {
			std::printf("found cuda compatible adapter\n");
			foundCudaCompatibleDevice = true;
			cudaCompatibleDevice = device;
			cudaCompatibleAdapter = adapter;
			break;
		} else {
			std::printf("this device is not cuda compatible, looking for more...\n");
		}
	}

	if(!foundCudaCompatibleDevice) {
		std::fprintf(stderr, "no available cuda compatible adapter found\n");
	}

	if(FAILED(cudaCompatibleAdapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &pickedGpuDriverVersion))) {
		std::printf("CheckInterfaceSupport for adapter failed");
	}
	
	return cudaCompatibleAdapter;
}

void obtainSwapchainResources() {
	ComPtr<ID3D11Texture2D> backBuffer;
	if (FAILED(swapchain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
		std::fprintf(stderr, "failed GetBuffer on swapchain\n");
	}

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

	// @NOTE: we dont need to use these states 
	// D3D11_DEPTH_STENCIL_DESC depthStencilDesc;
	// D3D11_BLEND_DESC blendDesc;
	// device->CreateDepthStencilState()
	// device->CreateBlendState()

	deviceContext->RSSetState(rasterState.Get());

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

void createTextureAndView(int width, int height, ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11ShaderResourceView>& srv, bool isInput) {
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
		.BindFlags = D3D11_BIND_SHADER_RESOURCE,
		.MiscFlags = 0,
	};

	// @TODO: make sure texture creation desc is set up correctly for raytracing

	if(isInput) {
		textureDesc.Usage = D3D11_USAGE_DYNAMIC;
		textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	} else {
		textureDesc.Usage = D3D11_USAGE_DEFAULT;
		textureDesc.CPUAccessFlags = 0;
	}

	if(FAILED(device->CreateTexture2D(&textureDesc, nullptr, &texture))) {
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
	
	if(FAILED(device->CreateShaderResourceView(texture.Get(), &srvDesc, &srv))) {
		std::fprintf(stderr, "CreateShaderResourceView failed");
	}
}

void releaseTextureAndView(ComPtr<ID3D11Texture2D>& texture, ComPtr<ID3D11ShaderResourceView>& srv) {
	texture.Reset();
	srv.Reset();
}