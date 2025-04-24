#include "shaders.h"

namespace Shaders 
{

WindowQuad windowQuad;

bool WindowQuad::compile() {
	auto source = getShaderSource();
	bool error = false;

	// compile vertex shader
	{
		ComPtr<ID3DBlob> errorBlob;

		if(FAILED(D3DCompile(
			source, 
			strlen(source), 
			__FUNCTION__ , 
			nullptr, nullptr, 
			"VS_Main", "vs_5_0", 0, 0, &vertexShaderBytecode, &errorBlob))) 
		{
			error = true;
			std::fprintf(stderr, "failed compilation of %s: %s\n", __FUNCTION__, (char*) errorBlob->GetBufferPointer());
		} else {
			if(FAILED(device->CreateVertexShader(
				vertexShaderBytecode->GetBufferPointer(), 
				vertexShaderBytecode->GetBufferSize(), nullptr, &vertexShader))) 
			{
				error = true;
				std::fprintf(stderr, "failed CreateVertexShader\n");
			}
		}
	}

	// compile pixel shader
	{
		ComPtr<ID3DBlob> pixelBlob;
		ComPtr<ID3DBlob> errorBlob;

		if(FAILED(D3DCompile(
			source, 
			strlen(source), 
			__FUNCTION__ , 
			nullptr, nullptr, 
			"PS_Main", "ps_5_0", 0, 0, &pixelBlob, &errorBlob))) 
		{
			error = true;
			std::fprintf(stderr, "failed compilation of %s: %s\n", __FUNCTION__, (char*) errorBlob->GetBufferPointer());
		} else {
			if(FAILED(device->CreatePixelShader(pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(), nullptr, &pixelShader))) 
			{
				error = true;
				std::fprintf(stderr, "failed CreateVertexShader\n");
			}
		}
	}

	// create cbuffers
	{
		WindowQuadData defaultData{
			.imageWidth = 1,
			.imageHeight = 1,
		};

		D3D11_BUFFER_DESC desc = {
			.ByteWidth = sizeof(WindowQuadData),
			.Usage = D3D11_USAGE_DYNAMIC,
			.BindFlags = D3D11_BIND_CONSTANT_BUFFER,
			.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE,
			.MiscFlags = 0,
			.StructureByteStride = 0,
		};

		D3D11_SUBRESOURCE_DATA initData = {
			.pSysMem = &defaultData,
			.SysMemPitch = 0,
			.SysMemSlicePitch = 0,
		};

		if(FAILED(device->CreateBuffer(&desc, &initData, &windowQuadDataBuffer))) {
			std::fprintf(stderr, "CreateBuffer failed\n");
		}
	}

	return !error;
}

void WindowQuad::setWindowQuadData(const WindowQuadData& data) {
	D3D11_MAPPED_SUBRESOURCE subresource{};
	if(!FAILED(deviceContext->Map(windowQuadDataBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &subresource))) {
		memcpy(subresource.pData, &data, sizeof(WindowQuadData));
		deviceContext->Unmap(windowQuadDataBuffer.Get(), 0);
	} else {
		std::fprintf(stderr, "WindowQuad::setWindowQuadData Map failed\n");
	}
}

const char* const WindowQuad::getShaderSource() {
	return R"(
		struct VSOut {
			float4 position: SV_Position;
			float2 uv: TEXCOORD0;
		};

		Texture2D<float4> outputTexture : register(t0);
		SamplerState texSampler : register(s0);

		cbuffer WindowQuadData : register(b0) {
			float imageWidth;
			float imageHeight;
			float screenWidth;
			float screenHeight;
		};

		VSOut VS_Main(uint vertexID: SV_VertexID, float4 position: SV_Position)
		{
			// Vertex positions in NDC (triangle list: 2 triangles = 6 vertices)
			float2 positions[6] = {
				float2(-1.0, -1.0), // Triangle 1
				float2( 1.0, -1.0),
				float2(-1.0,  1.0),

				float2( 1.0, -1.0), // Triangle 2
				float2( 1.0,  1.0),
				float2(-1.0,  1.0),
			};

			float2 uvs[6] = {
				float2(0, 1),
				float2(1, 1),
				float2(0, 0), 

				float2(1, 1),
				float2(1, 0),
				float2(0, 0),
			};

			VSOut vsout;
			float2 pos = positions[vertexID];
			vsout.position = float4(pos, 0, 1);
			vsout.uv = uvs[vertexID];

			return vsout;
		}

		float4 PS_Main(VSOut psin) : SV_Target
		{
			const float tiny = 0.00001;
			const float imageRatio = imageWidth / max(imageHeight, tiny);
			const float screenRatio = screenWidth / max(screenHeight, tiny);
			const float ratio = imageRatio / max(screenRatio, tiny);

			float2 uv;
			if(ratio > 1) {
				uv = float2(psin.uv.x, psin.uv.y * ratio);
			} else {
				uv = float2(psin.uv.x / ratio, psin.uv.y);
			}

			float4 texel = outputTexture.Sample(texSampler, uv);
			return texel;
		}
	)";
}

} // namespace Shaders