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

	return !error;
}

const char* const WindowQuad::getShaderSource() {
	return R"(
		struct VSOut {
			float4 position: SV_Position;
			float2 uv: TEXCOORD0;
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

			VSOut vsout;
			float2 pos = positions[vertexID];
			vsout.position = float4(pos, 0, 1);
			vsout.uv = pos * 0.5 + 0.5;

			return vsout;
		}

		float4 PS_Main(VSOut psin) : SV_Target
		{
			return float4(psin.uv, 0, 1);
		}
	)";
}

} // namespace Shaders