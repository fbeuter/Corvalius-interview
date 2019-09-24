#pragma once

#ifdef __cplusplus
extern "C" {
#endif

	void __declspec(dllexport) __cdecl sobelFilterCUDA(unsigned char* inputImageHost, unsigned char* outputImageHost, int imageWidth, int imageHeight);

#ifdef __cplusplus
}
#endif