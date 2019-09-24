#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include "math.h"

#include <stdio.h>

__global__ void sobelCUDA(unsigned char* inputImage, unsigned char* outputImage, int imageWidth, int imageHeight) {

	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = pixelX + pixelY * imageWidth;

	char hKernel[9] = { -1, 0, 1,
						-2, 0, 2,
						-1, 0, 1 };

	char vKernel[9] = { -1, -2, -1,
						 0,  0,  0,
						 1,  2,  1 };

	if (((pixelX > 0) && (pixelX < imageWidth-1)) && ((pixelY > 0) && (pixelY < imageHeight-1))) {

		// Magnitudes
		int magXR = 0, magXG = 0, magXB = 0;
		int magYR = 0, magYG = 0, magYB = 0;

		// Horizontal and Vertical convolution at point (x,y)
		for (int h = 0; h < 3; h++)
		{
			for (int v = 0; v < 3; v++)
			{
				// Current pixel in 3x3 convolution window
				int xn = pixelX + (h - 1);
				int yn = pixelY + (v - 1);

				int inputPixel = (xn + yn * imageWidth) * 4;
				// Horizontal Convolution
				int hKernelValue = hKernel[h*3 + v];
				magXR += inputImage[inputPixel] * hKernelValue;
				magXG += inputImage[inputPixel+1] * hKernelValue;
				magXB += inputImage[inputPixel+2] * hKernelValue;
				// Vertical Convolution
				int vKernelValue = vKernel[h*3 + v];
				magYR += inputImage[inputPixel] * vKernelValue;
				magYG += inputImage[inputPixel+1] * vKernelValue;
				magYB += inputImage[inputPixel+2] * vKernelValue;
			}
		}

		// Compute final pixel value
		// We clip the value to 255 in case we go over the 8-bit range
		// Instead of using norm, we apprximate using sum of abs, this
		// has been show to be as effective in this application
		int finalR = min(abs(magXR) + abs(magYR), 255);
		int finalG = min(abs(magXG) + abs(magYG), 255);
		int finalB = min(abs(magXB) + abs(magYB), 255);

		outputImage[offset * 4] = finalR;
		outputImage[offset * 4 + 1] = finalG;
		outputImage[offset * 4 + 2] = finalB;
		outputImage[offset * 4 + 3] = 255;
	}
}

void __declspec(dllexport) __cdecl sobelFilterCUDA(unsigned char* inputImageHost, unsigned char* outputImageHost, int imageWidth, int imageHeight) {

	unsigned char* inputImageDevice;
	unsigned char* outputImageDevice;
	int imageSize = imageWidth * imageHeight * 4 * sizeof(unsigned char);

	cudaMalloc((void**)&inputImageDevice, imageSize);
	cudaMalloc((void**)&outputImageDevice, imageSize);
	cudaMemcpy(inputImageDevice, inputImageHost, imageSize, cudaMemcpyHostToDevice);

	dim3 blockDims(16, 16);
	dim3 gridDims((unsigned int)ceil(((double)imageWidth  / blockDims.x)),
				  (unsigned int)ceil(((double)imageHeight / blockDims.y)));
	
	sobelCUDA<<<gridDims, blockDims>>>(inputImageDevice, outputImageDevice, imageWidth, imageHeight);

	cudaMemcpy(outputImageHost, outputImageDevice, imageSize, cudaMemcpyDeviceToHost);
	cudaFree(inputImageDevice);
	cudaFree(outputImageDevice);
}