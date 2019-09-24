// openCL-Filter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include "openCL-Filter.h"
// The device I'm currently using doesn't suppot OpenCL 1.2
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>

void checkError(cl_int error)
{
	if (error != CL_SUCCESS) {
		printf("OpenCL call failed with error %d\n", error);
	}
}

void __declspec(dllexport) __cdecl sobelFilterOpenCL(unsigned char* inputImageHost, unsigned char* outputImageHost, int imageWidth, int imageHeight) {

	cl_int error = CL_SUCCESS;

	// We get all the OpenCL platforms on the system
	// First we get the count of platforms
	cl_uint numberOfPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numberOfPlatforms);

	// Get the actual platforms
	cl_platform_id* platforms = (cl_platform_id*) malloc(numberOfPlatforms * sizeof(cl_platform_id));
	clGetPlatformIDs(numberOfPlatforms, platforms, NULL);

	// Get number of devices on the first platform
	cl_uint numberOfDevices = 0;
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numberOfDevices);
	
	// Get the actual devices on the first platform
	char devicesName[256];
	cl_device_id* devices = (cl_device_id*) malloc(numberOfDevices * sizeof(cl_device_id));
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numberOfDevices, devices, NULL);
	//printf("OpenCL device: ");
	for (cl_uint i = 0; i < numberOfDevices; i++) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(devicesName), &devicesName[0], NULL);
		//printf("%s\n", devicesName);
	}

	// Create Context
	cl_context context = clCreateContext(0, numberOfDevices, devices, NULL, NULL, &error);
	checkError(error);

	// Set up Memory
	static const cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };
	cl_mem inBuffer = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&format,
		imageWidth, imageHeight, 0,
		inputImageHost, &error);
	checkError(error);

	cl_mem outBuffer = clCreateImage2D(context,
		CL_MEM_WRITE_ONLY,
		&format,
		imageWidth, imageHeight, 0,
		NULL, &error);
	checkError(error);

	// Load kernel code
	FILE* sourcePointer;
	char* sourceString;
	int sourceSize;
	int finalSourceSize;

	if (fopen_s(&sourcePointer, "kernel.cl", "r") != 0) {
		printf("Invalid Kernel");
	}

	fseek(sourcePointer, 0, SEEK_END);
	sourceSize = ftell(sourcePointer);
	fseek(sourcePointer, 0, SEEK_SET);
	sourceString = (char*) malloc(sourceSize * sizeof(char));
	finalSourceSize = fread(sourceString, sizeof(char), sourceSize, sourcePointer);
	fclose(sourcePointer);
	
	// Create Program from Kernel Code
	size_t kernelsLength[] = { finalSourceSize };
	const char* kernelsSources[] = { sourceString };
	cl_program program = clCreateProgramWithSource(context, 1, kernelsSources, kernelsLength, &error);
	checkError(error);
	
	// Build Program
	checkError(clBuildProgram(program, numberOfDevices, devices, NULL, NULL, NULL));
	
	// Create Kernel
	cl_kernel kernel = clCreateKernel(program, "sobel", &error);
	checkError(error);

	// Set arguments for kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inBuffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outBuffer);

	// Create command queue on first device
	cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &error);
	checkError(error);

	// Enqueue all instances of kernel
	const size_t baseOffsets[3] = { 1,1,0 };
	const size_t filterRegion[3] = { imageWidth - 2, imageHeight - 2, 1 };
	checkError(clEnqueueNDRangeKernel(queue, kernel, 2, baseOffsets, filterRegion, NULL, 0, NULL, NULL));

	// Copy results back to system memory
	const size_t offsetOut[3] = { 0,0,0 };
	const size_t region[3] = { imageWidth, imageHeight, 1 };
	checkError(clEnqueueReadImage(queue, outBuffer, CL_TRUE, 
		offsetOut, region, 0, 0, outputImageHost, 0, NULL, NULL));
	
	clReleaseMemObject(inBuffer);
	clReleaseMemObject(outBuffer);
	clReleaseCommandQueue(queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseContext(context);
}
