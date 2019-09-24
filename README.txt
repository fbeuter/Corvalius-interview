Sobel Filter README

The compiled application can be run via the Command Prompt, the paramaters are as follows:

corvalius-filter filterVersion inputFile outputFile

filterVersion: Represents the version of the filter to use, 0 = Base filter
                                                            1 = Fast Memory Version
                                                            2 = Multithreaded Fast Memory Version
                                                            3 = CUDA Version
                                                            4 = OpenCL Version

inputFile: Input image to filter
outputFile: Filtered image output

In order to run the CUDA and OpenCL versions, the appropiate DLLs need to be compiled and added to the
executable folder in conjunction with the kernel.cl file, this file should be in the openCL-Filter folder