using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace corvalius_filter
{
    class Program
    {
        [DllImport("cuda-filter.dll", EntryPoint = "sobelFilterCUDA", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern unsafe void sobelFilterCUDA(byte[] inputImageHost, byte[] outputImageHost, int imageWidth, int imageHeight);

        [DllImport("openCL-filter.dll", EntryPoint = "sobelFilterOpenCL", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern unsafe void sobelFilterOpenCL(byte[] inputImageHost, byte[] outputImageHost, int imageWidth, int imageHeight);

        // Horizontal Kernel
        static int[,] hKernel = new int[3, 3]
        {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        // Vertical Kernel
        static int[,] vKernel = new int[3, 3]
        {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        public static Bitmap Sobel(Bitmap input)
        {
            Bitmap output = new Bitmap(input.Width, input.Height);

            for(int x=1; x<input.Width - 1; x++)
            {
                for(int y=1; y<input.Height - 1; y++)
                {
                    // Magnitudes
                    int magXR = 0, magXG = 0, magXB = 0;
                    int magYR = 0, magYG = 0, magYB = 0;

                    // Horizontal and Vertical convolution at point (x,y)
                    for (int h=0; h<3; h++)
                    {
                        for(int v=0; v<3; v++)
                        {
                            // Current pixel in 3x3 convolution window
                            int xn = x + (h-1);
                            int yn = y + (v-1);

                            Color inputPixel = input.GetPixel(xn, yn);
                            // Horizontal Convolution
                            int hKernelValue = hKernel[h, v];
                            magXR += inputPixel.R * hKernelValue;
                            magXG += inputPixel.G * hKernelValue;
                            magXB += inputPixel.B * hKernelValue;
                            // Vertical Convolution
                            int vKernelValue = vKernel[h, v];
                            magYR += inputPixel.R * vKernelValue;
                            magYG += inputPixel.G * vKernelValue;
                            magYB += inputPixel.B * vKernelValue;
                        }
                    }

                    // Compute final pixel value
                    // We clip the value to 255 in case we go over the 8-bit range
                    // Instead of using norm, we apprximate using sum of abs, this
                    // has been show to be as effective in this application
                    int finalR = Math.Min(255, Math.Abs(magXR) + Math.Abs(magYR));
                    int finalG = Math.Min(255, Math.Abs(magXG) + Math.Abs(magYG));
                    int finalB = Math.Min(255, Math.Abs(magXB) + Math.Abs(magYB));

                    Color outputPixel = Color.FromArgb(finalR, finalG, finalB);
                    output.SetPixel(x, y, outputPixel);
                }
            }

            return output;
        }

        public static void SobelFastMemory(byte[] rgbValues, byte[] outputArray, int inputWidth, int inputHeight)
        {
            for (int x = 1; x < inputWidth-1; x++)
            {
                for (int y = 1; y < inputHeight-1; y++)
                {
                    // Magnitudes
                    int magXR = 0, magXG = 0, magXB = 0;
                    int magYR = 0, magYG = 0, magYB = 0;

                    // Horizontal and Vertical convolution at point (x,y)
                    for (int h = 0; h < 3; h++)
                    {
                        for (int v = 0; v < 3; v++)
                        {
                            // Current pixel in 3x3 convolution window
                            int xn = x + (h - 1);
                            int yn = y + (v - 1);
                            int pixelOffset = (yn * inputWidth + xn)*4;
                            
                            // Horizontal Convolution
                            int hKernelValue = hKernel[h, v];
                            magXR += rgbValues[pixelOffset] * hKernelValue;
                            magXG += rgbValues[pixelOffset + 1] * hKernelValue;
                            magXB += rgbValues[pixelOffset + 2] * hKernelValue;
                            // Vertical Convolution
                            int vKernelValue = vKernel[h, v];
                            magYR += rgbValues[pixelOffset] * vKernelValue;
                            magYG += rgbValues[pixelOffset + 1] * vKernelValue;
                            magYB += rgbValues[pixelOffset + 2] * vKernelValue;
                        }
                    }

                    // Compute final pixel value
                    // We clip the value to 255 in case we go over the 8-bit range
                    // Instead of using norm, we apprximate using sum of abs, this
                    // has been show to be as effective in this application
                    int finalR = Math.Min(255, Math.Abs(magXR) + Math.Abs(magYR));
                    int finalG = Math.Min(255, Math.Abs(magXG) + Math.Abs(magYG));
                    int finalB = Math.Min(255, Math.Abs(magXB) + Math.Abs(magYB));

                    int finalPixelOffset = (y * inputWidth + x)*4;
                    outputArray[finalPixelOffset] = (byte)finalR;
                    outputArray[finalPixelOffset + 1] = (byte)finalG;
                    outputArray[finalPixelOffset + 2] = (byte)finalB;
                    outputArray[finalPixelOffset + 3] = 255;
                }
            }
        }

        public static void SobelFastMemotyMT(byte[] rgbValues, byte[] outputArray, int inputWidth, int inputHeight)
        {
            var forOptions = new ParallelOptions();
            forOptions.MaxDegreeOfParallelism = Environment.ProcessorCount - 1;

            Parallel.For(1, inputHeight-1, forOptions, y =>
            {
                for (int x = 1; x < inputWidth-1; x++)
                {
                    // Magnitudes
                    int magXR = 0, magXG = 0, magXB = 0;
                    int magYR = 0, magYG = 0, magYB = 0;

                    // Horizontal and Vertical convolution at point (x,y)
                    for (int h = 0; h < 3; h++)
                    {
                        for (int v = 0; v < 3; v++)
                        {
                            // Current pixel in 3x3 convolution window
                            int xn = x + (h - 1);
                            int yn = y + (v - 1);
                            int pixelOffset = (yn * inputWidth + xn) * 4;

                            // Horizontal Convolution
                            int hKernelValue = hKernel[h, v];
                            magXR += rgbValues[pixelOffset] * hKernelValue;
                            magXG += rgbValues[pixelOffset + 1] * hKernelValue;
                            magXB += rgbValues[pixelOffset + 2] * hKernelValue;
                            // Vertical Convolution
                            int vKernelValue = vKernel[h, v];
                            magYR += rgbValues[pixelOffset] * vKernelValue;
                            magYG += rgbValues[pixelOffset + 1] * vKernelValue;
                            magYB += rgbValues[pixelOffset + 2] * vKernelValue;
                        }
                    }

                    // Compute final pixel value
                    // We clip the value to 255 in case we go over the 8-bit range
                    // Instead of using norm, we apprximate using sum of abs, this
                    // has been show to be as effective in this application
                    int finalR = Math.Min(255, Math.Abs(magXR) + Math.Abs(magYR));
                    int finalG = Math.Min(255, Math.Abs(magXG) + Math.Abs(magYG));
                    int finalB = Math.Min(255, Math.Abs(magXB) + Math.Abs(magYB));

                    int finalPixelOffset = (y * inputWidth + x) * 4;
                    outputArray[finalPixelOffset] = (byte)finalR;
                    outputArray[finalPixelOffset + 1] = (byte)finalG;
                    outputArray[finalPixelOffset + 2] = (byte)finalB;
                    outputArray[finalPixelOffset + 3] = 255;
                }
            });
        }

        static void Main(string[] args)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            if (args.Length == 3)
            {
                Bitmap input = new Bitmap(args[1]);
                Bitmap output = new Bitmap(input.Width, input.Height);

                if (args[0] == "0")
                {
                    Console.WriteLine("Running basic Sobel");
                    output = Sobel(input);
                } else
                {
                    // We have to lock the bitmap to system memory
                    // Lock the bitmap's bits.  
                    Rectangle rect = new Rectangle(0, 0, input.Width, input.Height);
                    BitmapData bmpData =
                        input.LockBits(rect, ImageLockMode.ReadWrite,
                        PixelFormat.Format32bppRgb);

                    // Get the address of the first line.
                    IntPtr ptr = bmpData.Scan0;

                    // Declare an array to hold the bytes of the bitmap.
                    int bytes = input.Width * input.Height * 4;
                    byte[] rgbValues = new byte[bytes];

                    // Copy the RGB values into the array.
                    System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

                    // Unlock input bitmap's bits.
                    input.UnlockBits(bmpData);

                    // Create output array
                    byte[] outputArray = new byte[bytes];


                    if(args[0] == "1")
                    {
                        Console.WriteLine("Running fast memory Sobel");
                        SobelFastMemory(rgbValues, outputArray, input.Width, input.Height);
                    } else if(args[0] == "2")
                    {
                        Console.WriteLine("Running multi-thread fast memory Sobel");
                        SobelFastMemotyMT(rgbValues, outputArray, input.Width, input.Height);
                    } else if(args[0] == "3")
                    {
                        Console.WriteLine("Running CUDA Sobel");
                        sobelFilterCUDA(rgbValues, outputArray, input.Width, input.Height);
                    } else if(args[0] == "4")
                    {
                        Console.WriteLine("Running OpenCL Sobel");
                        sobelFilterOpenCL(rgbValues, outputArray, input.Width, input.Height);
                    } else
                    {
                        Console.WriteLine("Invalid parameter, check README");
                    }

                    // Fill output Bitmap
                    Rectangle outputRect = new Rectangle(0, 0, output.Width, output.Height);
                    BitmapData bmpDataOutput =
                        output.LockBits(outputRect, ImageLockMode.ReadWrite,
                        PixelFormat.Format32bppRgb);

                    IntPtr outputPtr = bmpDataOutput.Scan0;

                    // Copy the RGB values back to the bitmap
                    Marshal.Copy(outputArray, 0, outputPtr, bytes);

                    output.UnlockBits(bmpDataOutput);
                }

                output.Save(args[2], ImageFormat.Bmp);
            } else
            {
                Console.WriteLine("Invalid number of parameters, check README");
            }
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            using (System.IO.StreamWriter file =
                new System.IO.StreamWriter(@"time.txt", true))
            {
                file.WriteLine(elapsedMs);
            }
        }
    }
}
