__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

__kernel void sobel(
	__read_only image2d_t input,
	__write_only image2d_t output)
{
	const int hKernel[3][3] = {
		{-1, 0, 1 },
		{-2, 0, 2 },
		{-1, 0, 1 }
	};

	const int vKernel[3][3] = {
		{-1, -2, -1 },
		{ 0,  0,  0 },
		{ 1,  2,  1 }
	};

	const int2 pixelCoordinate = { get_global_id(0), get_global_id(1) };

	int4 magnitudeX = (int4)(0);
	int4 magnitudeY = (int4)(0);

	for (int h = 0; h < 3; h++) {
		for (int v = 0; v < 3; v++) {
			int2 inputPixelCoordinate = { get_global_id(0) + (h - 1), get_global_id(1) + (v - 1) };
			int4 inputPixel = read_imagei(input, sampler, inputPixelCoordinate);
			magnitudeX += inputPixel * hKernel[h][v];
			magnitudeY += inputPixel * vKernel[h][v];
		}
	}

	const uint4 maxRGBValues = (int)(255);
	uint4 finalPixel = abs(magnitudeX) + abs(magnitudeY);
	finalPixel.w = 255;
	uint4 finalPixelClipped = min(finalPixel, maxRGBValues);

	write_imageui(output, pixelCoordinate, finalPixelClipped);
}