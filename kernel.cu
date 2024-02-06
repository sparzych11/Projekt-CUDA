#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int numBins = 256; 

//Funkcja do obliczania Histogramu:
__global__
void computeHistogram(const unsigned char* image, int width, int height, int* histogram) {

    __shared__ int sharedHistogram[numBins];

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        sharedHistogram[i] = 0;
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < width * height) {
        atomicAdd(&sharedHistogram[image[idx]], 1);
        idx += stride;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numBins; i += blockDim.x) {
        atomicAdd(&histogram[i], sharedHistogram[i]);
    }

}

//Funkcja do liniowej modyfikacji Histogramu:
__global__
void linearTransform(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < width * height * 3) {
        float scalingFactor = 255.0 / (width * height);
        outputImage[idx] = static_cast<unsigned char>(histogram[inputImage[idx]] * scalingFactor);
        idx += stride;
    }
}
//Funkcja do wyrównania Histogramu:
__global__
void histogramEqualization(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < width * height * 3) {
        float cdf = 0.0;
        for (int i = 0; i <= inputImage[idx]; ++i) {
            cdf += histogram[i];
        }

        outputImage[idx] = static_cast<unsigned char>((cdf / (width * height * 3)) * 255.0);

        idx += stride;
    }
}

//Funkcja do rozszerzenia liniowego Histogramu:
__global__
void linearStretch(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int minValue = 0;
    int maxValue = numBins - 1;
    while (histogram[minValue] == 0) ++minValue;
    while (histogram[maxValue] == 0) --maxValue;

    while (idx < width * height * 3) {
        outputImage[idx] = static_cast<unsigned char>((inputImage[idx] - minValue) * (255.0 / (maxValue - minValue)));
        idx += stride;
    }
}
//Funkcja do rozszerzenia nieliniowego Histogramu:
__global__
void nonlinearStretch(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int minValue = 0;
    int maxValue = numBins - 1;
    while (histogram[minValue] == 0) ++minValue;
    while (histogram[maxValue] == 0) --maxValue;

    while (idx < width * height * 3) {
        outputImage[idx] = static_cast<unsigned char>(
            255.0 * sqrt((inputImage[idx] - minValue) / static_cast<float>(maxValue - minValue))
            );
        idx += stride;
    }
}

// Funkcja do progowania Histogramu:
__global__
void thresholdHistogram(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram, unsigned char threshold) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < width * height * 3) {
        outputImage[idx] = (inputImage[idx] > threshold) ? 255 : 0;
        idx += stride;
    }
}

// Funkcja do inwersji Histogramu:
__global__
void invertHistogram(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int* histogram) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < width * height * 3) {
        outputImage[idx] = 255 - inputImage[idx];
        idx += stride;
    }
}

int main() {
    // Wczytanie obrazu:
    int width, height, channels;
    unsigned char* image = stbi_load("test.jpg", &width, &height, &channels, 0);

    if (!image) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    // Wyznaczenie Histogramu:
    int histogram[numBins] = { 0 };
    unsigned char* d_image;
    int* d_histogram;
    cudaMalloc((void**)&d_image, width * height * channels);
    cudaMalloc((void**)&d_histogram, numBins * sizeof(int));
    cudaMemcpy(d_image, image, width * height * channels, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, numBins * sizeof(int));
    int blockSize = 256;
    int numBlocks = (width * height + blockSize - 1) / blockSize;
    computeHistogram << <numBlocks, blockSize >> > (d_image, width, height, d_histogram);
    cudaMemcpy(histogram, d_histogram, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numBins; ++i) {
        std::cout << "Bin " << i << ": " << histogram[i] << std::endl;
    }

    // Operacja liniowa na Histogramie:
    unsigned char* d_outputImage;
    cudaMalloc((void**)&d_outputImage, width * height * channels);
    linearTransform << <numBlocks, blockSize >> > (d_image, d_outputImage, width, height, d_histogram);
    unsigned char* outputImage = new unsigned char[width * height * channels];
    cudaMemcpy(outputImage, d_outputImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("test_linear.jpg", width, height, channels, outputImage, 100);

    // Wyrównanie Histogramu:
    unsigned char* d_equalizedImage;
    cudaMalloc((void**)&d_equalizedImage, width * height * channels);
    histogramEqualization << <numBlocks, blockSize >> > (d_image, d_equalizedImage, width, height, d_histogram);
    unsigned char* equalizedImage = new unsigned char[width * height * channels];
    cudaMemcpy(equalizedImage, d_equalizedImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("equal_test.jpg", width, height, channels, equalizedImage, 100);

    // Rozszerzenie liniowe Histogramu:
    unsigned char* d_stretchedImage;
    cudaMalloc((void**)&d_stretchedImage, width * height * channels);
    linearStretch << <numBlocks, blockSize >> > (d_image, d_stretchedImage, width, height, d_histogram);
    unsigned char* stretchedImage = new unsigned char[width * height * channels];
    cudaMemcpy(stretchedImage, d_stretchedImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("stretched_test.jpg", width, height, channels, stretchedImage, 100);

    // Rozszerzenie nieliniowe Histogramu:
    unsigned char* d_nonlinearStretchedImage;
    cudaMalloc((void**)&d_nonlinearStretchedImage, width * height * channels);
    nonlinearStretch << <numBlocks, blockSize >> > (d_image, d_nonlinearStretchedImage, width, height, d_histogram);
    unsigned char* nonlinearStretchedImage = new unsigned char[width * height * channels];
    cudaMemcpy(nonlinearStretchedImage, d_nonlinearStretchedImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("stretched_nonlinear_test.jpg", width, height, channels, nonlinearStretchedImage, 100);

    // Progowanie Histogramu:
    unsigned char* d_thresholdImage;
    cudaMalloc((void**)&d_thresholdImage, width * height * channels);
    unsigned char* thresholdImage = new unsigned char[width * height * channels];
    unsigned char thresholdValue = 128; // Przykładowy próg
    thresholdHistogram << <numBlocks, blockSize >> > (d_image, d_thresholdImage, width, height, d_histogram, thresholdValue);
    cudaMemcpy(thresholdImage, d_thresholdImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("threshold_test.jpg", width, height, channels, thresholdImage, 100);

    // Inwersja Histogramu:
    unsigned char* d_invertedImage;
    cudaMalloc((void**)&d_invertedImage, width * height * channels);
    unsigned char* invertedImage = new unsigned char[width * height * channels];
    invertHistogram << <numBlocks, blockSize >> > (d_image, d_invertedImage, width, height, d_histogram);
    cudaMemcpy(invertedImage, d_invertedImage, width * height * channels, cudaMemcpyDeviceToHost);
    stbi_write_jpg("inverted_test.jpg", width, height, channels, invertedImage, 100);

    // Zwolnienie pamięci GPU:
    cudaFree(d_image);
    cudaFree(d_histogram);
    cudaFree(d_outputImage);
    cudaFree(d_equalizedImage);
    cudaFree(d_stretchedImage);
    cudaFree(d_nonlinearStretchedImage);
    cudaFree(d_thresholdImage);
    cudaFree(d_invertedImage);

    // Zwolnienie pamięci CPU:
    stbi_image_free(image);
    delete[] outputImage;
    delete[] equalizedImage;
    delete[] stretchedImage;
    delete[] nonlinearStretchedImage;
    delete[] thresholdImage;
    delete[] invertedImage;
    return 0;
}