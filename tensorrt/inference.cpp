#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

#include <cuda_runtime.h>

#include "./csrc/engine.h"

using namespace std;
using namespace cv;

#define N 1000

int main(int argc, char *argv[]) {
	if (argc<3 || argc>4) {
		cerr << "Usage: " << argv[0] << " engine.plan image.jpg [<OUTPUT>.png]" << endl;
		return 1;
	}

	cout << "Loading engine..." << endl;
	auto engine = retinanet::Engine(argv[1]);

	cout << "Preparing data..." << endl;
	auto image = imread(argv[2], IMREAD_COLOR);
	auto inputSize = engine.getInputSize();
	cv::resize(image, image, Size(inputSize[0], inputSize[1]));
	cv::Mat pixels;
	image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

	int channels = 3;
	vector<float> img;
	vector<float> data (channels * inputSize[0] * inputSize[1]);

	if (pixels.isContinuous())
		img.assign((float*)pixels.datastart, (float*)pixels.dataend);
	else {
		cerr << "Error reading image " << argv[2] << endl;
		return -1;
	}

	vector<float> mean {0.485, 0.456, 0.406};
	vector<float> std {0.229, 0.224, 0.225};

	for (int c = 0; c < channels; c++) {
		for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
			data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
		}
	}        

	// Create device buffers
	void *data_d, *mask_d;
	auto num_det = engine.getMaxDetections();
	cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&mask_d, 1 * inputSize[0] * inputSize[1] * sizeof(float));

	// Copy image to device
	size_t dataSize = data.size() * sizeof(float);
	cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

	// Run inference n times
	cout << "Running inference..." << endl;
	const int count = 100;
	auto start = chrono::steady_clock::now();
 	vector<void *> buffers = { data_d, mask_d };
	// vector<void *> buffers = { data_d};
	for (int i = 0; i < count; i++) {
		engine.infer(buffers);
	}
	auto stop = chrono::steady_clock::now();
	auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
	cout << "Took " << timing.count() / count << " seconds per inference." << endl;

	// cudaFree(data_d);

	// Get back the bounding boxes
	unique_ptr<float[]> scores(new float[num_det]);
	unique_ptr<float[]> boxes(new float[num_det * 4]);
	unique_ptr<float[]> classes(new float[num_det]);

	Mat m = Mat(inputSize[0], inputSize[1], CV_32F);

	

	cudaMemcpy(m.data, mask_d, 1 * inputSize[0] * inputSize[1] * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(data_d);
	cout << "Channels" << channels << endl;
	cout << "inputSize[0]" << inputSize[0] << endl;
	cout << "inputSize[1]" << inputSize[1] << endl;
	cv::Mat bgr;
	cvtColor(m, bgr, CV_GRAY2BGR);

	// Write image
	string out_file = argc == 4 ? string(argv[3]) : "detections.png";
	cout << "Saving result to " << out_file << endl;
	imwrite(out_file, bgr);
	
	return 0;
}
