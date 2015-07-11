#define CPU_ONLY 1

#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <boost/filesystem.hpp>

#include "caffe/caffe.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;
using namespace caffe;

string train_path = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-set1-train1";
string test_path = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-set1-test-imgs";

int w = 100, h = 35;
int descriptor_size = 16;

struct Distance {
    string to;
    float value;
};

struct Descriptor {
    string writerId;
    vector<float> values;
    vector<Distance> distances;
};

struct Writer {
    string id;
    vector<Mat> images;
};

bool operator<(const Distance &d1, const Distance &d2) { return (d1.value < d2.value); }

Distance distance(Descriptor &d1, Descriptor &d2) {
    Distance d;
    d.to = d2.writerId;

    float total = 0;
    for (int i = 0; i < d1.values.size(); ++i) {
        total += (d1.values[i] - d2.values[i]) * (d1.values[i] - d2.values[i]);
    }
    d.value = total;

    return d;
}

void readWriter(filesystem::path writer_dir, Writer &writer) {
    cout << "reading writer " << writer.id << endl;
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(writer_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_regular_file(dir_iter->status())) {
            filesystem::path file_path = dir_iter->path();

            string fileName = file_path.string();
            if (fileName.rfind(".png") == string::npos)
                continue;

            Mat image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

            image.convertTo(image, CV_32F, 1/255.f);
            Size imgSize(image.cols, image.rows);
            resize(image, image, imgSize / 2);

            writer.images.push_back(image);
            //cout << "read " << fileName << endl;
        }
    }
}

vector<Writer> readWriters(string input_path) {
    vector<Writer> writers;

    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path writer_dir = dir_iter->path();

            Writer writer;
            writer.id = writer_dir.leaf().string();
            readWriter(writer_dir, writer);
            writers.push_back(writer);
        }
    }
    cout << "read " << writers.size() << " writers" << endl;
    return writers;
}

Mat readMean(int rows, int cols) {
    Mat mean(rows, cols, CV_32F);
    ifstream mean_in("train_mean.bin", ios::binary );
    mean_in.read((char*) mean.data, mean.cols * mean.rows * sizeof(float));
    //imshow("", mean);
    //waitKey();
    return mean;
}

vector<Descriptor> readDescriptors(Net<float> &net, Mat &mean, vector<Writer> &writers) {

    boost::shared_ptr<MemoryDataLayer<float> > input_layer =
            boost::dynamic_pointer_cast< MemoryDataLayer<float> > (net.layers()[0]);
    vector<Descriptor> descriptors;
    int batch_size = 50;

    for (Writer &writer: writers) {
        cout << "computing descriptors for " << writer.id << endl;
        int image_floats = writer.images[0].rows * writer.images[0].cols;
        vector<float> data(image_floats * batch_size, 0);
        vector<float> label(batch_size, atoi(writer.id.c_str()));

        for (int i = 0; i < writer.images.size(); ++i) {
            Mat &image = writer.images[i];
            Mat demeanImage = image - mean;
            memcpy(&data[image_floats * (i % batch_size)], demeanImage.data, image_floats * sizeof(float));

            if (i % batch_size == batch_size - 1) {
                input_layer->Reset(data.data(), label.data(), batch_size);

                net.ForwardPrefilled();
                boost::shared_ptr<Blob<float>> descriptorBlob = net.blob_by_name("descriptor");
                const float* descriptorData = descriptorBlob->cpu_data();
                for (int i = 0; i < batch_size; ++i) {
                    Descriptor descriptor;
                    descriptor.writerId = writer.id;
                    for (int j = 0; j < descriptor_size; ++j) {
                        descriptor.values.push_back(descriptorData[i * descriptor_size + j]);
                    }
                    descriptors.push_back(descriptor);
                }
            }
        }
    }

    return descriptors;
}

void computeDistances(vector<Descriptor> &descriptors, vector<Descriptor> &trainedDescriptors) {
    cout << "start computing distances " << endl;
    for (int i = 0; i < descriptors.size(); ++i) {
        for (int j = 0; j < trainedDescriptors.size(); ++j) {
            Distance d = distance(descriptors[i], trainedDescriptors[j]);
            if (i != j)
                descriptors[i].distances.push_back(d);
        }
        std::sort(descriptors[i].distances.begin(), descriptors[i].distances.end());
    }
    cout << "finished computing distances " << endl;
}

void countKNN(vector<Descriptor> &descriptors, int k) {
    int counter = 0;
    for (int i = 0; i < descriptors.size(); ++i) {
        int same = 0;
        for (int j = 0; j < k; ++j) {
            if (descriptors[i].writerId == descriptors[i].distances[j].to)
                same++;
        }
        //cout << i << " " << same << " " << k / 2 << endl;
        if (same > k / 2)
            counter++;
    }
    cout << k << "NN accuracy " << ((float) counter) / descriptors.size() << endl;
}

int main(int argc, char *argv[])
{
    vector<Writer> writers = readWriters(train_path);
    Mat mean = readMean(writers[0].images[0].rows, writers[0].images[0].cols);
    Net<float> net(argv[1], caffe::TEST);
    net.CopyTrainedLayersFrom(argv[2]);
    int batchSize = boost::dynamic_pointer_cast< MemoryDataLayer<float> >(net.layers()[0])->batch_size();

    cout << batchSize << endl;

    vector<Descriptor> trainedDescriptors = readDescriptors(net, mean, writers);
    for (int i = 0; i < descriptor_size; ++i) {
        cout << trainedDescriptors[0].values[i] << " ";
    }
    cout << endl;
    writers.clear();

    computeDistances(trainedDescriptors, trainedDescriptors);

    countKNN(trainedDescriptors, 1);
    countKNN(trainedDescriptors, 3);
    countKNN(trainedDescriptors, 5);
    countKNN(trainedDescriptors, 7);

    writers = readWriters(test_path);
    vector<Descriptor> descriptors = readDescriptors(net, mean, writers);
    writers.clear();

    computeDistances(descriptors, trainedDescriptors);

    countKNN(descriptors, 1);
    countKNN(descriptors, 3);
    countKNN(descriptors, 5);
    countKNN(descriptors, 7);
}
