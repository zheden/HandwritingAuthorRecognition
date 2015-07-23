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

string root_dir = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/";
string train_path = root_dir + "new-data-set1-train3";
string test_path = root_dir + "new-data-set1-test3";
string unknown_train_path = root_dir + "all-writers-test-train";
string unknown_test_path = root_dir + "all-writers-test-val";

int w = 100, h = 35;
int descriptor_size = 16;
int batch_size = 50;
float zeroPenalty = 15;

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

float euclid_sq_distance(Descriptor &d1, Descriptor &d2) {
    float total = 0;
    for (int i = 0; i < d1.values.size(); ++i) {
        float axisDist = (d1.values[i] - d2.values[i]) * (d1.values[i] - d2.values[i]);
        total += axisDist;
    }
    return total;
}

Distance distance(Descriptor &d1, Descriptor &d2) {
    Distance d;
    d.to = d2.writerId;
    d.value = euclid_sq_distance(d1, d2);
    return d;
}

Distance similarity(Descriptor &d1, Descriptor &d2) {
    Distance d;
    d.to = d2.writerId;
    float total = 0;
    for (int i = 0; i < d1.values.size(); ++i) {
        float axisDist = (d1.values[i] - d2.values[i]) * (d1.values[i] - d2.values[i]);
        if (d1.values[i] == 0 || d2.values[i] == 0)
            axisDist *= zeroPenalty;
        total += axisDist;
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

vector<Descriptor> computeDescriptors(Net<float> &net, Mat &mean, vector<Writer> &writers) {

    boost::shared_ptr<MemoryDataLayer<float> > input_layer =
            boost::dynamic_pointer_cast< MemoryDataLayer<float> > (net.layers()[0]);
    vector<Descriptor> descriptors;

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

        writer.images.clear();
    }

    return descriptors;
}

void computeDistances(vector<Descriptor> &descriptors, vector<Descriptor> &trainedDescriptors, bool test_data) {
    cout << "start computing distances " << endl;
    for (int i = 0; i < descriptors.size(); ++i) {
        for (int j = 0; j < trainedDescriptors.size(); ++j) {
            Distance d = distance(descriptors[i], trainedDescriptors[j]);
            if (test_data || i != j)
                descriptors[i].distances.push_back(d);
        }
    }
    cout << "finished computing distances " << endl;
}

vector<vector<int>> countKNN(vector<Writer> &writers, vector<Writer> &trainedWriters, vector<Descriptor> &descriptors, int k) {
    vector<vector<int>> hitCounts;
    map<string,int> writerRowIndex, writerColIndex;
    for (int i = 0; i < writers.size(); ++i) {
       vector<int> writerHitCounts(trainedWriters.size(), 0);
       hitCounts.push_back(writerHitCounts);
       writerRowIndex[writers[i].id] = i;
    }
    for (int i = 0; i < trainedWriters.size(); ++i) {
        writerColIndex[trainedWriters[i].id] = i;
    }

    float correct_dist = 0, wrong_dist = 0;
    int correct_count = 0, wrong_count = 0;

    int counter = 0;
    for (int i = 0; i < descriptors.size(); ++i) {
        string writerId = descriptors[i].writerId;
        int same = 0;
        std::sort(descriptors[i].distances.begin(), descriptors[i].distances.end());
        for (int j = 0; j < k; ++j) {
            string otherWriterId = descriptors[i].distances[j].to;
            hitCounts[writerRowIndex[writerId]][writerColIndex[otherWriterId]]++;
            if (writerId == otherWriterId) {
                same++;
                correct_dist += descriptors[i].distances[j].value;
                correct_count++;
            }
        }
        for (int j = 0; j < descriptors[i].distances.size(); ++j) {
            if (writerId != descriptors[i].distances[j].to) {
                wrong_dist += descriptors[i].distances[j].value;
                wrong_count++;
                break;
            }
        }
        //cout << i << " " << same << " " << k / 2 << endl;
        if (same > k / 2)
            counter++;
    }
    cout << k << "NN accuracy " << ((float) counter) / descriptors.size() << endl;

    if (trainedWriters.size() < 30) {
        cout << endl;
        for (int i = 0; i < writers.size(); ++i) {
            for (int j = 0; j < trainedWriters.size(); ++j) {
                cout << setfill(' ') << setw(4) << hitCounts[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    correct_dist /= correct_count;
    wrong_dist /= wrong_count;
    cout << "closest same: " << correct_dist << " closest other: " << wrong_dist << endl;

    return hitCounts;
}

vector<Descriptor> createMetaDescriptors(vector<Writer> &writers, vector<vector<int>> &hitCounts, float normalizationFactor) {
    vector<Descriptor> descriptors;
    for (int i = 0; i < writers.size(); ++i) {
        Descriptor descriptor;
        descriptor.writerId = writers[i].id;
        for (int j = 0; j < hitCounts[i].size(); ++j) {
            descriptor.values.push_back(hitCounts[i][j] / normalizationFactor);
        }
        descriptors.push_back(descriptor);
    }
    return descriptors;
}

void computeMetaDistances(vector<Descriptor> &descriptors, vector<Descriptor> &trainedDescriptors) {
    cout << "start computing distances " << endl;
    for (int i = 0; i < descriptors.size(); ++i) {
        for (int j = 0; j < trainedDescriptors.size(); ++j) {
            Distance d = similarity(descriptors[i], trainedDescriptors[j]);
            //cout << d.value << " ";
            descriptors[i].distances.push_back(d);
        }
        //cout << endl;
    }
    cout << "finished computing distances " << endl;
}

vector<Descriptor> computeCentroids(vector<Descriptor> &descriptors) {
    vector<Descriptor> centroids;
    string writerId = "first";
    Descriptor centroid;
    int counter = 0;
    for (Descriptor &descriptor: descriptors) {
        if (descriptor.writerId != writerId) {
            if (writerId != "first")
                centroids.push_back(centroid);
            Descriptor new_centroid;
            new_centroid.writerId = descriptor.writerId;
            new_centroid.values.assign(descriptor_size, 0);
            centroid = new_centroid;
            writerId = descriptor.writerId;
            counter = 0;
        }
        for (int i = 0; i < descriptor.values.size(); ++i) {
            centroid.values[i] += descriptor.values[i];
        }
        counter++;
    }
    centroids.push_back(centroid);

    for (int i = 0; i < centroids.size(); ++i) {
        for (int j = 0; j < descriptor_size; ++j) {
            centroids[i].values[j] /= counter;
        }
    }

    for (int i = 0; i < centroids.size(); ++i) {
        vector<float> distances;
        float radius = 0, std_dev = 0;
        for (Descriptor &descriptor: descriptors) {
            if (descriptor.writerId == centroids[i].writerId) {
                float dist = sqrt(euclid_sq_distance(descriptor, centroids[i]));
                radius += dist;
                distances.push_back(dist);
            }
        }
        radius /= counter;
        for (int j = 0; j < distances.size(); ++j) {
            std_dev += (radius - distances[j]) * (radius - distances[j]);
        }
        std_dev /= counter;
        std_dev = sqrt(std_dev);
        cout << centroids[i].writerId << " " << radius << " +/- " << std_dev << endl;
    }

    return centroids;
}

int main(int argc, char *argv[])
{
    vector<Writer> writers = readWriters(train_path);
    Mat mean = readMean(writers[0].images[0].rows, writers[0].images[0].cols);
    Net<float> net(argv[1], caffe::TEST);
    net.CopyTrainedLayersFrom(argv[2]);
//    int batchSize = boost::dynamic_pointer_cast< MemoryDataLayer<float> >(net.layers()[0])->batch_size();
//    cout << batchSize << endl;

    vector<Descriptor> trainedDescriptors = computeDescriptors(net, mean, writers);
//    for (int i = 0; i < trainedDescriptors.size(); ++i) {
//        for (int j = 0; j < descriptor_size; ++j) {
//            cout << trainedDescriptors[i].values[j] << " ";
//        }
//        cout << endl;
//    }

//    computeDistances(trainedDescriptors, trainedDescriptors, false);
//    countKNN(writers, trainedDescriptors, 3);

    vector<Descriptor> centroids = computeCentroids(trainedDescriptors);
//    for (int i = 0; i < centroids.size(); ++i) {
//        for (int j = 0; j < descriptor_size; ++j) {
//            cout << centroids[i].values[j] << " ";
//        }
//        cout << endl;
//    }

    computeDistances(centroids, centroids, false);

    cout << endl;
    for (int i = 0; i < centroids.size(); ++i) {
        for (int j = 0; j < centroids[i].distances.size(); ++j) {
            if (i == j)
                cout << "     -    ";
            else
                cout << setfill(' ') << setw(9) << centroids[i].distances[j].value << " ";
        }
        cout << endl;
    }
    cout << endl;

//    for (Descriptor &descriptor: trainedDescriptors)
//        descriptor.distances.clear();
    computeDistances(trainedDescriptors, centroids, true);
    countKNN(writers, writers, trainedDescriptors, 1);

    vector<Writer> unknown_writers_train = readWriters(unknown_train_path + argv[3]);
    vector<Descriptor> unknownDescriptors = computeDescriptors(net, mean, unknown_writers_train);
    computeDistances(unknownDescriptors, centroids, true);
    vector<vector<int>> hitCounts = countKNN(unknown_writers_train, writers, unknownDescriptors, 1);

    float normalizationFactor = unknownDescriptors.size() / (float) unknown_writers_train.size();
    vector<Descriptor> metaDescriptorsTrain = createMetaDescriptors(unknown_writers_train, hitCounts, normalizationFactor);

    vector<Writer> unknown_writers_val = readWriters(unknown_test_path + argv[3]);
    unknownDescriptors = computeDescriptors(net, mean, unknown_writers_val);
    computeDistances(unknownDescriptors, centroids, true);
    hitCounts = countKNN(unknown_writers_val, writers, unknownDescriptors, 1);
    vector<Descriptor> metaDescriptorsVal = createMetaDescriptors(unknown_writers_val, hitCounts, normalizationFactor);

    computeMetaDistances(metaDescriptorsVal, metaDescriptorsTrain);
    countKNN(unknown_writers_val, unknown_writers_train, metaDescriptorsVal, 1);

//    vector<Writer> testWriters = readWriters(test_path);
//    vector<Descriptor> descriptors = computeDescriptors(net, mean, testWriters);

////    computeDistances(descriptors, trainedDescriptors, true);
////    countKNN(testWriters, descriptors, 3);

////     for (Descriptor &descriptor: descriptors)
////        descriptor.distances.clear();
//     computeDistances(descriptors, centroids, true);
//     countKNN(testWriters, writers, descriptors, 1);
}
