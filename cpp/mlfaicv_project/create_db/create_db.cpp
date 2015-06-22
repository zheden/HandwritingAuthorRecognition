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

string new_data_path = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data";
int resizeFactor = 4;
int maxNewImagesPerImage = 900;
default_random_engine generator;

struct Writer {
    int id;
    vector<Mat> images;
};

vector<Writer> readWriters(string word) {
    vector<Writer> writers;

    string dir_string = new_data_path + "/" + word;
    filesystem::path dir(dir_string);
    if (!(filesystem::exists(dir) && filesystem::is_directory(dir)))
    {
        cout << "Could not open data in " << dir_string << ". Aborting..." << endl;
        return vector<Writer>();
    }

    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            Writer writer;
            writer.id = atoi(dir_iter->path().leaf().string().c_str());
            for(filesystem::directory_iterator dir_iter2(dir_iter->path()); dir_iter2 != end_iter ; ++dir_iter2) {
                if (filesystem::is_regular_file(dir_iter2->status())) {
                    filesystem::path filePath = dir_iter2->path();
                    string fileName = filePath.string();
                    if (fileName.rfind(".png") == string::npos)
                        continue;

                    Mat image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
                    image.convertTo(image, CV_32F, 1/255.f);
                    cout << filePath.leaf().string() << " from writer " << writer.id << " with size: [" << image.cols << " x " << image.rows << "]" << endl;
                    writer.images.push_back(image);
                }
            }
            writers.push_back(writer);
        }
    }

    return writers;
}

vector<Size> getInputSizes(vector<Writer> &writers) {
    vector<Size> sizes;

    vector<int> ws, hs;
    int min_w = 1000000, min_h = 1000000;
    int max_w = 0, max_h = 0;
    int sum_w = 0, sum_h = 0;
    int count = 0;

    for (Writer &writer: writers) {
        for (Mat &image: writer.images) {
            min_w = std::min(min_w, image.cols);
            max_w = std::max(max_w, image.cols);
            sum_w += image.cols;
            min_h = std::min(min_h, image.rows);
            max_h = std::max(max_h, image.rows);
            sum_h += image.rows;
            count++;

            ws.push_back(image.cols);
            hs.push_back(image.rows);
        }
    }

    float mean_w = sum_w / (float) count;
    float mean_h = sum_h / (float) count;

    float std_w = 0;
    float std_h = 0;

    for (uint i=0; i < ws.size();i++) {
        std_w += (mean_w - ws[i]) * (mean_w - ws[i]);
        std_h += (mean_h - hs[i]) * (mean_h - hs[i]);
    }
    std_w = std::sqrt(std_w / ws.size());
    std_h = std::sqrt(std_h / hs.size());

    cout << count << endl;
    cout << min_w << " " << mean_w << "/" << std_w << " " << max_w << endl;
    cout << min_h << " " << mean_h << "/" << std_h << " " << max_h << endl;

    int new_count = 0;
    for (uint i=0; i < ws.size();i++) {
        if (std::abs(ws[i] - mean_w) <= 2 * std_w && std::abs(hs[i] - mean_h) <= 2 * std_h)
            new_count++;
    }
    cout << new_count << endl;

    cout << mean_w - 2 * std_w << " " << mean_w + 2 * std_w << endl;
    cout << mean_h - 2 * std_h << " " << mean_h + 2 * std_h << endl;

    Size minSize;
    minSize.width = (int) round(mean_w - 2 * std_w);
    minSize.height = (int) round(mean_h - 2 * std_h);
    cout << "minSize:" << minSize << endl;
    Size maxSize;
    maxSize.width = (int) round(mean_w + 2 * std_w);
    maxSize.width += resizeFactor - maxSize.width % resizeFactor; // make it a multiple of resizeFactor
    maxSize.height = (int) round(mean_h + 2 * std_h);
    maxSize.height += resizeFactor - maxSize.height % resizeFactor;
    cout << "maxSize:" << maxSize << endl;
    sizes.push_back(minSize);
    sizes.push_back(maxSize);
    return sizes;
}

void prepareImages(vector<Writer> &writers, vector<Size> &sizes) {
    Size minSize = sizes[0];
    Size maxSize = sizes[1];
    Size finalSize = maxSize / resizeFactor;
    cout << "finalSize:" << finalSize << endl;

    for (Writer &writer: writers) {
        vector<Mat> newImages;

        for (Mat &image: writer.images) {
            if (image.cols < minSize.width || image.cols > maxSize.width || image.rows < minSize.height || image.rows > maxSize.height)
                continue;

            Mat invImage = Scalar::all(1) - image;
            int dw = max(1, maxSize.width - image.cols);
            int dh = max(1, maxSize.height - image.rows);
            int cw, ch;
            int x = (int) round(sqrt(maxNewImagesPerImage));
            if (dh < dw) {
                ch = dh < x ? dh : x;
                cw = (int) floor(maxNewImagesPerImage / ch);
            } else {
                cw = dw < x ? dw : x;
                ch = (int) floor(maxNewImagesPerImage / cw);
            }
            int iw = max(1, (int) floor(dw/cw));
            int ih = max(1, (int) floor(dh/ch));
            cout << dw << " " << dh << " " << cw << " " << ch << " " << iw << " " << ih << " " << (dw * dh) / (iw * ih) << endl;
            for (uint r = 0; r < maxSize.height - image.rows; r+=ih) {
                for (uint c = 0; c < maxSize.width - image.cols; c+=iw) {
                    Mat newImage = Mat::zeros(maxSize, CV_32F);
                    invImage.copyTo(newImage(Rect(c, r, image.cols, image.rows)));
                    resize(newImage, newImage, finalSize);
                    //imshow("", newImage);
                    //waitKey();

                    newImages.push_back(newImage);
               }
            }
        }

        writer.images = newImages;
        cout << "writerImages[" << writer.id << "] = " << newImages.size() << endl;
    }
}

vector<Writer> selectTrainWriters(vector<Writer> &writers) {
    vector<Writer> trainWriters;

    for (int i = 2; i < writers.size(); ++i) {
        if (writers[i].images.size() >= maxNewImagesPerImage)
            trainWriters.push_back(writers[i]);
        if (trainWriters.size() == 2)
            break;
    }

    return trainWriters;
}

vector<Writer> selectTrainData(vector<Writer> &writers) {
    vector<Writer> trainWriters;
    for (Writer &writer: writers) {
        uniform_int_distribution<int> dist(0, writer.images.size() - 1);

        Writer newWriter;
        newWriter.id = writer.id;
        for (int i = 0; i < 1000; ++i) {
            int index = dist(generator);
            if (i % 10 == 0)
                cout << index << " ";
            newWriter.images.push_back(writer.images[index]);
        }
        cout << endl;
        trainWriters.push_back(newWriter);
    }
    return trainWriters;
}

Mat computeMean(vector<Writer> &writers) {
    Mat meanImage(writers[0].images[0].rows, writers[0].images[0].cols, CV_32F);
    for (Writer &writer: writers) {
        for (Mat &image: writer.images)
            meanImage += image;
    }
    meanImage /= writers[0].images.size() + writers[1].images.size();
    //imshow("", meanImage);
    //waitKey();
    return meanImage;
}

void MatToDatum(const Mat &image, int label, Datum &datum)
{
    assert(image.type() == CV_32F);
    datum.set_channels(1);
    datum.set_height(image.rows);
    datum.set_width(image.cols);
    datum.set_label(label);
    datum.clear_float_data();
    google::protobuf::RepeatedField<float>* data_float = datum.mutable_float_data();
    for (int r = 0; r < image.rows; ++r)
        for (int c = 0; c < image.cols; ++c)
            data_float->Add(image.at<float>(r, c));
}

void createDB(vector<Writer> &writers, Mat &meanImage, string dbName) {
    cout << "creating db " << dbName << endl;

    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;

    filesystem::remove_all(filesystem::path(dbName));
    assert(leveldb::DB::Open(options, dbName, &db).ok());

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();

    Datum datum;
    string value;

    int count = 0;
    for (uint i = 0; i < writers.size(); i++) {
        Writer writer = writers[i];
        for (uint j = 0; j < writer.images.size(); j++)
        {
            stringstream ss;
            ss << count++;
            Mat image = writer.images[j];

            image -= meanImage;
            MatToDatum(image, i, datum);
            datum.SerializeToString(&value);
            batch->Put("pos_" + ss.str(), value);
        }
    }

    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;
}

int main(int argc, char *argv[])
{
    string word = argv[1];

    vector<Writer> writers = readWriters(word);
    vector<Size> sizes = getInputSizes(writers);
    prepareImages(writers, sizes);
    writers = selectTrainWriters(writers);
    cout << writers[0].images.size() << " " << writers[1].images.size() << endl;
    vector<Writer> trainWriters = selectTrainData(writers);
    cout << trainWriters[0].images.size() << " " << trainWriters[1].images.size() << endl;
    vector<Writer> testWriters = selectTrainData(writers);

    Mat meanImage = computeMean(trainWriters);
    createDB(trainWriters, meanImage, "train_db");
    createDB(testWriters, meanImage, "test_db");
}
