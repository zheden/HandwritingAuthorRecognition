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

string input_path = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-set1-train2";

default_random_engine generator;

struct ImgIndx {
    int writerIndex;
    int imageIndex;
};

struct Writer {
    string id;
    vector<Mat> images;
};

void readWriter(filesystem::path writer_dir, Writer &writer) {
    cout << "\treading writer " << writer.id << endl;
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
            cout << "read " << fileName << endl;
        }
    }
}

vector<Writer> readWriters() {
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

Mat computeMean(vector<Writer> &writers) {
    Mat meanImage(writers[0].images[0].rows, writers[0].images[0].cols, CV_32F);
    int count = 0;
    for (Writer &writer: writers) {
        for (Mat &image: writer.images) {
            meanImage += image;
            count++;
        }
    }
    meanImage /= count;

    ofstream mean_out("train_mean.bin", ofstream::binary );
    mean_out.write((char*) meanImage.data, meanImage.cols * meanImage.rows * sizeof(float));

//    imshow("", meanImage);
//    waitKey();
    return meanImage;
}

Datum makeTriplet(int label, const Mat &image, const Mat &similar, const Mat &other)
{
    Datum datum;
    assert(image.type() == CV_32F);
    datum.set_label(label);
    datum.set_channels(3);
    datum.set_height(image.rows);
    datum.set_width(image.cols);
    datum.clear_float_data();
    google::protobuf::RepeatedField<float>* data_float = datum.mutable_float_data();
    for (int i = 0; i < 3; i++) {
        const Mat &mat = (i == 0) ? image: ((i == 1) ? similar : other);
        for (int r = 0; r < mat.rows; ++r)
            for (int c = 0; c < mat.cols; ++c)
                data_float->Add(mat.at<float>(r, c));
    }
    return datum;
}

void showImages(const Mat &image1, const Mat &image2, const Mat &image3) {
    Mat out(image1.rows,image1.cols*3,CV_32F);
    image1.copyTo(out(Rect(0*image1.cols,0,image1.cols,image1.rows)));
    image2.copyTo(out(Rect(1*image1.cols,0,image1.cols,image1.rows)));
    image3.copyTo(out(Rect(2*image1.cols,0,image1.cols,image1.rows)));
    imshow("",out);
    waitKey();
}

vector<vector<ImgIndx>> createTable(int currentWriter, int writers, int images) {
    vector<vector<ImgIndx>> indexes;

    uniform_int_distribution<int> dist_writer(0, writers - 1);
    uniform_int_distribution<int> dist_img(0, images - 1);

    // initialize
    for (int i = 0; i < images; ++i) {
        vector<ImgIndx> row;
        for (int j = 0; j < images; ++j) {
            ImgIndx index;
            index.writerIndex = -1;
            index.imageIndex = -1;
            row.push_back(index);
        }
        indexes.push_back(row);
    }

    // select same text oposites
    for (int i = 0; i < images; ++i) {
        int j = 0;
        do {
            if (j == currentWriter) {
                j++;
                continue;
            }

            int pos = dist_img(generator);
            if (pos != i && indexes[i][pos].writerIndex == -1) {
                indexes[i][pos].writerIndex = j;
                indexes[i][pos].imageIndex = i;
                j++;
            }
        } while (j < writers);
    }

//    for (int i = 0; i < images; ++i) {
//        for (int j = 0; j < images; ++j)
//            cout << indexes[i][j].writerIndex << ":" << indexes[i][j].imageIndex << " ";
//        cout << endl;
//    }

    // fill up table
    for (int i = 0; i < images; ++i) {
        int j = 0;
        do {
            if (j == i || indexes[i][j].writerIndex != -1) {
                j++;
                continue;
            }

            int writer = dist_writer(generator);
            if (writer != currentWriter) {
                indexes[i][j].writerIndex = writer;
                indexes[i][j].imageIndex = dist_img(generator);
                j++;
            }
        } while (j < images);
    }

//    for (int i = 0; i < images; ++i) {
//        for (int j = 0; j < images; ++j)
//            cout << indexes[i][j].writerIndex << ":" << indexes[i][j].imageIndex << " ";
//        cout << endl;
//    }

    return indexes;
}

void createDB(vector<Writer> &writers, Mat &meanImage) {
    string dbName = "train_db";
    cout << "creating db " << dbName << endl;

    leveldb::DB* db;
    leveldb::Options options;
    options.create_if_missing = true;
    options.write_buffer_size = 268435456;

    filesystem::remove_all(filesystem::path(dbName));
    assert(leveldb::DB::Open(options, dbName, &db).ok());

    leveldb::WriteBatch* batch = new leveldb::WriteBatch();

    string value;

    int counter = 0;
    for (uint i = 0; i < writers.size(); i++) {
        Writer writer = writers[i];
        int writerId = atoi(writer.id.c_str());

        cout << "creating triplets for writer " << writerId << endl;

        vector<vector<ImgIndx>> others = createTable(i, writers.size(), writer.images.size());
        for (uint j = 0; j < writer.images.size(); j++) {
            Mat image = writer.images[j];
            Mat unmeanImage = image - meanImage;

            for (uint k = 0; k < writer.images.size(); k++) {
                if (k == j)
                    continue;

                Mat similarImage = writer.images[k];
                Mat unmeanSimilar = similarImage - meanImage;

                ImgIndx index = others[j][k];
                Mat otherImage = writers[index.writerIndex].images[index.imageIndex];
                Mat unmeanOther = otherImage - meanImage;

                //showImages(image, similarImage, otherImage);

                Datum datum = makeTriplet(writerId, unmeanImage, unmeanSimilar, unmeanOther);
                datum.SerializeToString(&value);
                stringstream ss;
                ss << counter++;
                batch->Put("t" + ss.str(), value);

                if ((counter % 100) == 0) // write to disk after enough puts
                {
                    db->Write(leveldb::WriteOptions(), batch);
                    delete batch;
                    batch = new leveldb::WriteBatch();
                }
            }
        }
    }

    db->Write(leveldb::WriteOptions(), batch);
    delete batch;
    delete db;
}

int main(int argc, char *argv[])
{
    vector<Writer> writers = readWriters();
    Mat meanImage = computeMean(writers);
//    for (int r = 0; r < meanImage.rows; ++r)
//        for (int c = 0; c < meanImage.cols; ++c)
//            cout << meanImage.at<float>(r, c) << endl;
    createDB(writers, meanImage);
}
