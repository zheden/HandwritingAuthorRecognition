#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <boost/filesystem.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace boost;

string root_dir = "/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/";
string input_path = root_dir + "new-data-set1-test-small";
string output_path = root_dir + "new-data-set1-test-imgs";

default_random_engine generator;

struct Writer {
    string id;
    vector<Mat> images;
};

int w = 200, h = 70;
int counter = 0;
int initial_space = 10;
int spacing = 30;

void saveImage(Writer &writer, Mat &textImg) {
    Mat img = Mat::zeros(h, w, CV_8U);
    textImg(Rect(0, 0, w, h)).copyTo(img);

    string new_image_dir = output_path + "/" + writer.id;
    filesystem::path dirPath(new_image_dir);
    filesystem::create_directories(dirPath);

    stringstream ss;
    ss << counter++;
    string new_image_file = new_image_dir + "/img" + ss.str() + ".png";

    imwrite(new_image_file, img);
}

void appendPaddedOrCut(Mat &img, Mat &word_img, int pos) {
    //cout << img.cols << "x" << img.rows << "+" << word_img.cols << "x" << word_img.rows << "@" << pos << endl;
    if (h < word_img.rows)
        word_img(Rect(0, (word_img.rows - h) / 2, word_img.cols, h)).copyTo(img(Rect(pos, 0, word_img.cols, h)));
    else
        word_img.copyTo(img(Rect(pos, (h - word_img.rows) / 2, word_img.cols, word_img.rows)));
}

Mat makeFirstImage(Mat &word_img) {
    Mat img = Mat::zeros(h, word_img.cols + spacing, CV_8U);
    appendPaddedOrCut(img, word_img, initial_space);
    return img;
}

Mat mergeImages(Mat &img, Mat &word_img) {
    Mat new_img = Mat::zeros(h, img.cols + word_img.cols + spacing, CV_8U);
    img.copyTo(new_img(Rect(0, 0, img.cols, img.rows)));
    appendPaddedOrCut(new_img, word_img, img.cols + spacing);
    return new_img;
}

void createImages(vector<Writer> &writers, int counter) {
    // add randomly until above 200 and then cut to size
    Mat img;
    for (Writer &writer: writers) {
        cout << "creating " << counter << " images for writer " << writer.id << endl;
        uniform_int_distribution<int> dist_img(0, writer.images.size() - 1);
        for (int i = 0; i < counter; ++i) {
            int img_w = 0;
            while (img_w < w) {
                Mat word_img = writer.images[dist_img(generator)];
                if (img_w == 0)
                    img = makeFirstImage(word_img);
                else
                    img = mergeImages(img, word_img);
                img_w = img.cols;
            }
            saveImage(writer, img);
        }
    }
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
            writer.images.push_back(image);
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

int main(int argc, char *argv[])
{
    vector<Writer> writers = readWriters();
    createImages(writers, atoi(argv[1]));
}
