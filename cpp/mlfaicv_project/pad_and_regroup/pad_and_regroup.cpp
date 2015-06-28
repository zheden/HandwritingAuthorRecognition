#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

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
string input_path = root_dir + "new-data-set1-small";
string output_path = root_dir + "new-data-set1-writers";

struct Image {
    Mat mat;
    filesystem::path path;
};

void processImages(filesystem::path word_dir) {
    cout << "processing word: " << word_dir.leaf().string() << endl;

    int max_w = 0, max_h = 0;
    vector<Image> images;
    filesystem::recursive_directory_iterator end_iter;
    for(filesystem::recursive_directory_iterator dir_iter(word_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_regular_file(dir_iter->status())) {
            filesystem::path filePath = dir_iter->path();
            string fileName = filePath.string();
            if (fileName.rfind(".png") == string::npos)
                continue;

            Mat mat = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
            Image image;
            image.mat = mat;
            image.path = filePath;
            images.push_back(image);

            max_w = std::max(max_w, mat.cols);
            max_h = std::max(max_h, mat.rows);
        }
    }

    for (Image &image: images) {
        Mat newImage = Mat::zeros(max_h, max_w, CV_8U);
        int c = (max_w - image.mat.cols) / 2;
        int r = (max_h - image.mat.rows) / 2;
        image.mat.copyTo(newImage(Rect(c, r, image.mat.cols, image.mat.rows)));

        string new_image_dir = output_path + "/" + image.path.parent_path().leaf().string() +
                "/" + image.path.parent_path().parent_path().leaf().string();
        filesystem::path dirPath(new_image_dir);
        filesystem::create_directories(dirPath);

        string new_image_file = new_image_dir + "/" + image.path.leaf().string();
        imwrite(new_image_file, newImage);
        cout << new_image_file << " with size: [" << newImage.cols << " x " << newImage.rows << "]" << endl;
    }
}

int main(int argc, char *argv[])
{
    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path word_dir = dir_iter->path();
            processImages(word_dir);
        }
    }
}
