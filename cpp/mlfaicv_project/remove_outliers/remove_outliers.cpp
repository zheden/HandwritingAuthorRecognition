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
string input_path = root_dir + "new-data";
string output_path = root_dir + "new-data-clean";
int resizeFactor = 4;

struct Image {
    Mat mat;
    filesystem::path path;
};

void processImages(filesystem::path word_dir) {
    cout << "processing word: " << word_dir.leaf().string() << endl;

    vector<Image> images;
    filesystem::recursive_directory_iterator end_iter;
    for(filesystem::recursive_directory_iterator dir_iter(word_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_regular_file(dir_iter->status())) {
            filesystem::path filePath = dir_iter->path();
            string fileName = filePath.string();
            if (fileName.rfind(".png") == string::npos)
                continue;

            Mat mat = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
            mat.convertTo(mat, CV_32F, 1/255.f);
            //cout << filePath.leaf().string() << " with size: [" << mat.cols << " x " << mat.rows << "]" << endl;
            Image image;
            image.mat = mat;
            image.path = filePath;
            images.push_back(image);
        }
    }

    int min_w = 1000000, min_h = 1000000;
    int max_w = 0, max_h = 0;
    int sum_w = 0, sum_h = 0;
    int count = 0;

    for (Image &imageFile: images) {
        Mat image = imageFile.mat;
        min_w = std::min(min_w, image.cols);
        max_w = std::max(max_w, image.cols);
        sum_w += image.cols;
        min_h = std::min(min_h, image.rows);
        max_h = std::max(max_h, image.rows);
        sum_h += image.rows;
        count++;
    }

    float mean_w = sum_w / (float) count;
    float mean_h = sum_h / (float) count;

    float std_w = 0;
    float std_h = 0;

    for (Image &imageFile: images) {
        Mat image = imageFile.mat;
        std_w += (mean_w - image.cols) * (mean_w - image.cols);
        std_h += (mean_h - image.rows) * (mean_h - image.rows);
    }
    std_w = std::sqrt(std_w / (images.size() - 1));
    std_h = std::sqrt(std_h / (images.size() - 1));

    cout << "width: " << min_w << " " << mean_w << "/" << std_w << " " << max_w << endl;
    cout << "height: " << min_h << " " << mean_h << "/" << std_h << " " << max_h << endl;

    int std_num = 3;
    for (Image &imageFile: images) {
        Mat image = imageFile.mat;
        if (std::abs(image.cols - mean_w) <= std_num * std_w && std::abs(image.rows - mean_h) <= std_num * std_h) {
            string new_image_dir = output_path + "/" + imageFile.path.parent_path().parent_path().leaf().string()
                    + "/" + imageFile.path.parent_path().leaf().string();
            filesystem::path dirPath(new_image_dir);
            filesystem::create_directories(dirPath);
            filesystem::path newFilePath(new_image_dir + "/" + imageFile.path.leaf().string());
            filesystem::copy(imageFile.path, newFilePath);
        } else {
            cout << "outlier: " << imageFile.path << endl;
        }
    }

    Size maxSize;
    maxSize.width = (int) round(mean_w + std_num * std_w);
    maxSize.width += resizeFactor - maxSize.width % resizeFactor; // make it a multiple of resizeFactor
    maxSize.height = (int) round(mean_h + std_num * std_h);
    maxSize.height += resizeFactor - maxSize.height % resizeFactor;
    cout << "maxSize:" << maxSize << endl;
}

int main(int argc, char *argv[])
{
    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path word_dir = dir_iter->path();
            processImages(word_dir);
            //cout << "process " << word_dir << endl;
        }
    }
}
