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
string input_path = root_dir + "new-data-set1-imgs";
string output_path = root_dir + "new-data-set1-train1";

int selection = 50;
int total = 2137;

default_random_engine generator;
uniform_int_distribution<int> dist_img(0, total);

int main(int argc, char *argv[])
{
    vector<string> writers;

    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            writers.push_back(dir_iter->path().leaf().string());
        }
    }

    writers.erase(writers.end() - 1);
    for (int i = 0; i < selection; ++i) {
        int imgNr = dist_img(generator);
        stringstream ss;
        ss << imgNr;
        for (string writer : writers) {
            string imgFile = input_path + "/" + writer + "/img" + ss.str() + ".png";
            filesystem::path inputFile(imgFile);
            string destDir = output_path + "/" + writer;
            filesystem::path dirPath(destDir);
            filesystem::create_directories(dirPath);
            filesystem::path outputFile(destDir + "/img" + ss.str() + ".png");
            filesystem::copy(inputFile, outputFile);
        }
    }
}
