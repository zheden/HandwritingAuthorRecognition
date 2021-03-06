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
string input_path = root_dir + "all-writers2";
string output_path = root_dir + "all-writers2-small";
int resizeFactor = 2;

void processImages(filesystem::path writer_dir) {
    cout << "processing writer: " << writer_dir.leaf().string() << endl;

    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(writer_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_regular_file(dir_iter->status())) {
            filesystem::path filePath = dir_iter->path();
            string fileName = filePath.string();
            if (fileName.rfind(".png") == string::npos)
                continue;

            Mat mat = Scalar::all(255) - imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
            if (mat.cols < 5 || mat.rows < 5)
                continue;
            Size matSize(mat.cols, mat.rows);
            Size finalSize = matSize / resizeFactor;
            resize(mat, mat, finalSize);

            string new_image_dir = output_path + "/" + filePath.parent_path().leaf().string();
            filesystem::path dirPath(new_image_dir);
            filesystem::create_directories(dirPath);

            string new_image_file = new_image_dir + "/" + filePath.leaf().string();
            imwrite(new_image_file, mat);
            //cout << new_image_file << " with size: [" << mat.cols << " x " << mat.rows << "]" << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    filesystem::path input_dir(input_path);
    filesystem::directory_iterator end_iter;
    for(filesystem::directory_iterator dir_iter(input_dir); dir_iter != end_iter ; ++dir_iter) {
        if (filesystem::is_directory(dir_iter->status())) {
            filesystem::path writer_dir = dir_iter->path();
            processImages(writer_dir);
            //cout << "process " << word_dir << endl;
        }
    }

}
