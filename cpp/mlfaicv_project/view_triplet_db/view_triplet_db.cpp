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

void readTriplet(const Datum &datum, Mat &img1, Mat &img2, Mat &img3)
{
    const float* data = datum.float_data().data();
    int counter=0;
    assert(datum.channels() == 3);
    img1 = Mat(datum.height(),datum.width(),CV_32F);
    img2 = Mat(datum.height(),datum.width(),CV_32F);
    img3 = Mat(datum.height(),datum.width(),CV_32F);
    for (int i = 0; i < 3; i++) {
        Mat &img = (i == 0) ? img1 : ((i == 1) ? img2 : img3);
        for (int r = 0; r < img.rows; ++r)
            for (int c = 0; c < img.cols; ++c)
            {
                img.at<float>(r,c) = data[counter];
                counter++;
            }
    }
}

void showImages(const Mat &image1, const Mat &image2, const Mat &image3) {
    Mat out(image1.rows,image1.cols*3,CV_32F);
    image1.copyTo(out(Rect(0*image1.cols,0,image1.cols,image1.rows)));
    image2.copyTo(out(Rect(1*image1.cols,0,image1.cols,image1.rows)));
    image3.copyTo(out(Rect(2*image1.cols,0,image1.cols,image1.rows)));
    imshow("",out);
    waitKey();
}

void viewDB(string path) {
    // Count number of test images
    leveldb::DB *test_db;
    leveldb::Options options;
    assert(leveldb::DB::Open(options, path, &test_db).ok());
    leveldb::Iterator *it = test_db->NewIterator(leveldb::ReadOptions());
    it->SeekToFirst();
    int test_samples = 0;

    while(it->Valid())
    {
        string value = it->value().ToString();
        Mat temp1, temp2, temp3;
        Datum datum;
        datum.ParseFromString(value);
        readTriplet(datum,temp1, temp2, temp3);
        //showImages(temp1, temp2, temp3);
        //cout << datum.label() << endl;

        test_samples++;
        it->Next();
    }

    delete it;
    delete test_db;
    cout << test_samples << endl;
}

int main(int argc, char *argv[])
{
    viewDB(argv[1]);
    return 0;
}
