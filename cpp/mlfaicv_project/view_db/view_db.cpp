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

void DatumToMat(const Datum &datum, Mat &img)
{
    const float* data = datum.float_data().data();
    int counter=0;
    assert(datum.channels() == 1);
    img = Mat(datum.height(),datum.width(),CV_32F);
    for (int r = 0; r < img.rows; ++r)
        for (int c = 0; c < img.cols; ++c)
        {
            img.at<float>(r,c) = data[counter];
            counter++;
        }
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
        Mat temp;
        Datum datum;
        datum.ParseFromString(value);
        DatumToMat(datum,temp);
        imshow("", temp);
        waitKey();

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
