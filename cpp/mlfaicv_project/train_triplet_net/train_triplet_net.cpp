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

void trainNet() {
    Caffe::set_mode(Caffe::CPU);
    caffe::SolverParameter solver_param;
    solver_param.set_base_lr(0.001f);
    solver_param.set_momentum(0.9);
    //solver_param.set_weight_decay(0.0001);

    solver_param.set_solver_type(SolverParameter_SolverType_SGD);

    solver_param.set_stepsize(100);
    solver_param.set_lr_policy("step");
    solver_param.set_gamma(0.1);

    solver_param.set_solver_mode(SolverParameter_SolverMode_CPU);

    solver_param.set_max_iter(1000);

    //solver_param.set_test_interval(100);

    // This number should be #testsamples/#testbatches
    //solver_param.add_test_iter(10) ;//testImagesCount / 100);

    solver_param.set_snapshot(100);
    solver_param.set_snapshot_prefix("writer_triplet");

    solver_param.set_display(20);

    solver_param.set_net("writer_triplet.prototxt");

    boost::shared_ptr<caffe::Solver<float> > solver(caffe::GetSolver<float>(solver_param));
    solver->Solve();
}

int main(int argc, char *argv[])
{
    trainNet();
}
