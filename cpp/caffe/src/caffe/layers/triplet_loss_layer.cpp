#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_same_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_other_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_same_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_other_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  uncut_loss_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // xi
      bottom[1]->cpu_data(),  // xj
      diff_same_.mutable_cpu_data());  // xi_i-xj_i
  caffe_sub(
      count,
      bottom[0]->cpu_data(), // xi
      bottom[2]->cpu_data(), // xk
      diff_other_.mutable_cpu_data()); // xi_i-xk_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_same_.mutable_cpu_data()[i] = sqrt(caffe_cpu_dot(channels,
        diff_same_.cpu_data() + (i*channels), diff_same_.cpu_data() + (i*channels)));
    dist_other_.mutable_cpu_data()[i] = sqrt(caffe_cpu_dot(channels,
        diff_other_.cpu_data() + (i*channels), diff_other_.cpu_data() + (i*channels)));
    uncut_loss_.mutable_cpu_data()[i] = 1 - dist_other_.cpu_data()[i] / (margin + dist_same_.cpu_data()[i]);
    loss += std::max(uncut_loss_.cpu_data()[i], Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
          Dtype* bout = bottom[i]->mutable_cpu_diff();
          if (uncut_loss_.cpu_data()[j] > Dtype(0.0)) {
              Dtype d_same = dist_same_.cpu_data()[j]; // B
              Dtype d_other = dist_other_.cpu_data()[j]; // A
              // compute gradients
              if (i == 0) { // gradient for xi = (A * A * diffB - B * (m + B) * diffA) / (A * B * (m + B) * (m + B))
                  Dtype denominator = d_same * d_other * (margin + d_same) * (margin + d_same);
                  Dtype beta = alpha * d_same * (margin + d_same) / denominator;
                  caffe_cpu_axpby(
                      channels,
                      beta,
                      diff_other_.cpu_data() + (j*channels),
                      Dtype(0.0),
                      bout + (j*channels));
                  Dtype gamma = alpha * d_other * d_other / denominator;
                  caffe_cpu_axpby(
                      channels,
                      gamma,
                      diff_same_.cpu_data() + (j*channels),
                      Dtype(-1.0),
                      bout + (j*channels));
              } else if (i == 1) { // gradient for xj = (A * diffB) / (B * (m + B) * (m + B))
                 Dtype denominator = d_same * (margin + d_same) * (margin + d_same);
                 Dtype beta = alpha * d_other / denominator;
                 caffe_cpu_axpby(
                     channels,
                     beta,
                     diff_same_.cpu_data() + (j*channels),
                     Dtype(0.0),
                     bout + (j*channels));
              } else { // gradient for xk = (-1 * diffA) / (A * (m + B))
                  Dtype denominator = d_other * (margin + d_same);
                  Dtype beta = -alpha / denominator;
                  caffe_cpu_axpby(
                      channels,
                      beta,
                      diff_other_.cpu_data() + (j*channels),
                      Dtype(0.0),
                      bout + (j*channels));
              }
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
