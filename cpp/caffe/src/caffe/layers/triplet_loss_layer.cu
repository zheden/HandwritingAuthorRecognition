#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // xi
      bottom[1]->gpu_data(),  // xj
      diff_same_.mutable_gpu_data());  // xi_i-xj_i
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // xi
      bottom[2]->gpu_data(),  // xk
      diff_other_.mutable_gpu_data());  // xi_i-xk_i
  caffe_gpu_powx(
      count,
      diff_same_.gpu_data(),  // xi_i-xj_i
      Dtype(2),
      diff_same_sq_.mutable_gpu_data());  // (xi_i-xj_i)^2
  caffe_gpu_powx(
      count,
      diff_other_.gpu_data(),  // xi_i-xk_i
      Dtype(2),
      diff_other_sq_.mutable_gpu_data());  // (xi_i-xk_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_same_sq_.gpu_data(),  // (xi_i-xj_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_same_sq_.mutable_gpu_data());  // \Sum (xi_i-xj_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_other_sq_.gpu_data(),  // (xi_i-xk_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_other_sq_.mutable_gpu_data());  // \Sum (xi_i-xk_i)^2
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_same_.mutable_cpu_data()[i] = sqrt(dist_same_sq_.cpu_data()[i]);
    dist_other_.mutable_cpu_data()[i] = sqrt(dist_other_sq_.cpu_data()[i]);
    uncut_loss_.mutable_cpu_data()[i] = 1 - dist_other_.cpu_data()[i] / (margin + dist_same_.cpu_data()[i]);
    loss += std::max(uncut_loss_.cpu_data()[i], Dtype(0.0));
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackwardXi(const int count, const int channels,
    const Dtype margin, const Dtype alpha, const Dtype* diff_same, const Dtype* dist_same,
    const Dtype* diff_other, const Dtype* dist_other, const Dtype* uncut_loss, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int j = i / channels;  // the num index, to access dist and uncut_loss
    if (uncut_loss[j] > 0) {  // similar pairs
      Dtype d_same = dist_same[j];
      Dtype d_other = dist_other[j];
      bottom_diff[i] = alpha * (d_other * d_other * diff_same[i] - d_same * (margin + d_same) * diff_other[i])
              / (d_same * d_other * (margin + d_same) * (margin + d_same));
    } else {
      bottom_diff[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void CLLBackwardXj(const int count, const int channels,
    const Dtype margin, const Dtype alpha, const Dtype* diff_same, const Dtype* dist_same,
    const Dtype* dist_other, const Dtype* uncut_loss, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int j = i / channels;  // the num index, to access dist and uncut_loss
    if (uncut_loss[j] > 0) {  // similar pairs
      Dtype d_same = dist_same[j];
      bottom_diff[i] = alpha * dist_other[j] * diff_same[i] / (d_same * (margin + d_same) * (margin + d_same));
    } else {
      bottom_diff[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void CLLBackwardXk(const int count, const int channels,
    const Dtype margin, const Dtype alpha, const Dtype* dist_same,
     const Dtype* diff_other, const Dtype* dist_other, const Dtype* uncut_loss, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int j = i / channels;  // the num index, to access dist and uncut_loss
    if (uncut_loss[j] > 0) {  // similar pairs
      bottom_diff[i] = -alpha * diff_other[i] / (dist_other[j] * (margin + dist_same[j]));
    } else {
      bottom_diff[i] = 0;
    }
  }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 3; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.triplet_loss_param().margin();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[0]->num());
      if (i == 0) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          CLLBackwardXi<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, channels, margin, alpha,
              diff_same_.gpu_data(),  // the cached eltwise difference between xi and xj
              dist_same_.gpu_data(),  // the cached distance between xi and xj
              diff_other_.gpu_data(), // the cached eltwise difference between xi and xk
              dist_other_.gpu_data(), // the cached distance between xi and xk
              uncut_loss_.gpu_data(), // the cached uncut loss
              bottom[i]->mutable_gpu_diff());
          CUDA_POST_KERNEL_CHECK;
      } else if (i == 1) {
          // NOLINT_NEXT_LINE(whitespace/operators)
          CLLBackwardXj<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, channels, margin, alpha,
              diff_same_.gpu_data(),  // the cached eltwise difference between xi and xj
              dist_same_.gpu_data(),  // the cached distance between xi and xj
              dist_other_.gpu_data(), // the cached distance between xi and xk
              uncut_loss_.gpu_data(), // the cached uncut loss
              bottom[i]->mutable_gpu_diff());
          CUDA_POST_KERNEL_CHECK;
      } else {
          // NOLINT_NEXT_LINE(whitespace/operators)
          CLLBackwardXk<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, channels, margin, alpha,
              dist_same_.gpu_data(),  // the cached distance between xi and xj
              diff_other_.gpu_data(), // the cached eltwise difference between xi and xk
              dist_other_.gpu_data(), // the cached distance between xi and xk
              uncut_loss_.gpu_data(), // the cached uncut loss
              bottom[i]->mutable_gpu_diff());
          CUDA_POST_KERNEL_CHECK;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
