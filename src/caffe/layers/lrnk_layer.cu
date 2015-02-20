#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
                                 const Dtype* scale, const Dtype negative_beta,
                                 Dtype* out);


template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads, const Dtype* bottom_data,
                               const Dtype* top_data, const Dtype* scale,
                               const Dtype* top_diff, const int num,
                               const int channels, const int height,
                               const int width, const int size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               Dtype* bottom_diff);


template <typename Dtype>
__global__ void LRNKFillScale(const int nthreads, const Dtype* in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha,
    Dtype* scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 2. + accum_scale * alpha;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 2. + accum_scale * alpha;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 2. + accum_scale * alpha;
      ++head;
    }
  }
}


template <typename Dtype>
void LRNKLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = this->scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = this->num_ * this->height_ * this->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNKFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, this->num_, this->channels_,
      this->height_, this->width_, this->size_,
      this->alpha_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -this->beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNKLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNKLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


template <typename Dtype>
void LRNKLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int n_threads = this->num_ * this->height_ * this->width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      this->scale_.gpu_data(), top[0]->gpu_diff(), this->num_,
      this->channels_, this->height_, this->width_,
      this->size_, -this->beta_, Dtype(2. * this->alpha_ * this->beta_),
      bottom[0]->mutable_gpu_diff());
}
template void LRNKLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNKLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);


}  // namespace caffe
