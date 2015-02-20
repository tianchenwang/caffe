#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LRNKLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = this->scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < this->scale_.count(); ++i) {
    scale_data[i] = 2.;
  }
  Blob<Dtype> padded_square(1, this->channels_ + this->size_ - 1,
                            this->height_, this->width_);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  caffe_set(padded_square.count(), Dtype(0), padded_square_data);
  // go through the images
  for (int n = 0; n < this->num_; ++n) {
    // compute the padded square
    caffe_sqr(this->channels_ * this->height_ * this->width_,
        bottom_data + bottom[0]->offset(n),
        padded_square_data + padded_square.offset(0, this->pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < this->size_; ++c) {
      caffe_axpy<Dtype>(this->height_ * this->width_, this->alpha_,
          padded_square_data + padded_square.offset(0, c),
          scale_data + this->scale_.offset(n, 0));
    }
    for (int c = 1; c < this->channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(this->height_ * this->width_,
          scale_data + this->scale_.offset(n, c - 1),
          scale_data + this->scale_.offset(n, c));
      // add head
      caffe_axpy<Dtype>(this->height_ * this->width_, this->alpha_,
          padded_square_data + padded_square.offset(0, c + this->size_ - 1),
          scale_data + this->scale_.offset(n, c));
      // subtract tail
      caffe_axpy<Dtype>(this->height_ * this->width_, -this->alpha_,
          padded_square_data + padded_square.offset(0, c - 1),
          scale_data + this->scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(this->scale_.count(), scale_data, -this->beta_, top_data);
  caffe_mul<Dtype>(this->scale_.count(), top_data, bottom_data, top_data);
}


template <typename Dtype>
void LRNKLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = this->scale_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Blob<Dtype> padded_ratio(1, this->channels_ + this->size_ - 1,
                           this->height_, this->width_);
  Blob<Dtype> accum_ratio(1, 1, this->height_, this->width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
//  Dtype cache_ratio_value = 2. * this->alpha_ * this->beta_ / this->size_;
  Dtype cache_ratio_value = 2. * this->alpha_ * this->beta_;

  caffe_powx<Dtype>(this->scale_.count(), scale_data,
                    -this->beta_, bottom_diff);
  caffe_mul<Dtype>(this->scale_.count(), top_diff,
                   bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = this->size_ - (this->size_ + 1) / 2;
  for (int n = 0; n < this->num_; ++n) {
    int block_offset = this->scale_.offset(n);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(this->channels_ * this->height_ * this->width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    caffe_div<Dtype>(this->channels_ * this->height_ * this->width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    // Now, compute the accumulated ratios and the bottom diff
    caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
    for (int c = 0; c < this->size_ - 1; ++c) {
      caffe_axpy<Dtype>(this->height_ * this->width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
    for (int c = 0; c < this->channels_; ++c) {
      caffe_axpy<Dtype>(this->height_ * this->width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + this->size_ - 1),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(this->height_ * this->width_,
          bottom_data + top[0]->offset(n, c),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(this->height_ * this->width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
      caffe_axpy<Dtype>(this->height_ * this->width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(LRNKLayer);
STUB_GPU_FORWARD(LRNKLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNKLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRNKLayer);
REGISTER_LAYER_CLASS(LRNK);


}  // namespace caffe
