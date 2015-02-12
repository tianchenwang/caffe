#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void DrivingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data. This assumes that all data come before labels.
  for (int i = 0; i < prefetch_datas_.size(); ++i) {
    caffe_copy(prefetch_datas_[i]->count(), prefetch_datas_[i]->cpu_data(),
        (*top)[i]->mutable_gpu_data());
  }
  if (this->output_labels_) {
    for (int i = 0; i < prefetch_labels_.size(); ++i) {
      const int offset_idx = prefetch_datas_.size() + i;
      caffe_copy(prefetch_labels_[i]->count(), prefetch_labels_[i]->cpu_data(),
          (*top)[offset_idx]->mutable_gpu_data());
    }
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_CLASS(DrivingDataLayer);

}  // namespace caffe
