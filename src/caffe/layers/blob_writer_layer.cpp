#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"
namespace caffe {

template <typename Dtype>
void BlobWriterLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  batch_size_ = bottom[0]->num();
  prefix_ = this->layer_param_.blob_writer_param().prefix();
  iter_ = 0;
}

template <typename Dtype>
void BlobWriterLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(top->size(), 1)
      << "top must be a dummy blob of size 1";
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void BlobWriterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  BlobProtoVector blob_proto_vec;
  std::ostringstream stringStream;
  stringStream <<prefix_<< "_batch"<<iter_<<".proto";                          
  std::string save_name = stringStream.str();
  for (int i=0; i<bottom.size(); ++i)
  {
    bottom[i]->ToProto(blob_proto_vec.add_blobs());
  }
  WriteProtoToBinaryFile(blob_proto_vec, save_name.c_str());
  blob_proto_vec.Clear();
  iter_++;
}

INSTANTIATE_CLASS(BlobWriterLayer);

}  // namespace caffe
