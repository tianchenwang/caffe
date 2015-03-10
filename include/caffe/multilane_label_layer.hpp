#ifndef CAFFE_MULTILANE_LABEL_LAYER_HPP_
#define CAFFE_MULTILANE_LABEL_LAYER_HPP_


#include "caffe/data_layers.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <boost/numpy.hpp>

namespace caffe {


/**
 * @brief Provides multilane labels to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MultilaneLabelLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultilaneLabelLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultilaneLabelLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VideoData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleBatches();
  virtual void InternalThreadEntry();
  //boost::numpy::ndarray ReadLabelBatch(const string& filename, std::vector<size_t> &frame_ids, std::vector<size_t> trans, const int height, const int width);

  vector<std::pair<std::string, std::pair<std::vector<int>, std::vector<int> > > > lines_;
  int lines_id_;
  bool firstTime;
  boost::python::object pyModule;                                              
  boost::python::object pyClass;                                          
  boost::python::object pyReader;
};


}  // namespace caffe

#endif  // CAFFE_MULTILANE_LABEL_LAYER_HPP_
