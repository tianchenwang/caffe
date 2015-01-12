#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include "caffe/multilane_label_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/mpi.hpp"
#include <boost/algorithm/string.hpp>
#include <ctime>
namespace py = boost::python; // create namespace variable for boost::python  
namespace np = boost::numpy; // create namespace variable for boost::python  
namespace caffe {

//np::ndarray ReadLabelBatch(const string& filename, std::vector<size_t> &frame_ids, std::vector<size_t> &trans,
//    const int height, const int width) {
//};
template <typename Dtype>
MultilaneLabelLayer<Dtype>::~MultilaneLabelLayer<Dtype>() {
  this->JoinPrefetchThread();
}

py::object arr_handle;

template <typename Dtype>
void MultilaneLabelLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.multilane_label_param().new_height();
  const int new_width  = this->layer_param_.multilane_label_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multilane_label_param().source();
  LOG(INFO) << "Opening schedule file " << source;
  std::ifstream infile(source.c_str());



  string batch_string;

  string filename;
  //while (infile >> batch_string) {
  while (getline (infile, batch_string)) {
    if(!infile)
    {
      if(infile.eof())
      {
        LOG(INFO) << "Reached EOF of schedule file.";
        break;
      }
      else
        LOG(FATAL)<< "Error while reading schedule file. Possibly corrupted.";
    }
    std::vector<string> batch_fields;
    // first split a line into fields with delimiter ",". Fields should be [filename, frame_ids, transform_ids]
    boost::split(batch_fields, batch_string, boost::is_any_of(","),
                 boost::token_compress_on);
    if(batch_fields.size()!=3)
      LOG(FATAL) << "Each line must have 3 fields separated by comma, "
                 <<batch_fields.size()<<" found instead";
    // store filename
    filename = batch_fields[0];
    // store frame ids
    std::vector<string> frame_ids_str;
    std::vector<int> frame_ids;
    boost::split(frame_ids_str, batch_fields[1], boost::is_any_of(" "),
                 boost::token_compress_on);
    for (int f=0; f<frame_ids_str.size(); ++f) {
      frame_ids.push_back((int)atoi(frame_ids_str[f].c_str()));
    }

    // store persp transform ids
    std::vector<string> trans_ids_str;
    std::vector<int> trans_ids;
    boost::split(trans_ids_str, batch_fields[2], boost::is_any_of(" "),
                 boost::token_compress_on);
    for (int f=0; f<trans_ids_str.size(); ++f) {
      trans_ids.push_back((int)atoi(trans_ids_str[f].c_str()));
    }

    lines_.push_back(std::make_pair(filename, std::make_pair(frame_ids, trans_ids)));
  }

  if (this->layer_param_.multilane_label_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling batches";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleBatches();
  }
  LOG(INFO) << "A total of " << lines_.size() << " batches.";

  lines_id_ = mpiRank(MPI_COMM_WORLD);
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.multilane_label_param().rand_skip()) {
    //unsigned int skip = caffe_rng_rand() % this->layer_param_.multilane_label_param().rand_skip();
    unsigned int skip = this->layer_param_.multilane_label_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip+mpiRank(MPI_COMM_WORLD);
  }
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.multilane_label_param().batch_size();
  const bool predict_depth = this->layer_param_.multilane_label_param().predict_depth();
  const int blob_depth = predict_depth? 112:80; //TODO: make this changable in caffe.proto
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, blob_depth, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, blob_depth, crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, blob_depth, 15,20);
    this->prefetch_data_.Reshape(batch_size, blob_depth, 15,20);
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  //(*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = blob_depth;
  this->datum_height_ = 15;
  this->datum_width_ = 20;
  this->datum_size_ = blob_depth*15*20;
}

template <typename Dtype>
void MultilaneLabelLayer<Dtype>::ShuffleBatches() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}



// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultilaneLabelLayer<Dtype>::InternalThreadEntry() {
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  MultilaneLabelParameter multilane_label_param = this->layer_param_.multilane_label_param();
  const int batch_size = multilane_label_param.batch_size();
  const int new_height = multilane_label_param.new_height();
  const int new_width = multilane_label_param.new_width();

  // datum scales
  const int lines_size = lines_.size();
  string filename = lines_[lines_id_].first;
  std::vector<int> frameIds = lines_[lines_id_].second.first;
  std::vector<int> trans = lines_[lines_id_].second.second;
  if (batch_size!=frameIds.size() || batch_size!=trans.size())
    LOG(ERROR)<<"Frame count mismatch!";
  if(mpiRank(MPI_COMM_WORLD)==0)
    LOG(INFO)<<"Rank "<<mpiRank(MPI_COMM_WORLD)<<"/"<<mpiSize(MPI_COMM_WORLD)<<": reading label batch "<<lines_id_<<", "<<filename;

  clock_t begin = clock();
  if(!this->pyModule)
  { 
    try{
      LOG(INFO)<<"Initializing python";
      Py_Initialize();
      PyRun_SimpleString("import sys");
      PyRun_SimpleString("sys.path.append(\"/afs/cs.stanford.edu/u/twangcat/scratch/caffenet/src/caffe/py_lane_label_reader/\")");
      np::initialize();
    }catch(py::error_already_set const &){
      PyErr_Print();
      LOG(FATAL) <<"boost python and numpy initialization failed.";
    }
  }
  //py::object arr_handle;
  // get a label blob from python helper class
  try{
    // initialize python helper class
    const bool predict_depth = this->layer_param_.multilane_label_param().predict_depth();
    if(!this->pyModule){
      this->pyModule = py::import("multilane_label_reader");
      this->pyClass = this->pyModule.attr("MultilaneLabelReader");
      this->pyReader = this->pyClass(mpiRank(MPI_COMM_WORLD),predict_depth, batch_size);
    }
    np::dtype dt = np::dtype::get_builtin<int>();
    py::tuple shape = py::make_tuple(batch_size) ;
    py::tuple stride = py::make_tuple(sizeof(int)) ;
    py::object own1;
    py::object own2;
    int* framePtr = const_cast<int *>(&frameIds[0]);
    int* transPtr = const_cast<int *>(&trans[0]);
    np::ndarray frameArr = np::from_data(framePtr,dt,shape,stride,own1);
    np::ndarray transArr = np::from_data(transPtr,dt,shape,stride,own2);
    // call function
    arr_handle = this->pyReader.attr("runLabelling")(filename, frameArr, transArr);
  }catch(py::error_already_set const &){
    PyErr_Print();
    LOG(FATAL) <<"numpy label reading function failed!";
  }
  np::ndarray pArray = np::from_object(arr_handle);

  int array_size = py::extract<int>(pArray.attr("size"));
  //LOG(INFO)<<"numpy array size = "<<array_size;
  float* pArrayPtr = (float*)(pArray.get_data());
  std::copy(pArrayPtr, pArrayPtr+array_size, top_data);
  // go to the next iter
  lines_id_+=mpiSize(MPI_COMM_WORLD);
  if (lines_id_ >= lines_size) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = mpiRank(MPI_COMM_WORLD);
    if (this->layer_param_.image_data_param().shuffle()) {
      ShuffleBatches();
    }
  }
  clock_t end = clock();
  //LOG(INFO)<<"label reading done! took "<<double(end - begin) / CLOCKS_PER_SEC;
}

INSTANTIATE_CLASS(MultilaneLabelLayer);

}  // namespace caffe
