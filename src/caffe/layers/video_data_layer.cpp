#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <boost/algorithm/string.hpp>
namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
bool VideoDataLayer<Dtype>:: ReadVideoFrameToDatum(const string& filename, size_t id, size_t persp,
    const int height, const int width, Datum* datum) {
  int cam_num = (int)(filename.at(filename.length()-5) - '0');
  cam_num = cam_num>2?2:cam_num; // 3rd cam is for testing only. So using cam2 distortions as dummy
  //int numPersp = mTransforms.size()/2;
  cv::Mat cv_img, cv_img_origin;
  bool set_ok = this->cap->set(CV_CAP_PROP_POS_FRAMES, id );
  if(!set_ok) {
    LOG(ERROR)<<"Failed to set video frame"; 
    return false;
  }
  bool read_ok = this->cap->read(cv_img_origin);
  if(!read_ok) {
    LOG(ERROR)<<"Failed to read video frame";
    return false;
  }
  // resize image if necessary
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  // apply perspective transform
  //cv::Mat warpMatrix = mTransforms[persp+(cam_num-1)*numPersp];
  //cv::warpPerspective(cv_img, cv_img, warpMatrix, frame.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  // copy data to datum  
  int num_channels = 3;
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(0); // dummy label for now.
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < num_channels; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  return true;
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
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
    std::vector<size_t> frame_ids;
    boost::split(frame_ids_str, batch_fields[1], boost::is_any_of(" "), 
                 boost::token_compress_on);
    for (int f=0; f<frame_ids_str.size(); ++f) {
      frame_ids.push_back((size_t)atoi(frame_ids_str[f].c_str()));
    }

    // store persp transform ids
    std::vector<string> trans_ids_str;
    std::vector<size_t> trans_ids;
    boost::split(trans_ids_str, batch_fields[2], boost::is_any_of(" "), 
                 boost::token_compress_on);
    for (int f=0; f<trans_ids_str.size(); ++f) {
      trans_ids.push_back((size_t)atoi(trans_ids_str[f].c_str()));
    }
       
    lines_.push_back(std::make_pair(filename, std::make_pair(frame_ids, trans_ids)));
  }

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling batches";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleBatches();
  }
  LOG(INFO) << "A total of " << lines_.size() << " batches.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data batch, and use it to initialize the top blob.
  this->cap = new cv::VideoCapture(lines_[lines_id_].first);
  Datum datum;
  CHECK(ReadVideoFrameToDatum(lines_[lines_id_].first, lines_[lines_id_].second.first[0],
                         lines_[lines_id_].second.second[0], new_height, new_width, &datum)); 
  this->cap->release();
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  //(*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleBatches() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}



// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  //Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();

  // datum scales
  const int lines_size = lines_.size();
  string filename = lines_[lines_id_].first;
  std::vector<size_t> frameIds = lines_[lines_id_].second.first;
  std::vector<size_t> trans = lines_[lines_id_].second.second;
  if (batch_size!=frameIds.size() || batch_size!=trans.size())
    LOG(ERROR)<<"Frame count mismatch!";
  LOG(INFO)<<"reading video file "<<filename;
  this->cap = new cv::VideoCapture(filename);
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    //CHECK_GT(lines_size, lines_id_);
    if (!ReadVideoFrameToDatum(filename, frameIds[item_id], trans[item_id],
          new_height, new_width, &datum)) {
      LOG(ERROR)<< "Error reading frame from video!";
      continue;
    }

    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleBatches();
      }
    }
  }
  this->cap->release();
}

INSTANTIATE_CLASS(VideoDataLayer);

}  // namespace caffe
