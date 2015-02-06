#include <leveldb/db.h>
#include <stdint.h>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"



// Top layer types. Should be consistent with the prototxt file.
enum TopLayerDataType {
  IMAGE
};

enum TopLayerLabelType {
  CAR_MERGED_LABELS
};

/*
enum TopLayerType {
  IMAGE,
  CAR_PIXEL_LABEL,
  CAR_BB_LABEL,
  CAR_BB_DIM_NORMALIZATION_LABEL,
  CAR_BB_NUM_PIXEL_NORMALIZATION_LABEL
};
*/

const int kNumData = 1;
const int kNumLabels = 1;
const int kNumBBRegressionCoords = 4;
const int kNumRegressionMasks = 8;

namespace caffe {

template <typename Dtype>
DrivingDataLayer<Dtype>::~DrivingDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    break;  // do nothing
  case DataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  this->data_transformer_.InitRand();
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  for (int i = 0; i < prefetch_datas_.size(); ++i) {
    caffe_copy(prefetch_datas_[i]->count(), prefetch_datas_[i]->cpu_data(),
               (*top)[i]->mutable_cpu_data());
  }
  // Copy the labels if applicable.
  if (this->output_labels_) {
    for (int i = 0; i < prefetch_labels_.size(); ++i) {
      const int offset_idx = prefetch_datas_.size() + i;
      caffe_copy(prefetch_labels_[i]->count(), prefetch_labels_[i]->cpu_data(),
                 (*top)[offset_idx]->mutable_cpu_data());
    }
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  for (int i = 0 ; i < this->prefetch_datas_.size(); ++i) {
    this->prefetch_datas_[i]->mutable_cpu_data();
  }
  if (this->output_labels_) {
    for (int i = 0 ; i < this->prefetch_labels_.size(); ++i) {
      this->prefetch_labels_[i]->mutable_cpu_data();
    }
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DrivingDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // Initialize DB
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case DataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
    // Set cursor at begining
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
             MDB_SUCCESS);
    // Jump forward to our MPI rank
    for (int i = 0; i < Caffe::mpi()->rank(); ++i) {
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT),
               MDB_SUCCESS);
    }
    // read our first data point
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                            &mdb_value_, MDB_GET_CURRENT),
             MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    Caffe::mpi()->Bcast(1, &skip);
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
        iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
      case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
      default:
        LOG(FATAL) << "Unknown database backend";
      }
    }
  }

  // Read a data point, and use it to initialize the top blob.
  DrivingData data;
  switch (this->layer_param_.data_param().backend()) {
  case DataParameter_DB_LEVELDB:
    data.ParseFromString(iter_->value().ToString());
    break;
  case DataParameter_DB_LMDB:
    data.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }

  // For now we still assume there's just one image input datum.
  // TODO(willsong): Remove all single datum dependencies.
  const Datum& datum = data.car_image_datum();
  this->prefetch_datas_.resize(kNumData);
  for (int i = 0 ; i < this->prefetch_datas_.size(); ++i) {
    this->prefetch_datas_[i] = new Blob<Dtype>();
  }
  if (this->output_labels_) {
    this->prefetch_labels_.resize(kNumLabels);
    for (int i = 0 ; i < this->prefetch_labels_.size(); ++i) {
      this->prefetch_labels_[i] = new Blob<Dtype>();
    }
  }

  // image data
  (*top)[IMAGE]->Reshape(
      this->layer_param_.data_param().batch_size(), datum.channels(),
      data.car_cropped_height(), data.car_cropped_width());
  this->prefetch_datas_[IMAGE]->Reshape(
      this->layer_param_.data_param().batch_size(),
      datum.channels(), data.car_cropped_height(), data.car_cropped_width());
  LOG(INFO) << "output image data size: " << (*top)[IMAGE]->num() << ","
      << (*top)[IMAGE]->channels() << "," << (*top)[IMAGE]->height() << ","
      << (*top)[IMAGE]->width();

  // labels
  if (this->output_labels_) {
    // Initializing car pixel label mask
    /* Without tiling layer.
    (*top)[kNumData + CAR_MERGED_LABELS]->Reshape(
        this->layer_param_.data_param().batch_size(),
        data.car_label_resolution() * data.car_label_resolution() * kNumRegressionMasks,
        data.car_label_height(),
        data.car_label_width());
    this->prefetch_labels_[CAR_MERGED_LABELS]->Reshape(
        this->layer_param_.data_param().batch_size(),
        data.car_label_resolution() * data.car_label_resolution() * kNumRegressionMasks,
        data.car_label_height(),
        data.car_label_width());
    */
    (*top)[kNumData + CAR_MERGED_LABELS]->Reshape(
        this->layer_param_.data_param().batch_size(),
        kNumRegressionMasks,
        data.car_label_height() * data.car_label_resolution(),
        data.car_label_width() * data.car_label_resolution());
    this->prefetch_labels_[CAR_MERGED_LABELS]->Reshape(
        this->layer_param_.data_param().batch_size(),
        kNumRegressionMasks,
        data.car_label_height() * data.car_label_resolution(),
        data.car_label_width() * data.car_label_resolution());
  }

  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = data.car_cropped_height();
  this->datum_width_ = data.car_cropped_width();
  this->datum_size_ = datum.channels() * data.car_cropped_height() \
    * data.car_cropped_width();

  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
unsigned int DrivingDataLayer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

template <typename Dtype>
bool DrivingDataLayer<Dtype>::ReadBoundingBoxLabelToDatum(
    const DrivingData& data, Datum* datum, const int h_off, const int w_off) {
  const int grid_dim = data.car_label_resolution();
  const int width = data.car_label_width();
  const int height = data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const float half_shrink_factor = data.car_shrink_factor() / 2;
  const float scaling = static_cast<float>(full_label_width) \
    / data.car_cropped_width();

  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = kNumRegressionMasks;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }

  for (int i = 0; i < data.car_boxes_size(); ++i) {
    int xmin = data.car_boxes(i).xmin();
    int ymin = data.car_boxes(i).ymin();
    int xmax = data.car_boxes(i).xmax();
    int ymax = data.car_boxes(i).ymax();
    xmin = std::min(std::max(0, xmin - w_off), data.car_cropped_width());
    xmax = std::min(std::max(0, xmax - w_off), data.car_cropped_width());
    ymin = std::min(std::max(0, ymin - h_off), data.car_cropped_height());
    ymax = std::min(std::max(0, ymax - h_off), data.car_cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }
    // shrink bboxes
    int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
    int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);
    int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
    int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    if (gxmin >= full_label_width) {
      gxmin = full_label_width - 1;
    }
    if (gymin >= full_label_height) {
      gymin = full_label_height - 1;
    }
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, full_label_width);
    CHECK_LE(gymax, full_label_height);
    if (gxmin == gxmax) {
      if (gxmax < full_label_width - 1) {
        gxmax++;
      } else if (gxmin > 0) {
        gxmin--;
      }
    }
    if (gymin == gymax) {
      if (gymax < full_label_height - 1) {
        gymax++;
      } else if (gymin > 0) {
        gymin--;
      }
    }
    CHECK_LT(gxmin, gxmax);
    CHECK_LT(gymin, gymax);
    if (gxmax == full_label_width) {
      gxmax--;
    }
    if (gymax == full_label_height) {
      gymax--;
    }
    cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);

    float flabels[num_total_labels] =
        {1.0, xmin, ymin, xmax, ymax, 1.0 / w, 1.0 / h, 1.0};
    for (int j = 0; j < num_total_labels; ++j) {
      cv::Mat roi(*labels[j], r);
      roi = cv::Scalar(flabels[j]);
    }
  }

  int total_num_pixels = 0;
  for (int y = 0; y < full_label_height; ++y) {
    for (int x = 0; x < full_label_width; ++x) {
      if (labels[num_total_labels - 1]->at<float>(y, x) == 1.0) {
        total_num_pixels++;
      }
    }
  }
  if (total_num_pixels != 0) {
    float reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        if (labels[num_total_labels - 1]->at<float>(y, x) == 1.0) {
          labels[num_total_labels - 1]->at<float>(y, x) = reweight_value;
        }
      }
    }
  }

  datum->set_channels(num_total_labels);
  datum->set_height(full_label_height);
  datum->set_width(full_label_width);
  datum->set_label(0);  // dummy label
  datum->clear_data();
  datum->clear_float_data();

  for (int m = 0; m < num_total_labels; ++m) {
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        float adjustment = 0;
        float val = labels[m]->at<float>(y, x);
        if (m == 0 || m > 4) {
          // do nothing
        } else if (labels[0]->at<float>(y, x) == 0.0) {
          // do nothing
        } else if (m % 2 == 1) {
          // x coordinate
          adjustment = x / scaling;
        } else {
          // y coordinate
          adjustment = y / scaling;
        }
        datum->add_float_data(val - adjustment);
      }
    }
  }

  CHECK_EQ(datum->float_data_size(),
           num_total_labels * full_label_height * full_label_width);
  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }

  return true;
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DrivingDataLayer<Dtype>::InternalThreadEntry() {
  DrivingData data;
  CHECK(this->prefetch_datas_[IMAGE]->count());
  Dtype* top_data = this->prefetch_datas_[IMAGE]->mutable_cpu_data();
  vector<Dtype*> top_labels;
  for (int i = 0; i < this->prefetch_labels_.size(); ++i) {
    top_labels.push_back(this->prefetch_labels_[i]->mutable_cpu_data());
  }
  const int batch_size = this->layer_param_.data_param().batch_size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      data.ParseFromString(iter_->value().ToString());
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      data.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    // Apply data transformations
    // this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
    const Datum& img_datum = data.car_image_datum();
    const string& img_datum_data = img_datum.data();
    int h_off = img_datum.height() == data.car_cropped_height() ?
        0 : Rand() % (img_datum.height() - data.car_cropped_height());
    int w_off = img_datum.width() == data.car_cropped_width() ?
        0 : Rand() % (img_datum.width() - data.car_cropped_width());
    for (int c = 0; c < img_datum.channels(); ++c) {
      for (int h = 0; h < data.car_cropped_height(); ++h) {
        for (int w = 0; w < data.car_cropped_width(); ++w) {
          int top_index = ((item_id * img_datum.channels() + c) \
                           * data.car_cropped_height() + h)
              * data.car_cropped_width() + w;
          int data_index = (c * img_datum.height() + h + h_off) \
            * img_datum.width() + w + w_off;
          uint8_t datum_element_ui8 = \
            static_cast<uint8_t>(img_datum_data[data_index]);
          Dtype datum_element = static_cast<Dtype>(datum_element_ui8);

          top_data[top_index] = datum_element - this->mean_[data_index];
        }
      }
    }

    vector<Datum> label_datums(kNumLabels);
    if (this->output_labels_) {
      // Call appropriate functions for genearting each label
      ReadBoundingBoxLabelToDatum(data, &label_datums[CAR_MERGED_LABELS],
                                  h_off, w_off);
    }
    for (int i = 0; i < kNumLabels; ++i) {
      for (int c = 0; c < label_datums[i].channels(); ++c) {
        for (int h = 0; h < label_datums[i].height(); ++h) {
          for (int w = 0; w < label_datums[i].width(); ++w) {
            const int top_index = ((item_id * label_datums[i].channels() + c)
                * label_datums[i].height() + h) * label_datums[i].width() + w;
            const int data_index = (c * label_datums[i].height() + h) * \
              label_datums[i].width() + w;
            float label_datum_elem = label_datums[i].float_data(data_index);
            top_labels[i][top_index] = static_cast<Dtype>(label_datum_elem);
          }
        }
      }
    }

    // go to the next iter
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      for (int i = 0; i < Caffe::mpi()->size(); ++i) {
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
                           &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                                  &mdb_value_, MDB_FIRST), MDB_SUCCESS);
        }
      }

      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(DrivingDataLayer);

}  // namespace caffe
