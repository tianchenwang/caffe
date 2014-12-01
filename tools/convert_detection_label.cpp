// This program converts detection bounding box labels to Dataum proto buffers
// and save them in LMDB.
// Usage:
//   convert_detection_label [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using google::protobuf::Message;

DEFINE_bool(test_run, false, "If set to true, only generate 100 images.");
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb", "The backend for storing the result");
DEFINE_int32(width, 20, "Number of grids horizontally.");
DEFINE_int32(height, 15, "Number of grids vertically.");
DEFINE_int32(grid_dim, 4, "grid_dim x grid_dim number of pixels per each grid.");
DEFINE_int32(resize_width, 640, "Width images are resized to");
DEFINE_int32(resize_height, 480, "Height images are resized to");
DEFINE_double(scaling, 1.0 / 8.0, "Amount of downscaling applied up to the dense layers.");

bool ReadBoundingBoxLabelToDatum(
    const vector<int>& bbs, const int width, const int height,
    const int grid_dim, const float scaling, Datum* datum) {
  const int img_width = width * grid_dim;
  const int img_height = height * grid_dim;

  // 1 pixel label, 4 bounding box coordinates.
  const int num_mask_label = 1;
  const int num_bb_labels = 4;
  const int num_norm_labels = 2;
  const int num_total_labels = num_mask_label + num_bb_labels + num_norm_labels;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(img_height, img_width, CV_32F, cv::Scalar(0.0)));
  }

  CHECK_EQ(bbs.size() % num_bb_labels, 0);
  int total_num_pixels = 0;
  for (int i = 0; i < bbs.size(); i += 4) {
    float xmin = bbs[i];
    float ymin = bbs[i + 1];
    float xmax = bbs[i + 2];
    float ymax = bbs[i + 3];
    float w = xmax - xmin;
    float h = ymax - ymin;
    // shrink bboxes
    int gxmin = cvRound((xmin + w / 4) * scaling);
    int gxmax = cvRound((xmax - w / 4) * scaling);
    int gymin = cvRound((ymin + h / 4) * scaling);
    int gymax = cvRound((ymax - h / 4) * scaling);

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, img_width);
    CHECK_LE(gymax, img_height);
    if (gxmin >= img_width) {
      gxmin = img_width - 1;
    }
    if (gymin >= img_height) {
      gymin = img_height - 1;
    }
    CHECK_LT(gxmin, img_width);
    CHECK_LT(gymin, img_height);
    cv::Rect r(gxmin, gymin,
               gxmax - gxmin + (gxmax == gxmin && gxmax < img_width ? 1 : 0),
               gymax - gymin + (gymax == gymin && gymax < img_height ? 1 : 0));

    total_num_pixels += r.area();
    int normalization_height = ymax - ymin == 0 ? 1 : ymax - ymin;
    CHECK_GT(normalization_height, 0);
    float flabels[num_total_labels] =
        {1.0, xmin, ymin, xmax, ymax, 1.0 / normalization_height, 1.0};
    for (int j = 0; j < num_total_labels; ++j) {
      cv::Mat roi(*labels[j], r);
      roi = cv::Scalar(flabels[j]);
    }
  }

  if (total_num_pixels != 0) {
    float reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < img_height; ++y) {
      for (int x = 0; x < img_width; ++x) {
        if (labels[num_total_labels - 1]->at<float>(y, x) == 1.0) {
          labels[num_total_labels - 1]->at<float>(y, x) = reweight_value;
        }
      }
    }
  }

  datum->set_channels(num_total_labels * grid_dim * grid_dim);
  datum->set_height(height);
  datum->set_width(width);
  datum->set_label(0); // dummy label
  datum->clear_data();
  datum->clear_float_data();

  for (int m = 0; m < num_total_labels; ++m) {
    for (int dy = 0; dy < grid_dim; ++dy) {
      for (int dx = 0; dx < grid_dim; ++dx) {
        for (int y = 0; y < img_height; y += grid_dim) {
          for (int x = 0; x < img_width; x += grid_dim) {
            float adjustment = 0;
            float val = labels[m]->at<float>(y + dy, x + dx);
            if (m == 0 || m > 4) {
              // do nothing
            } else if (labels[0]->at<float>(y + dy, x + dx) == 0.0) {
              // do nothing
            } else if (m % 2 == 1) {
              // x coordinate
              adjustment = (x + grid_dim / 2) / scaling;
            } else {
              // y coordinate
              adjustment = (y + grid_dim / 2) / scaling;
            }
            datum->add_float_data(val - adjustment);
            //std::cout << val << "," << adjustment << "," << val - adjustment << " ";
            //std::cout << val - adjustment << " ";
          }
        }
      }
    }
  }

  CHECK_EQ(datum->float_data_size(), num_total_labels * img_height * img_width);
  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }

  return true;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_detection_labels [FLAGS] ROOTFOLDER/ LISTFILE IMG_DB_NAME LABEL_DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_detection_label");
    return 1;
  }

  bool is_color = !FLAGS_gray;
  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, vector<int> > > lines;
  string filename;
  int num_labels, tmp;
  while (infile >> filename >> num_labels) {
    vector<int> bbs;
    int num_numbers = num_labels * 4;
    for (int i = 0; i < num_numbers; i++) {
      infile >> tmp;
      bbs.push_back(tmp);
    }
    lines.push_back(std::make_pair(filename, bbs));
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  const string& db_backend = FLAGS_backend;
  const char* img_db_path = argv[3];
  const char* label_db_path = argv[4];

  bool generate_img = true;
  std::string img_db_str(img_db_path);
  if (img_db_str == "none") {
    generate_img = false;
  }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // lmdb
  MDB_env *mdb_env_img;
  MDB_dbi mdb_dbi_img;
  MDB_val mdb_key_img, mdb_data_img;
  MDB_txn *mdb_txn_img;

  MDB_env *mdb_env_label;
  MDB_dbi mdb_dbi_label;
  MDB_val mdb_key_label, mdb_data_label;
  MDB_txn *mdb_txn_label;

  // leveldb
  leveldb::DB* img_db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* img_batch = NULL;

  // Open db
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << img_db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, img_db_path, &img_db);
    CHECK(status.ok()) << "Failed to open leveldb " << img_db_path
        << ". Is it already existing?";
    img_batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    if (generate_img) {
      LOG(INFO) << "Opening lmdb " << img_db_path;
      CHECK_EQ(mkdir(img_db_path, 0744), 0)
          << "mkdir " << img_db_path << "failed";
      CHECK_EQ(mdb_env_create(&mdb_env_img), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_img, 1099511627776), MDB_SUCCESS)  // 1TB
          << "mdb_env_set_mapsize failed";
      CHECK_EQ(mdb_env_open(mdb_env_img, img_db_path, 0, 0664), MDB_SUCCESS)
          << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(mdb_env_img, NULL, 0, &mdb_txn_img), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(mdb_txn_img, NULL, 0, &mdb_dbi_img), MDB_SUCCESS)
          << "mdb_open failed. Does the lmdb already exist? ";
    }

    LOG(INFO) << "Opening lmdb " << label_db_path;
    CHECK_EQ(mkdir(label_db_path, 0744), 0)
        << "mkdir " << label_db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env_label), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_label, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env_label, label_db_path, 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_label, NULL, 0, &mdb_txn_label), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_label, NULL, 0, &mdb_dbi_label), MDB_SUCCESS)
        << "mdb_open failed. Does the lmdb already exist? ";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  // Storing to db
  string root_folder(argv[1]);
  Datum datum;
  Datum datum_label;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    if (!ReadImageToDatum(root_folder + lines[line_id].first, 0,
        resize_height, resize_width, is_color, &datum)) {
      continue;
    }
    CHECK(ReadBoundingBoxLabelToDatum(lines[line_id].second, FLAGS_width, FLAGS_height,
        FLAGS_grid_dim, FLAGS_scaling, &datum_label));
    if (!data_size_initialized) {
      data_size = datum.channels() * datum.height() * datum.width();
      data_size_initialized = true;
    } else {
      const string& data = datum.data();
      CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
          << data.size();
    }
    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());
    string value, label_value;
    datum.SerializeToString(&value);
    datum_label.SerializeToString(&label_value);
    string keystr(key_cstr);

    // Put in db
    if (db_backend == "leveldb") {  // leveldb
      img_batch->Put(keystr, value);
    } else if (db_backend == "lmdb") {  // lmdb
      if (generate_img) {
        mdb_data_img.mv_size = value.size();
        mdb_data_img.mv_data = reinterpret_cast<void*>(&value[0]);
        mdb_key_img.mv_size = keystr.size();
        mdb_key_img.mv_data = reinterpret_cast<void*>(&keystr[0]);
        CHECK_EQ(mdb_put(mdb_txn_img, mdb_dbi_img, &mdb_key_img, &mdb_data_img, 0), MDB_SUCCESS)
            << "mdb_put failed";
      }
      mdb_data_label.mv_size = label_value.size();
      mdb_data_label.mv_data = reinterpret_cast<void*>(&label_value[0]);
      mdb_key_label.mv_size = keystr.size();
      mdb_key_label.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn_label, mdb_dbi_label, &mdb_key_label, &mdb_data_label, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 1000 == 0) {
      // Commit txn
      if (db_backend == "leveldb") {
        img_db->Write(leveldb::WriteOptions(), img_batch);
        delete img_batch;
        img_batch = new leveldb::WriteBatch();
      } else if (db_backend == "lmdb") {  // lmdb
        if (generate_img) {
          CHECK_EQ(mdb_txn_commit(mdb_txn_img), MDB_SUCCESS)
              << "mdb_txn_commit failed";
          CHECK_EQ(mdb_txn_begin(mdb_env_img, NULL, 0, &mdb_txn_img), MDB_SUCCESS)
              << "mdb_txn_begin failed";
        }
        CHECK_EQ(mdb_txn_commit(mdb_txn_label), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env_label, NULL, 0, &mdb_txn_label), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }
      LOG(ERROR) << "Processed " << count << " files.";
    }

    if (FLAGS_test_run && count == 100) {
      break;
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    if (db_backend == "leveldb") {  // leveldb
      img_db->Write(leveldb::WriteOptions(), img_batch);
      delete img_batch;
      delete img_db;
    } else if (db_backend == "lmdb") {  // lmdb
      if (generate_img) {
        CHECK_EQ(mdb_txn_commit(mdb_txn_img), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env_img, mdb_dbi_img);
        mdb_env_close(mdb_env_img);
      }
      CHECK_EQ(mdb_txn_commit(mdb_txn_label), MDB_SUCCESS) << "mdb_txn_commit failed";
      mdb_close(mdb_env_label, mdb_dbi_label);
      mdb_env_close(mdb_env_label);
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
