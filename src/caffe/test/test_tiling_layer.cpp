#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TilingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TilingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 8, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TilingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TilingLayerTest, TestDtypesAndDevices);

TYPED_TEST(TilingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TilingParameter* tiling_param = layer_param.mutable_tiling_param();
  tiling_param->set_tile_dim(2);
  TilingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 8);
}

TYPED_TEST(TilingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TilingParameter* tiling_param = layer_param.mutable_tiling_param();
  tiling_param->set_tile_dim(2);
  TilingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 8; ++c) {
      for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
          EXPECT_EQ(
              this->blob_bottom_->data_at(n, c, y, x),
              this->blob_top_->data_at(n, c / 4, y * 2 + (c % 4) / 2, x * 2 + c % 2));
        }
      }
    }
  }
}

TYPED_TEST(TilingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TilingParameter* tiling_param = layer_param.mutable_tiling_param();
  tiling_param->set_tile_dim(2);
  TilingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(
      &layer, &(this->blob_bottom_vec_), &(this->blob_top_vec_));
}

}  // namespace caffe
