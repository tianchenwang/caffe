#ifndef CAFFE_UTIL_PLOT_H_
#define CAFFE_UTIL_PLOT_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
namespace caffe {
  cv::Scalar dist2color(double dist, double max_dist = 90.0);


// had to implement this in hpp file due to template.

  template <typename Dtype>
  void drawResults(cv::Mat& image, const Dtype* pix_label, const Dtype* reg_label, bool predict_depth, double scaling, int num_regression, int quad_height, int quad_width, int grid_dim){
      // draw ground truth and predictions on image.
      int grid_length = grid_dim*grid_dim;
      int label_height = quad_height*grid_dim;
      int label_width = quad_width*grid_dim;
      int img_width = image.cols;
      int img_height = image.rows;
      cv::cvtColor(image, image, CV_BGR2RGB);
      double thresh=0.4;
      // retrieve labels and predictions 
      for (int z=0; z<grid_length;++z){
        for (int qy = 0; qy < quad_height; ++qy) {
          for (int qx = 0; qx < quad_width; ++qx) {
            int dx = z%grid_dim;
            int dy = z/grid_dim;
            int x = qx*grid_dim+dx;
            int y = qy*grid_dim+dy;
            double label_prob = (double)(*(pix_label+((z*quad_height+qy)*quad_width+qx)));
            label_prob = 1. / (1. + exp(-label_prob));
            //std::cout<<label_prob<<" ";
            // draw pixel label/pred
            double x1 = x-0.5<0? 0:x-0.5;
            double y1 = y-0.5<0? 0:y-0.5;
            double w = scaling - (x<0.5? 0.5-x:0) - (x1+scaling>img_width? x1+scaling-img_width:0);
            double h = scaling - (y<0.5? 0.5-y:0) - (y1+scaling>img_height? y1+scaling-img_height:0);
            cv::Mat roi = image(cv::Rect(x1*scaling, y1*scaling, w, h));
            cv::Mat color(roi.size(), CV_32FC3, cv::Scalar(0, 255, 0)); 
            cv::addWeighted(color, label_prob, roi, 1.0 - label_prob , 0.0, roi); 
            if (label_prob > thresh) {

              // draw reg label/pred
              Dtype x_adj = (qx*grid_dim + grid_dim / 2) * scaling;
              Dtype y_adj = (qy*grid_dim + grid_dim / 2) * scaling;
              Dtype x_min = *(reg_label+((z*quad_height+qy)*quad_width+qx))+x_adj;
              Dtype y_min = *(reg_label+(((z+grid_length)*quad_height+qy)*quad_width+qx))+y_adj;
              Dtype x_max = *(reg_label+(((z+grid_length*2)*quad_height+qy)*quad_width+qx))+x_adj;
              Dtype y_max = *(reg_label+(((z+grid_length*3)*quad_height+qy)*quad_width+qx))+y_adj;
              cv::Point p1(x_min, y_min);
              cv::Point p2(x_max, y_max);
              cv::Scalar lineColor(100,100,200);
              if(predict_depth){
                Dtype min_depth = *(reg_label+(((z+grid_length*4)*quad_height+qy)*quad_width+qx));
                Dtype max_depth = *(reg_label+(((z+grid_length*5)*quad_height+qy)*quad_width+qx));
                lineColor = dist2color((min_depth+max_depth)/2.);
              }
              // draw label and predictions on image.
              cv::line(image,p1,p2,lineColor, 2);
            }
          }
        }
      }
  }



/*template <typename Dtype>
  void drawResults(cv::Mat& image, const Dtype* pix_label, double scaling, int quad_height, int quad_width, int grid_dim){
      // draw ground truth and predictions on image.
      int grid_length = grid_dim*grid_dim;
      int label_height = quad_height*grid_dim;
      int label_width = quad_width*grid_dim;
      int img_width = image.cols;
      int img_height = image.rows;
      double thresh=0.4;
      // retrieve labels and predictions 
      for (int qy = 0; qy < quad_height; ++qy) {
        for (int qx = 0; qx < quad_width; ++qx) {
          for (int z=0; z<grid_length;++z){
            int dx = z%grid_dim;
            int dy = z/grid_dim;
            int x = qx*grid_dim+dx;
            int y = qy*grid_dim+dy;
            double label_prob = (double)(*(pix_label+((z*quad_height+qy)*quad_width+qx)));
            label_prob = 1. / (1. + exp(-label_prob));
            double x1 = x-0.5<0? 0:x-0.5;
            double y1 = y-0.5<0? 0:y-0.5;
            double w = scaling - (x<0.5? 0.5-x:0) - (x1+scaling>img_width? x1+scaling-img_width:0);
            double h = scaling - (y<0.5? 0.5-y:0) - (y1+scaling>img_height? y1+scaling-img_height:0);
            cv::Mat roi = image(cv::Rect(x1*scaling, y1*scaling, w, h));
            //LOG(INFO)<<x1*scaling<<" "<<y1*scaling<<" "<<w<<" "<<h<<" "<<label_prob;
            cv::Mat color(roi.size(), CV_32FC3, cv::Scalar(0, 255, 0)); 
            cv::addWeighted(color, label_prob, roi, 1.0 - label_prob , 0.0, roi); 
          }
        }
      }
  }*/
}  // namespace caffe

#endif   // CAFFE_UTIL_PLOT_H_
