#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/plot.hpp"

namespace caffe {

cv::Scalar dist2color(double dist, double max_dist){
  //given a distance and a maximum distance, gives a color code for the distance.
  //red being closest, green is mid-range, blue being furthest
  double alpha = (dist/max_dist);
  cv::Scalar color(0,0,0);
  if(alpha<0.5)
  {
    color[2] = 255*(1-alpha*2);
    color[1] = 255*alpha*2;
  }
  else
  {  
    double beta = alpha-0.5;
    color[1] = 255*(1-beta*2);
    color[0] = 255*beta*2;
  }
  return color;
}


}  // namespace caffe
