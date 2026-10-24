#ifndef AUTO_AIM__OUTPOST_REFINER_HPP
#define AUTO_AIM__OUTPOST_REFINER_HPP

#include <opencv2/opencv.hpp>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"

namespace auto_aim
{
class OutpostRefiner
{
public:
  OutpostRefiner(const std::string & config_path, Solver & solver);

  bool refine(
    Armor & armor, const Target & target, int current_id, const cv::Mat & bgr_img) const;

private:
  Solver & solver_;
  double threshold_;
  double max_angle_error_;
  double min_lightbar_ratio_;
  double max_lightbar_ratio_;
  double min_lightbar_length_;

  bool find_lightbar(
    const cv::Mat & bgr_img, const cv::Point2f & predicted_a, const cv::Point2f & predicted_b,
    Color armor_color, const cv::Point2f & own_left, const cv::Point2f & own_right,
    double own_guard, Lightbar & lightbar, double & score) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_REFINER_HPP
