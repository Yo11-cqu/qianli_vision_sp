#include "outpost_refiner.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
//最小的精修代码，确保只有在远距离再使用
constexpr double MIN_REFINE_DISTANCE = 4.0;
//所允许的最大重投影误差，单位px
constexpr double MAX_MEAN_REPROJECTION_ERROR = 10.0;
constexpr double MAX_MATCH_ANGLE_ERROR = 35.0 * CV_PI / 180.0;
constexpr double MAX_ENDPOINT_ERROR = 12.0;
}  // namespace

OutpostRefiner::OutpostRefiner(const std::string & config_path, Solver & solver) : solver_(solver)
{
  auto yaml = YAML::LoadFile(config_path);
  threshold_ = yaml["threshold"].as<double>();
  max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3;
  min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
  max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
  min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
}

bool OutpostRefiner::refine(
  Armor & armor, const Target & target, int current_id, const cv::Mat & bgr_img) const
{
  if (
    armor.name != ArmorName::outpost || armor.type != ArmorType::small || bgr_img.empty())
    return false;
  if (target.armor_num() != 3) return false;

  Eigen::VectorXd x = target.ekf_x();
  if (std::sqrt(x[0] * x[0] + x[2] * x[2]) < MIN_REFINE_DISTANCE) return false;

  Eigen::Vector4d current = target.armor_xyza(current_id);
  Eigen::Vector3d current_xyz = current.head<3>();
  Eigen::Matrix3d current_R = solver_.R_armor2world(current[3], ArmorName::outpost);

  const auto & armor_points = solver_.object_points(ArmorType::small);
  if (armor_points.size() < 4 || armor.points.size() != 4) return false;
  const double half_width = armor_points[0].y;
  const double half_length = armor_points[0].z;

  double own_width = std::max(
    cv::norm(armor.points[0] - armor.points[1]),
    cv::norm(armor.points[3] - armor.points[2]));
  cv::Point2f own_left = (armor.points[0] + armor.points[3]) / 2;
  cv::Point2f own_right = (armor.points[1] + armor.points[2]) / 2;
  double own_guard = std::max(4.0, own_width * 0.25);

  bool found = false;
  double best_score = std::numeric_limits<double>::max();
  cv::Point2f best_predicted_top;
  cv::Point2f best_predicted_bottom;
  Eigen::Vector3d best_object_top;
  Eigen::Vector3d best_object_bottom;
  Lightbar best_lightbar;

  for (const auto & adjacent : target.adjacent_armor_xyza_list(current_id)) {
    Eigen::Vector3d adjacent_xyz = adjacent.head<3>();
    Eigen::Matrix3d adjacent_R = solver_.R_armor2world(adjacent[3], ArmorName::outpost);

    Eigen::Vector3d lightbar_center_a =
      adjacent_xyz + adjacent_R * Eigen::Vector3d(0, half_width, 0);
    Eigen::Vector3d lightbar_center_b =
      adjacent_xyz + adjacent_R * Eigen::Vector3d(0, -half_width, 0);
    double side = (lightbar_center_a - current_xyz).squaredNorm() <
                       (lightbar_center_b - current_xyz).squaredNorm()
                    ? 1.0
                    : -1.0;

    Eigen::Vector3d top_in_world =
      adjacent_xyz + adjacent_R * Eigen::Vector3d(0, side * half_width, half_length);
    Eigen::Vector3d bottom_in_world =
      adjacent_xyz + adjacent_R * Eigen::Vector3d(0, side * half_width, -half_length);
    Eigen::Vector3d top_in_armor = current_R.transpose() * (top_in_world - current_xyz);
    Eigen::Vector3d bottom_in_armor = current_R.transpose() * (bottom_in_world - current_xyz);

    std::vector<cv::Point3f> world_points{
      {static_cast<float>(top_in_world.x()), static_cast<float>(top_in_world.y()),
       static_cast<float>(top_in_world.z())},
      {static_cast<float>(bottom_in_world.x()), static_cast<float>(bottom_in_world.y()),
       static_cast<float>(bottom_in_world.z())}};
    auto predicted = solver_.world2pixel(world_points);
    if (predicted.size() != 2) continue;

    const double margin = 40.0;
    if (
      predicted[0].x < -margin || predicted[0].x > bgr_img.cols + margin ||
      predicted[0].y < -margin || predicted[0].y > bgr_img.rows + margin ||
      predicted[1].x < -margin || predicted[1].x > bgr_img.cols + margin ||
      predicted[1].y < -margin || predicted[1].y > bgr_img.rows + margin)
      continue;

    Lightbar lightbar;
    double score = 0.0;
    if (!find_lightbar(
          bgr_img, predicted[0], predicted[1], armor.color, own_left, own_right, own_guard,
          lightbar, score))
      continue;

    if (score < best_score) {
      found = true;
      best_score = score;
      best_predicted_top = predicted[0];
      best_predicted_bottom = predicted[1];
      best_object_top = top_in_armor;
      best_object_bottom = bottom_in_armor;
      best_lightbar = lightbar;
    }
  }

  if (!found) return false;

  cv::Point2f actual_top = best_lightbar.top;
  cv::Point2f actual_bottom = best_lightbar.bottom;
  Eigen::Vector3d object_top = best_object_top;
  Eigen::Vector3d object_bottom = best_object_bottom;

  double direct_error =
    cv::norm(actual_top - best_predicted_top) + cv::norm(actual_bottom - best_predicted_bottom);
  double swapped_error =
    cv::norm(actual_top - best_predicted_bottom) + cv::norm(actual_bottom - best_predicted_top);
  if (swapped_error < direct_error) {
    std::swap(actual_top, actual_bottom);
    std::swap(object_top, object_bottom);
  }

  std::vector<cv::Point3f> object_points = armor_points;
  object_points.emplace_back(
    static_cast<float>(object_top.x()), static_cast<float>(object_top.y()),
    static_cast<float>(object_top.z()));
  object_points.emplace_back(
    static_cast<float>(object_bottom.x()), static_cast<float>(object_bottom.y()),
    static_cast<float>(object_bottom.z()));

  std::vector<cv::Point2f> image_points = armor.points;
  image_points.push_back(actual_top);
  image_points.push_back(actual_bottom);

  Armor backup = armor;
  double mean_error = std::numeric_limits<double>::max();
  if (!solver_.solve(armor, object_points, image_points, &mean_error)) {
    armor = backup;
    return false;
  }

  if (mean_error > MAX_MEAN_REPROJECTION_ERROR) {
    tools::logger()->debug(
      "[OutpostRefiner] reject 6-point solve, mean error {:.2f}px", mean_error);
    armor = backup;
    return false;
  }

  tools::logger()->debug(
    "[OutpostRefiner] 6-point solve accepted, mean error {:.2f}px, score {:.2f}", mean_error,
    best_score);
  return true;
}

bool OutpostRefiner::find_lightbar(
  const cv::Mat & bgr_img, const cv::Point2f & predicted_a, const cv::Point2f & predicted_b,
  Color armor_color, const cv::Point2f & own_left, const cv::Point2f & own_right,
  double own_guard, Lightbar & lightbar, double & score) const
{
  //判断预测的灯条会不会太短
  double predicted_length = cv::norm(predicted_b - predicted_a);
  if (predicted_length < 3.0) return false;

  //圈出ROI
  cv::Rect roi_rect = cv::boundingRect(std::vector<cv::Point2f>{predicted_a, predicted_b});
  //为了避免框太小（或者一条直线）将框进行一定的放大
  int margin = std::max(12, static_cast<int>(predicted_length * 1.0));
  roi_rect.x -= margin;
  roi_rect.y -= margin;
  roi_rect.width += 2 * margin;
  roi_rect.height += 2 * margin;
  roi_rect &= cv::Rect(0, 0, bgr_img.cols, bgr_img.rows);
  if (roi_rect.width < 5 || roi_rect.height < 5) return false;

  //二值化、提出轮廓
  cv::Mat roi = bgr_img(roi_rect);
  cv::Mat gray_img;
  cv::cvtColor(roi, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  cv::Point2f predicted_center = (predicted_a + predicted_b) / 2;
  double predicted_angle =
    std::atan2(predicted_b.y - predicted_a.y, predicted_b.x - predicted_a.x);

  const double max_endpoint_error = std::max(MAX_ENDPOINT_ERROR, predicted_length * 1.5);

  bool found = false;
  score = std::numeric_limits<double>::max();
  std::size_t lightbar_id = 0;

  //遍历，构造lightbar候选
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    Lightbar candidate(rotated_rect, lightbar_id++);

    //自适应过滤长度（直接用yaml的容易因为距离变化而产生问题，于是就加上了预测的长度）
    double min_length = std::max(2.0, std::min(min_lightbar_length_ * 0.5, predicted_length * 0.6));
    if (candidate.length < min_length) continue;
    if (
      //考虑到距离等原因产生的误差，为避免误杀而放大了范围
      candidate.ratio < min_lightbar_ratio_ * 0.6 ||
      candidate.ratio > max_lightbar_ratio_ * 1.5)
      continue;
    if (candidate.angle_error > max_angle_error_) continue;

    int red_sum = 0;
    int blue_sum = 0;
    for (const auto & point : contour) {
      const auto & bgr = roi.at<cv::Vec3b>(point);
      red_sum += bgr[2];
      blue_sum += bgr[0];
    }
    Color candidate_color = blue_sum > red_sum ? Color::blue : Color::red;

    double angle_error = std::abs(tools::limit_rad(candidate.angle - predicted_angle));
    angle_error = std::min(angle_error, std::abs(CV_PI - angle_error));
    if (angle_error > MAX_MATCH_ANGLE_ERROR) continue;

    //防止找到当前装甲板自己的灯条
    cv::Point2f offset(roi_rect.x, roi_rect.y);
    //回到原图
    cv::Point2f candidate_center = candidate.center + offset;
    if (
      cv::norm(candidate_center - own_left) < own_guard ||
      cv::norm(candidate_center - own_right) < own_guard)
      continue;

    cv::Point2f candidate_top = candidate.top + offset;
    cv::Point2f candidate_bottom = candidate.bottom + offset;

    double direct_error =
      cv::norm(candidate_top - predicted_a) + cv::norm(candidate_bottom - predicted_b);
    double swapped_error =
      cv::norm(candidate_top - predicted_b) + cv::norm(candidate_bottom - predicted_a);
    double endpoint_error = std::min(direct_error, swapped_error) / 2;
    if (endpoint_error > max_endpoint_error) continue;

    double center_error = cv::norm(candidate_center - predicted_center);
    if (center_error > std::max(10.0, predicted_length * 1.2)) continue;

    double length_error = std::abs(std::log(candidate.length / predicted_length));
    double color_penalty = candidate_color == armor_color ? 0.0 : 1.0;
    double candidate_score =
      center_error / std::max(5.0, predicted_length) + 2.0 * angle_error + length_error +
      color_penalty + endpoint_error / std::max(5.0, predicted_length);

    if (candidate_score < score) {
      found = true;
      score = candidate_score;

      candidate.center += offset;
      candidate.top += offset;
      candidate.bottom += offset;
      candidate.top2bottom = candidate.bottom - candidate.top;
      for (auto & point : candidate.points) point += offset;
      lightbar = candidate;
    }
  }

  return found;
}

}  // namespace auto_aim
