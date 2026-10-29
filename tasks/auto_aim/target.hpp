#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  static constexpr int OUTPOST_ARMOR_NUM = 3;
  static constexpr int OUTPOST_HEIGHT_CONFIRM_FRAMES = 5;
  static constexpr int OUTPOST_REMAP_CONFIRM_FRAMES = 7;
  static constexpr double OUTPOST_HEIGHT_RESIDUAL_TOLERANCE = 0.025;
  static constexpr double OUTPOST_PENDING_CONSISTENCY_TOLERANCE = 0.018;
  static constexpr double OUTPOST_HEIGHT_MIN_SEPARATION = 0.035;
  static constexpr double OUTPOST_MAX_HEIGHT_OFFSET = 0.65;
  static constexpr std::chrono::milliseconds OUTPOST_OBSERVATION_MAX_AGE{120};
  static constexpr std::chrono::milliseconds OUTPOST_HEIGHT_SAMPLE_MAX_GAP{120};

  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig,
    double outpost_min_height_separation = OUTPOST_HEIGHT_MIN_SEPARATION);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

  bool outpost_height_ready() const;
  bool armor_height_known(int id) const;
  int outpost_height_level(int id) const;
  double outpost_height_offset(int id) const;
  int expected_next_outpost_id() const;
  bool outpost_observation_fresh(std::chrono::steady_clock::time_point now) const;
  std::optional<Eigen::Vector4d> outpost_observed_xyza() const;

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;
  bool is_outpost_ = false;
  bool outpost_height_ready_ = false;
  bool outpost_height_separation_valid_ = true;
  double outpost_min_height_separation_ = OUTPOST_HEIGHT_MIN_SEPARATION;

  std::optional<Eigen::Vector4d> outpost_observed_xyza_;
  std::chrono::steady_clock::time_point outpost_observed_t_{};
  double outpost_observed_confidence_ = -1.0;
  bool has_outpost_observation_ = false;

  std::array<double, OUTPOST_ARMOR_NUM> outpost_height_offsets_{{0.0, 0.0, 0.0}};
  std::array<bool, OUTPOST_ARMOR_NUM> outpost_height_known_{{false, false, false}};
  std::array<int, OUTPOST_ARMOR_NUM> outpost_height_levels_{{-1, -1, -1}};
  std::array<std::array<double, OUTPOST_REMAP_CONFIRM_FRAMES>, OUTPOST_ARMOR_NUM>
    outpost_pending_samples_{};
  std::array<int, OUTPOST_ARMOR_NUM> outpost_pending_count_{{0, 0, 0}};
  std::array<double, OUTPOST_ARMOR_NUM> outpost_frame_height_sample_{};
  std::array<double, OUTPOST_ARMOR_NUM> outpost_frame_sample_confidence_{};
  std::array<std::chrono::steady_clock::time_point, OUTPOST_ARMOR_NUM> outpost_frame_sample_t_{};
  std::array<bool, OUTPOST_ARMOR_NUM> outpost_has_frame_sample_{};

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  void update_outpost_height(const Armor & armor, int id);
  void accept_outpost_height_sample(int id, double delta);
  void normalize_outpost_height_levels();
  double outpost_pending_median(int id) const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
