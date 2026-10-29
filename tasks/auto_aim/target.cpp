#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, double outpost_min_height_separation)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(false),
  is_outpost_(armor.name == ArmorName::outpost),
  outpost_min_height_separation_(
    std::isfinite(outpost_min_height_separation) && outpost_min_height_separation >= 0.015 &&
        outpost_min_height_separation <= 0.15
      ? outpost_min_height_separation
      : OUTPOST_HEIGHT_MIN_SEPARATION),
  t_(t)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）

  if (is_outpost_) {
    outpost_observed_xyza_ = Eigen::Vector4d{xyz[0], xyz[1], xyz[2], ypr[0]};
    outpost_observed_t_ = t;
    outpost_observed_confidence_ = std::isfinite(armor.confidence) ? armor.confidence : 0.0;
    has_outpost_observation_ = true;
    outpost_height_known_[0] = true;
    outpost_height_offsets_[0] = 0.0;
  }
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;   // 前哨站加速度方差
    v2 = 0.1;  // 前哨站角加速度方差
  } else {
    v1 = 100;  // 加速度方差
    v2 = 400;  // 角加速度方差
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

void Target::update(const Armor & armor)
{
  if (
    is_outpost_ &&
    (!armor.xyz_in_world.allFinite() || !armor.ypr_in_world.allFinite() ||
     !armor.ypd_in_world.allFinite()))
    return;

  // 装甲板匹配
  int id;
  if (is_outpost_) id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  update_outpost_height(armor, id);

  if (is_outpost_) {
    const double expected_z = h_armor_xyz(ekf_.x, id)[2];
    const double z_residual = armor.xyz_in_world[2] - expected_z;
    const double z_gate = OUTPOST_HEIGHT_RESIDUAL_TOLERANCE;
    const double observed_offset = armor.xyz_in_world[2] - ekf_.x[4];
    const bool finite_height = std::isfinite(observed_offset);
    const bool height_plausible = outpost_height_known_[id]
                                    ? std::abs(z_residual) <= z_gate
                                    : finite_height &&
                                        std::abs(observed_offset) <= OUTPOST_MAX_HEIGHT_OFFSET;

    const double confidence = std::isfinite(armor.confidence) ? armor.confidence : 0.0;
    if (
      height_plausible &&
      (!has_outpost_observation_ || t_ > outpost_observed_t_ ||
       (t_ == outpost_observed_t_ && confidence > outpost_observed_confidence_))) {
      outpost_observed_xyza_ = Eigen::Vector4d{
        armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2],
        armor.ypr_in_world[0]};
      outpost_observed_t_ = t_;
      outpost_observed_confidence_ = confidence;
      has_outpost_observation_ = true;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  if (is_outpost_) {
    Armor filtered_armor = armor;
    const double expected_z = h_armor_xyz(ekf_.x, id)[2];
    const double z_residual = armor.xyz_in_world[2] - expected_z;
    const double z_gate = OUTPOST_HEIGHT_RESIDUAL_TOLERANCE;
    if (
      !outpost_height_known_[id] || !std::isfinite(z_residual) ||
      std::abs(z_residual) > z_gate)
      filtered_armor.xyz_in_world[2] = expected_z;
    filtered_armor.ypd_in_world = tools::xyz2ypd(filtered_armor.xyz_in_world);
    update_ypda(filtered_armor, id);
  } else {
    update_ypda(armor, id);
  }
}

void Target::update_ypda(const Armor & armor, int id)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

bool Target::outpost_height_ready() const
{
  if (!is_outpost_ || !outpost_height_ready_) return false;
  for (int i = 0; i < OUTPOST_ARMOR_NUM; ++i) {
    if (!armor_height_known(i)) return false;
  }
  return true;
}

bool Target::armor_height_known(int id) const
{
  if (!is_outpost_ || id < 0 || id >= OUTPOST_ARMOR_NUM) return true;
  if (
    !outpost_height_separation_valid_ || !outpost_height_known_[id] ||
    outpost_pending_count_[id] > 0)
    return false;
  if (
    outpost_has_frame_sample_[id] &&
    std::abs(outpost_frame_height_sample_[id] - outpost_height_offsets_[id]) >
      OUTPOST_PENDING_CONSISTENCY_TOLERANCE)
    return false;
  return true;
}

int Target::outpost_height_level(int id) const
{
  if (!is_outpost_ || id < 0 || id >= OUTPOST_ARMOR_NUM || !armor_height_known(id)) return -1;
  return outpost_height_levels_[id];
}

double Target::outpost_height_offset(int id) const
{
  if (!is_outpost_ || id < 0 || id >= OUTPOST_ARMOR_NUM) return 0.0;
  return outpost_height_offsets_[id];
}

int Target::expected_next_outpost_id() const
{
  if (!is_outpost_ || last_id < 0 || last_id >= OUTPOST_ARMOR_NUM) return -1;
  if (std::abs(ekf_.x[7]) < 0.3) return -1;
  return ekf_.x[7] > 0 ? (last_id + 2) % OUTPOST_ARMOR_NUM : (last_id + 1) % OUTPOST_ARMOR_NUM;
}

bool Target::outpost_observation_fresh(std::chrono::steady_clock::time_point now) const
{
  if (!is_outpost_ || !has_outpost_observation_ || now < outpost_observed_t_) return false;
  return now - outpost_observed_t_ <= OUTPOST_OBSERVATION_MAX_AGE;
}

std::optional<Eigen::Vector4d> Target::outpost_observed_xyza() const
{
  if (!outpost_observation_fresh(t_)) return std::nullopt;
  return outpost_observed_xyza_;
}

void Target::update_outpost_height(const Armor & armor, int id)
{
  if (!is_outpost_ || id < 0 || id >= OUTPOST_ARMOR_NUM) return;
  const double delta = armor.xyz_in_world[2] - ekf_.x[4];
  if (!std::isfinite(delta) || std::abs(delta) > OUTPOST_MAX_HEIGHT_OFFSET) return;

  const double confidence = std::isfinite(armor.confidence) ? armor.confidence : 0.0;
  if (outpost_has_frame_sample_[id]) {
    if (t_ == outpost_frame_sample_t_[id]) {
      if (confidence > outpost_frame_sample_confidence_[id]) {
        outpost_frame_height_sample_[id] = delta;
        outpost_frame_sample_confidence_[id] = confidence;
      }
      return;
    }
    if (t_ < outpost_frame_sample_t_[id]) return;
    if (t_ - outpost_frame_sample_t_[id] > OUTPOST_HEIGHT_SAMPLE_MAX_GAP) {
      outpost_pending_count_[id] = 0;
    } else {
      accept_outpost_height_sample(id, outpost_frame_height_sample_[id]);
    }
  }

  outpost_has_frame_sample_[id] = true;
  outpost_frame_sample_t_[id] = t_;
  outpost_frame_height_sample_[id] = delta;
  outpost_frame_sample_confidence_[id] = confidence;
}

void Target::accept_outpost_height_sample(int id, double delta)
{
  if (outpost_height_known_[id] && std::abs(delta - outpost_height_offsets_[id]) <=
                                      OUTPOST_PENDING_CONSISTENCY_TOLERANCE) {
    outpost_pending_count_[id] = 0;
    return;
  }

  if (outpost_pending_count_[id] > 0) {
    const double median = outpost_pending_median(id);
    if (std::abs(delta - median) > OUTPOST_PENDING_CONSISTENCY_TOLERANCE) {
      outpost_pending_count_[id] = 0;
    }
  }
  if (outpost_pending_count_[id] >= OUTPOST_REMAP_CONFIRM_FRAMES)
    outpost_pending_count_[id] = OUTPOST_REMAP_CONFIRM_FRAMES - 1;
  outpost_pending_samples_[id][outpost_pending_count_[id]] = delta;
  outpost_pending_count_[id]++;

  const double candidate = outpost_pending_median(id);
  const int required_frames = outpost_height_known_[id]
                                ? OUTPOST_REMAP_CONFIRM_FRAMES
                                : OUTPOST_HEIGHT_CONFIRM_FRAMES;
  if (outpost_pending_count_[id] < required_frames) return;

  const bool was_known = outpost_height_known_[id];
  outpost_height_offsets_[id] = candidate;
  outpost_height_known_[id] = true;
  outpost_pending_count_[id] = 0;
  normalize_outpost_height_levels();

  if (outpost_height_ready_ && !was_known) {
    tools::logger()->info(
      "[Target] outpost id {} height offset {:.3f} m, levels: {}/{}/{}", id,
      outpost_height_offsets_[id], outpost_height_levels_[0], outpost_height_levels_[1],
      outpost_height_levels_[2]);
  } else if (was_known) {
    tools::logger()->info(
      "[Target] outpost id {} height mapping updated to {:.3f} m", id, candidate);
  }
}

void Target::normalize_outpost_height_levels()
{
  outpost_height_ready_ = false;
  outpost_height_separation_valid_ = true;
  outpost_height_levels_.fill(-1);
  for (int i = 0; i < OUTPOST_ARMOR_NUM; i++) {
    if (!outpost_height_known_[i]) return;
  }

  std::array<double, OUTPOST_ARMOR_NUM> sorted_offsets = outpost_height_offsets_;
  std::sort(sorted_offsets.begin(), sorted_offsets.end());
  if (
    sorted_offsets[1] - sorted_offsets[0] < outpost_min_height_separation_ ||
    sorted_offsets[2] - sorted_offsets[1] < outpost_min_height_separation_) {
    outpost_height_separation_valid_ = false;
    tools::logger()->warn(
      "[Target] outpost height planes too close: {:.3f}/{:.3f}/{:.3f} m", sorted_offsets[0],
      sorted_offsets[1], sorted_offsets[2]);
    return;
  }

  for (int i = 0; i < OUTPOST_ARMOR_NUM; ++i) {
    const auto it = std::find(
      sorted_offsets.begin(), sorted_offsets.end(), outpost_height_offsets_[i]);
    outpost_height_levels_[i] = static_cast<int>(std::distance(sorted_offsets.begin(), it));
  }
  outpost_height_ready_ = true;
}

double Target::outpost_pending_median(int id) const
{
  std::array<double, OUTPOST_REMAP_CONFIRM_FRAMES> sorted{};
  std::copy_n(outpost_pending_samples_[id].begin(), outpost_pending_count_[id], sorted.begin());
  std::sort(sorted.begin(), sorted.begin() + outpost_pending_count_[id]);
  return sorted[outpost_pending_count_[id] / 2];
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  if (is_outpost_ && id >= 0 && id < OUTPOST_ARMOR_NUM)
    armor_z += outpost_height_offsets_[id];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;
  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
