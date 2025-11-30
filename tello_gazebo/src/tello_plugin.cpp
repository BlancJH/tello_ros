#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Entity.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/common/Console.hh>

#include <rclcpp/rclcpp.hpp>
#include "geometry_msgs/msg/twist.hpp"
#include "tello_msgs/msg/flight_data.hpp"
#include "tello_msgs/msg/tello_response.hpp"
#include "tello_msgs/srv/tello_action.hpp"

#include "pid.hpp"

using namespace std::chrono_literals;

//===============================================================================================
// TelloPlugin features:
// -- generates trivial flight data at 10Hz
// -- video is managed by a gazebo_ros_pkgs plugin, see tello_description/urdf/tello.xml
// -- responds to "takeoff" and "land" commands
// -- responds to cmd_vel and "rc x y z yaw" commands
// -- battery state
//
// Tello flight dynamics are sophisticated and difficult to model. TelloPlugin keeps it simple:
// -- x, y, z and yaw velocities are controlled by a P controller
// -- roll and pitch are always 0
//
// Possible extensions:
// -- simulate network latency
// -- add some randomness to the flight dynamics
// -- improve battery simulation
//===============================================================================================

namespace tello_gazebo
{

  namespace
  {
    const double MAX_XY_V = 8.0;
    const double MAX_Z_V = 4.0;
    const double MAX_ANG_V = M_PI;

    const double MAX_XY_A = 8.0;
    const double MAX_Z_A = 4.0;
    const double MAX_ANG_A = M_PI;

    const double TAKEOFF_Z = 1.0;
    const double TAKEOFF_Z_V = 0.5;

    const double LAND_Z = 0.1;
    const double LAND_Z_V = -0.5;

    const int BATTERY_DURATION = 6000;

    inline double clamp(const double v, const double max)
    {
      return v > max ? max : (v < -max ? -max : v);
    }
  }  // namespace

  class TelloPlugin : public gz::sim::System,
                      public gz::sim::ISystemConfigure,
                      public gz::sim::ISystemPreUpdate  {
    enum class FlightState
    {
      landed,
      taking_off,
      flying,
      landing,
      dead_battery,
    };


    std::map<FlightState, const char *> state_strs_{{
      {FlightState::landed, "landed"},
      {FlightState::taking_off, "taking_off"},
      {FlightState::flying, "flying"},
      {FlightState::landing, "landing"},
      {FlightState::dead_battery, "dead_battery"},
    }};

    FlightState flight_state_{FlightState::landed};

    gz::sim::Model model_{gz::sim::kNullEntity};
    gz::sim::Link base_link_{gz::sim::kNullEntity};
    gz::math::Vector3d gravity_{0, 0, -9.81};
    gz::math::Vector3d center_of_mass_{0, 0, 0};
    std::chrono::duration<double> battery_duration_{std::chrono::seconds(BATTERY_DURATION)};

    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::SingleThreadedExecutor executor_;

    rclcpp::Publisher<tello_msgs::msg::FlightData>::SharedPtr flight_data_pub_;
    rclcpp::Publisher<tello_msgs::msg::TelloResponse>::SharedPtr tello_response_pub_;

    rclcpp::Service<tello_msgs::srv::TelloAction>::SharedPtr command_srv_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    std::chrono::steady_clock::duration last_10hz_{};

    pid::Controller x_controller_{false, 2, 0, 0};
    pid::Controller y_controller_{false, 2, 0, 0};
    pid::Controller z_controller_{false, 2, 0, 0};
    pid::Controller yaw_controller_{false, 2, 0, 0};

  public:
    TelloPlugin()
    {
      transition(FlightState::landed);
    }

    void Configure(
      const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &sdf,
      gz::sim::EntityComponentManager &ecm,
      gz::sim::EventManager & /*event_mgr*/) override
    {
      model_ = gz::sim::Model(entity);

      std::string link_name{"base_link"};
      if (sdf && sdf->HasElement("link_name")) {
        link_name = sdf->Get<std::string>("link_name");
      }

      if (sdf) {
        center_of_mass_ = sdf->Get<gz::math::Vector3d>("center_of_mass", center_of_mass_).first;
        battery_duration_ = std::chrono::seconds(
          sdf->Get<int>("battery_duration", BATTERY_DURATION).first);
      }

      const auto link_entity = model_.LinkByName(ecm, link_name);
      if (link_entity == gz::sim::kNullEntity) {
        gzerr << "Missing link: " << link_name << std::endl;
        return;
      }

      base_link_ = gz::sim::Link(link_entity);

      gz::sim::World world(model_.World(ecm));
      auto gravity_opt = world.Gravity(ecm);
      if (gravity_opt) {
        gravity_ = *gravity_opt;
      }

      if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
      }

      node_ = rclcpp::Node::make_shared("tello_gazebo");
      node_->set_parameter(rclcpp::Parameter("use_sim_time", true));
      executor_.add_node(node_);

      flight_data_pub_ = node_->create_publisher<tello_msgs::msg::FlightData>("flight_data", 1);
      tello_response_pub_ = node_->create_publisher<tello_msgs::msg::TelloResponse>("tello_response", 1);

      command_srv_ = node_->create_service<tello_msgs::srv::TelloAction>(
        "tello_action",
        std::bind(&TelloPlugin::command_callback, this,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3));

      cmd_vel_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", rclcpp::QoS(10),
        std::bind(&TelloPlugin::cmd_vel_callback, this, std::placeholders::_1));
    }

    void PreUpdate(
      const gz::sim::UpdateInfo &info, gz::sim::EntityComponentManager &ecm) override
    {
      if (!node_) {
        return;
      }

      executor_.spin_some();

      const auto sim_time = info.simTime;

      if ((sim_time - last_10hz_) > 100ms) {
        spin_10Hz(sim_time, ecm);
        last_10hz_ = sim_time;
      }

      const double dt = std::chrono::duration<double>(info.dt).count();
      if (dt <= 0) {
        return;
      }

      if (flight_state_ != FlightState::landed) {
        const auto linear_velocity_opt = base_link_.WorldLinearVelocity(ecm);
        const auto angular_velocity_opt = base_link_.WorldAngularVelocity(ecm);
        const auto mass_matrix_opt = base_link_.MassMatrix(ecm);

        if (!linear_velocity_opt || !angular_velocity_opt || !mass_matrix_opt) {
          return;
        }

        auto linear_velocity = *linear_velocity_opt;
        auto angular_velocity = *angular_velocity_opt;
        auto mass_matrix = *mass_matrix_opt;

        gz::math::Vector3d lin_ubar, ang_ubar;
        lin_ubar.X(x_controller_.calc(linear_velocity.X(), dt, 0));
        lin_ubar.Y(y_controller_.calc(linear_velocity.Y(), dt, 0));
        lin_ubar.Z(z_controller_.calc(linear_velocity.Z(), dt, 0));
        ang_ubar.Z(yaw_controller_.calc(angular_velocity.Z(), dt, 0));

        lin_ubar.X() = clamp(lin_ubar.X(), MAX_XY_A);
        lin_ubar.Y() = clamp(lin_ubar.Y(), MAX_XY_A);
        lin_ubar.Z() = clamp(lin_ubar.Z(), MAX_Z_A);
        ang_ubar.Z() = clamp(ang_ubar.Z(), MAX_ANG_A);

        lin_ubar -= gravity_;

        const auto principal_moments = mass_matrix.PrincipalMoments();
        const gz::math::Vector3d force = lin_ubar * mass_matrix.Mass();
        const gz::math::Vector3d torque(
          ang_ubar.X() * principal_moments.X(),
          ang_ubar.Y() * principal_moments.Y(),
          ang_ubar.Z() * principal_moments.Z());

        const auto pose_opt = base_link_.WorldPose(ecm);
        if (pose_opt) {
          auto pose = *pose_opt;
          auto rot = pose.Rot();
          rot.Set(0, 0, rot.Yaw());
          pose.Set(pose.Pos(), rot);
          base_link_.SetWorldPose(ecm, pose);
        }

        base_link_.AddWorldWrench(ecm, force, torque);
      }
    }

  private:
    void set_target_velocities(const double x, const double y, const double z, const double yaw)
    {
      x_controller_.set_target(x);
      y_controller_.set_target(y);
      z_controller_.set_target(z);
      yaw_controller_.set_target(yaw);
    }

    void set_target_velocities(const std::string &rc_command)
    {
      double x{}, y{}, z{}, yaw{};

      try {
        std::istringstream iss(rc_command, std::istringstream::in);
        std::string s;
        iss >> s;
        iss >> s;
        x = std::stof(s);
        iss >> s;
        y = std::stof(s);
        iss >> s;
        z = std::stof(s);
        iss >> s;
        yaw = std::stof(s);
      } catch (const std::exception &e) {
        RCLCPP_ERROR(node_->get_logger(), "can't parse rc command '%s', exception %s",
          rc_command.c_str(), e.what());
        return;
      }

      set_target_velocities(x * MAX_XY_V, y * MAX_XY_V, z * MAX_Z_V, yaw * MAX_ANG_V);
    }

    void transition(const FlightState next)
    {
      if (node_ != nullptr) {
        RCLCPP_INFO(
          node_->get_logger(), "transition from '%s' to '%s'", state_strs_[flight_state_],
          state_strs_[next]);
      }

      flight_state_ = next;

      switch (flight_state_) {
        case FlightState::landed:
        case FlightState::flying:
        case FlightState::dead_battery:
          set_target_velocities(0, 0, 0, 0);
          break;

        case FlightState::taking_off:
          set_target_velocities(0, 0, TAKEOFF_Z_V, 0);
          break;

        case FlightState::landing:
          set_target_velocities(0, 0, LAND_Z_V, 0);
          break;
      }
    }

    void command_callback(
      const std::shared_ptr<rmw_request_id_t> /*request_header*/,
      const std::shared_ptr<tello_msgs::srv::TelloAction::Request> request,
      std::shared_ptr<tello_msgs::srv::TelloAction::Response> response)
    {
      if (request->cmd == "takeoff" && flight_state_ == FlightState::landed) {
        transition(FlightState::taking_off);
        response->rc = response->OK;
      } else if (request->cmd == "land" && flight_state_ == FlightState::flying) {
        transition(FlightState::landing);
        response->rc = response->OK;
      } else if (is_prefix("rc", request->cmd) && flight_state_ == FlightState::flying) {
        set_target_velocities(request->cmd);
        response->rc = response->OK;
      } else {
        RCLCPP_WARN(node_->get_logger(), "ignoring command '%s'", request->cmd.c_str());
        response->rc = response->ERROR_BUSY;
      }
    }

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
      if (flight_state_ == FlightState::flying) {
        set_target_velocities(
          msg->linear.x * MAX_XY_V,
          msg->linear.y * MAX_XY_V,
          msg->linear.z * MAX_Z_V,
          msg->angular.z * MAX_ANG_V);
      }
    }

    static bool is_prefix(const std::string &prefix, const std::string &str)
    {
      return std::equal(prefix.begin(), prefix.end(), str.begin());
    }

    void respond_ok()
    {
      tello_msgs::msg::TelloResponse msg;
      msg.rc = msg.OK;
      msg.str = "ok";
      tello_response_pub_->publish(msg);
    }

    void spin_10Hz(
      const std::chrono::steady_clock::duration &sim_time,
      gz::sim::EntityComponentManager &ecm)
    {
      const auto sim_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(sim_time);
      const rclcpp::Time ros_time(sim_time_ns.count(), RCL_ROS_TIME);

      if (ros_time.seconds() < 1.0) {
        return;
      }

      const double elapsed = std::chrono::duration<double>(sim_time).count();
      const int battery_percent = static_cast<int>((battery_duration_.count() - elapsed) /
        battery_duration_.count() * 100);
      if (battery_percent <= 0) {
        transition(FlightState::dead_battery);
        return;
      }

      tello_msgs::msg::FlightData flight_data;
      flight_data.header.stamp = ros_time;
      flight_data.sdk = flight_data.SDK_1_3;
      flight_data.bat = battery_percent;
      flight_data_pub_->publish(flight_data);

      const auto pose_opt = base_link_.WorldPose(ecm);
      if (!pose_opt) {
        return;
      }

      const auto &pose = *pose_opt;
      if (flight_state_ == FlightState::taking_off && pose.Pos().Z() > TAKEOFF_Z) {
        transition(FlightState::flying);
        respond_ok();
      } else if (flight_state_ == FlightState::landing && pose.Pos().Z() < LAND_Z) {
        transition(FlightState::landed);
        respond_ok();
      }
    }
  };

  GZ_ADD_PLUGIN(
    TelloPlugin,
    gz::sim::System,
    gz::sim::ISystemConfigure,
    gz::sim::ISystemPreUpdate);
}  // namespace tello_gazebo