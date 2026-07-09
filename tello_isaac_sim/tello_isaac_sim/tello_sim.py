#!/usr/bin/env python3

import os
import sys

# Set up Isaac Sim app first
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# You might need to build tello_msgs in the workspace to import this
try:
    from tello_msgs.srv import TelloAction
    from tello_msgs.msg import FlightData
except ImportError:
    print("Warning: tello_msgs not found. TelloAction service will not be available.")
    TelloAction = None

from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.stage import get_current_stage
import omni.graph.core as og
from omni.isaac.urdf import _urdf
from pxr import UsdGeom, Gf, UsdPhysics

class TelloIsaacSimNode(Node):
    def __init__(self, world):
        super().__init__('tello_isaac_sim')
        self.world = world
        
        # Load URDF plugin
        enable_extension("omni.isaac.urdf")
        
        # Import Tello URDF
        urdf_interface = _urdf.acquire_urdf_interface()
        
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = False
        import_config.make_default_prim = False
        
        # Paths
        pkg_share = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tello_description')
        if not os.path.exists(pkg_share):
            # Assume running from install space
            pkg_share = '/home/blancjh/tello_ros/tello_description'
            
        urdf_path = os.path.join(pkg_share, 'urdf', 'tello.xml')
        dest_path = '/tello'
        
        result, prim_path = urdf_interface.parse_urdf(urdf_path, import_config, dest_path)
        
        if result:
            self.get_logger().info(f"Loaded Tello URDF at {prim_path}")
        else:
            self.get_logger().error(f"Failed to load Tello URDF from {urdf_path}")
            
        # ROS 2 Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # ROS 2 Services
        if TelloAction:
            self.action_srv = self.create_service(
                TelloAction,
                'tello_action',
                self.action_callback
            )
            
        self.response_pub = self.create_publisher(String, 'tello_response', 10)
        
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_yaw_rate = 0.0
        self.is_flying = False
        
        # Setup ROS2 Camera Bridge via OmniGraph
        self.setup_camera_bridge()

    def setup_camera_bridge(self):
        # Create a basic camera and attach it to the drone's camera_link
        import omni.isaac.core.utils.prims as prims_utils
        camera_path = "/tello/camera_link/Camera"
        prims_utils.create_prim(camera_path, "Camera", translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
        
        try:
            keys = og.Controller.Keys
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": "/ROS_CameraGraph", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("CreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ],
                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim", [camera_path]),
                        ("CreateRenderProduct.inputs:resolution", [960, 720]),
                        ("ROS2CameraHelper.inputs:topicName", "image_raw"),
                        ("ROS2CameraHelper.inputs:type", "rgb"),
                        ("ROS2CameraHelper.inputs:nodeNamespace", ""),
                    ],
                },
            )
            self.get_logger().info("Setup ROS2 Camera Bridge graph.")
        except Exception as e:
            self.get_logger().error(f"Failed to setup Camera bridge: {e}")

    def cmd_vel_callback(self, msg):
        # Scale inputs similarly to the real driver
        self.target_velocity[0] = msg.linear.x
        self.target_velocity[1] = msg.linear.y
        self.target_velocity[2] = msg.linear.z
        self.target_yaw_rate = msg.angular.z

    def action_callback(self, request, response):
        cmd = request.cmd
        self.get_logger().info(f"Received tello action: {cmd}")
        
        res_msg = String()
        if cmd == 'takeoff':
            self.is_flying = True
            self.target_velocity[2] = 0.5  # move up
            response.rc = 1
            res_msg.data = 'ok'
        elif cmd == 'land':
            self.is_flying = False
            self.target_velocity[2] = -0.5 # move down
            response.rc = 1
            res_msg.data = 'ok'
        else:
            response.rc = 0
            res_msg.data = 'error'
            
        self.response_pub.publish(res_msg)
        return response
        
    def physics_step(self, step_size):
        # Here we apply forces to the USD prim to simulate quadcopter flight
        stage = get_current_stage()
        tello_prim = stage.GetPrimAtPath('/tello/base_link')
        if not tello_prim.IsValid():
            return
            
        rigid_body_api = UsdPhysics.RigidBodyAPI(tello_prim)
        if not rigid_body_api:
            # Need to apply physics API to the prim if not present
            UsdPhysics.RigidBodyAPI.Apply(tello_prim)
            UsdPhysics.MassAPI.Apply(tello_prim)
            
        if self.is_flying:
            import omni.isaac.core.utils.prims as prims_utils
            from pxr import Gf
            
            # Simple proportional controller on velocity
            # In a real scenario, use Isaac Sim's multirotor physics controller
            current_velocity = rigid_body_api.GetVelocityAttr().Get()
            if current_velocity is None:
                current_velocity = Gf.Vec3f(0.0, 0.0, 0.0)
            
            error_x = self.target_velocity[0] - current_velocity[0]
            error_y = self.target_velocity[1] - current_velocity[1]
            error_z = self.target_velocity[2] - current_velocity[2]
            
            force_scale = 10.0 # simple P gain
            force = Gf.Vec3f(error_x * force_scale, error_y * force_scale, error_z * force_scale)
            
            # Apply force (using simplified set velocity for demo purposes)
            rigid_body_api.GetVelocityAttr().Set(Gf.Vec3f(
                float(self.target_velocity[0]),
                float(self.target_velocity[1]),
                float(self.target_velocity[2])
            ))
            
            # Also set angular velocity
            angular_velocity = rigid_body_api.GetAngularVelocityAttr().Get()
            rigid_body_api.GetAngularVelocityAttr().Set(Gf.Vec3f(
                0.0,
                0.0,
                float(self.target_yaw_rate)
            ))
        else:
            # If not flying, let gravity pull it down (assuming physics is enabled)
            pass

def main(args=None):
    rclpy.init(args=args)
    
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    node = TelloIsaacSimNode(world)
    
    world.reset()
    
    while simulation_app.is_running():
        rclpy.spin_once(node, timeout_sec=0.0)
        world.step(render=True)
        node.physics_step(world.get_physics_dt())
        
    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == '__main__':
    main()
