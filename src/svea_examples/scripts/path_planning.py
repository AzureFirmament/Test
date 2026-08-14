#! /usr/bin/env python3

import csv
import math
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from svea_core.interfaces import LocalizationInterface, ActuationInterface
from svea_core.controllers.pure_pursuit import PurePursuitController
try:
    from svea_core.controllers.mpc import MPC
except ImportError:
    MPC = None
from svea_core.interfaces import ShowMarker, ShowPath

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseArray, Pose

from svea_core import rosonic as rx


qos_subber = QoSProfile(
    depth=10,
)

qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def _quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class path_planning(rx.Node):
    r"""CSV path publisher for SVEA.

    #**Background**

    This node reads a path from a CSV file and publishes it as a
    ``geometry_msgs/PoseArray`` on the ``/static_path`` topic so that a path
    tracking node (e.g. ``path_tracking``) can follow it.

    The CSV file must contain three columns named ``x``, ``y`` and ``yaw``.

    #**Preparation**

    The ``csv_path`` parameter must point to a readable CSV file.

    #**Simulation**

    To run this node in simulation, launch the simulator and start this node
    separately. For example:

    ```bash
    ros2 launch svea_examples floor2.xml is_sim:=true
    ros2 run svea_examples path_planning \
        csv_path:=/path/to/path.csv
    ```

    Attributes:
        csv_path: Absolute path to the CSV file describing the path.
    """

    csv_path = rx.Parameter('')

    path_pub = rx.Publisher(PoseArray, '/static_path', qos_profile=qos_pubber)

    def on_startup(self):
        if not self.csv_path:
            self.get_logger().warn('No csv_path parameter provided; nothing will be published.')
            return

        try:
            poses = self._load_csv(self.csv_path)
        except Exception as e:
            self.get_logger().error(f'Failed to load path from "{self.csv_path}": {e}')
            return

        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.poses = poses
        self.path_pub.publish(msg)
        self.get_logger().info(f'Published path with {len(poses)} points from "{self.csv_path}".')

    def _load_csv(self, csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError('CSV file is empty or has no header.')
            cols = [c.strip().lower() for c in reader.fieldnames]
            for required in ('x', 'y', 'yaw'):
                if required not in cols:
                    raise ValueError(f"CSV must contain a '{required}' column.")
            x_idx = cols.index('x')
            y_idx = cols.index('y')
            yaw_idx = cols.index('yaw')

            xs, ys, yaws = [], [], []
            for row in reader:
                xs.append(float(row[reader.fieldnames[x_idx]]))
                ys.append(float(row[reader.fieldnames[y_idx]]))
                yaws.append(float(row[reader.fieldnames[yaw_idx]]))

        if not xs:
            raise ValueError('CSV file contains no data rows.')

        poses = []
        for x, y, yaw in zip(xs, ys, yaws):
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.z = math.sin(float(yaw) / 2.0)
            pose.orientation.w = math.cos(float(yaw) / 2.0)
            poses.append(pose)
        return poses


class path_tracking(rx.Node):
    r"""Path tracking node for SVEA.

    #**Background**

    This node subscribes to a ``PoseArray`` published on ``/static_path`` (for
    instance by the ``path_planning`` node, which reads the path from a CSV
    file) and follows it. Two controllers are supported and can be selected via
    the ``controller_type`` parameter: ``pure_pursuit`` and ``mpc``.

    The style of this node mirrors ``pure_pursuit.py`` and ``mpc_path_tracking.py``
    so that it fits naturally with the other example scripts.

    #**Preparation**

    Before this node can track, a path must be available on ``/static_path``,
    e.g. by launching ``path_planning`` with a valid ``csv_path`` parameter.

    #**Simulation**

    ```bash
    ros2 launch svea_examples floor2.xml is_sim:=true
    ros2 run svea_examples path_planning csv_path:=/path/to/path.csv
    ros2 run svea_examples path_tracking
    ```

    Attributes:
        controller_type: Either ``pure_pursuit`` or ``mpc``.
        target_velocity: Desired forward velocity (pure pursuit).
        target_speed: Desired forward speed (mpc).
        actuation: Actuation interface for sending control commands.
        localizer: Localization interface for receiving state information.
        goal_marker: ShowMarker for visualizing the current waypoint (pure pursuit).
        path: ShowPath for visualizing the received path.
    """

    DELTA_TIME = 0.1

    controller_type = rx.Parameter('pure_pursuit')

    target_velocity = rx.Parameter(0.6)

    is_sim = rx.Parameter(True)
    mpc_freq = rx.Parameter(10)
    target_speed = rx.Parameter(0.5)
    svea_mocap_name = rx.Parameter('svea7')
    mpc_config_ns = rx.Parameter('/mpc')
    time_step = rx.Parameter(0.2)
    prediction_horizon = rx.Parameter(5)

    actuation = ActuationInterface()
    localizer = LocalizationInterface()

    goal_marker = ShowMarker()
    path = ShowPath()

    def on_startup(self):
        self._has_path = False
        self._finished = False
        self._path_xs = None
        self._path_ys = None
        self._path_yaws = None
        self._curr = 0

        ctype = str(self.controller_type).lower()
        if ctype == 'mpc':
            if MPC is None:
                self.get_logger().error('MPC controller is not available; falling back to pure_pursuit.')
                ctype = 'pure_pursuit'
            else:
                self.controller = MPC(self)
                self.mpc_dt = 1.0 / self.mpc_freq
                self.mpc_propagation_dt = self.time_step
                self.initial_horizon = self.prediction_horizon
                self.static_path_plan = np.empty((3, 0))
                self.steering = 0.0
                self.velocity = 0.0
                self.mpc_last_time = 0.0
                self._setup_steering_bias()
        if ctype == 'pure_pursuit':
            self.controller = PurePursuitController()
            self.controller.target_velocity = self.target_velocity

        self._ctype = ctype
        self.create_timer(self.DELTA_TIME, self.loop)

    def _setup_steering_bias(self):
        if self.is_sim:
            self.steering_bias = 0.0
            return
        unitless_steering_map = {
            "svea0": 28,
            "svea7": 7,
        }
        svea_name = self.svea_mocap_name.lower()
        unitless_steering = unitless_steering_map.get(svea_name, 0)
        PERC_TO_LLI_COEFF = 1.27
        MAX_STEERING_ANGLE = 40 * math.pi / 180
        steer_percent = unitless_steering / PERC_TO_LLI_COEFF
        self.steering_bias = (steer_percent / 100.0) * MAX_STEERING_ANGLE

    def loop(self):
        if not self._has_path or self._finished:
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self._ctype == 'pure_pursuit':
            if self.controller.is_finished:
                self._curr += 1
                if self._curr >= len(self._path_xs):
                    self._finished = True
                    self.get_logger().info('Reached end of path; stopping.')
                    return
                self._update_goal()
                self._update_traj(x, y)
            steering, velocity = self.controller.compute_control(state)
            self.actuation.send_control(steering, velocity)

        elif self._ctype == 'mpc':
            if self.static_path_plan.shape[1] == 0:
                return
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.mpc_last_time == 0.0:
                self.mpc_last_time = current_time
            measured_dt = current_time - self.mpc_last_time
            if measured_dt >= self.mpc_dt:
                reference_trajectory = self._get_mpc_reference()
                steering_rate, acceleration = self.controller.compute_control(
                    [state[0], state[1], state[2], state[3], self.steering],
                    reference_trajectory,
                )
                self.steering += steering_rate * measured_dt
                self.velocity += acceleration * measured_dt
                self.mpc_last_time = current_time
            self.actuation.send_control(self.steering + self.steering_bias, self.velocity)

    @rx.Subscriber(PoseArray, '/static_path')
    def path_cb(self, msg):
        if not msg.poses:
            self.get_logger().warn('Received empty path; ignoring.')
            return

        self._path_xs = np.array([p.position.x for p in msg.poses])
        self._path_ys = np.array([p.position.y for p in msg.poses])
        self._path_yaws = np.array([_quat_to_yaw(p.orientation) for p in msg.poses])
        self.static_path_plan = np.vstack((self._path_xs, self._path_ys, self._path_yaws))
        self._has_path = True
        self._finished = False
        self._curr = 0

        if self._ctype == 'mpc':
            self.N = len(self._path_xs)
            self.steering = 0.0
            self.velocity = 0.0
            self.mpc_last_time = 0.0

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self._ctype == 'pure_pursuit':
            self._update_goal()
            self._update_traj(x, y)

        self.path.publish_path(self._path_xs, self._path_ys)
        self.get_logger().info(
            f'Received path with {len(msg.poses)} points '
            f'(controller={self._ctype}).'
        )

    def _update_goal(self):
        self.goal = [float(self._path_xs[self._curr]), float(self._path_ys[self._curr])]
        self.controller.is_finished = False
        self.goal_marker.place([*self.goal, 0.5], color='blue')

    def _update_traj(self, x, y):
        xs = np.linspace(x, self.goal[0], 20)
        ys = np.linspace(y, self.goal[1], 20)
        self.controller.traj_x = xs
        self.controller.traj_y = ys

    def _get_mpc_reference(self):
        distances = np.linalg.norm(
            self.static_path_plan[:2, :] - np.array([self.localizer.get_state()[0],
                                                     self.localizer.get_state()[1]])[:, None],
            axis=0,
        )
        start_index = int(np.argmin(distances)) + 1
        end_index = start_index + self.initial_horizon + 1
        if end_index > self.N:
            x_ref = self.static_path_plan[:, start_index:self.N]
            while x_ref.shape[1] < self.initial_horizon + 1:
                x_ref = np.concatenate((x_ref, self.static_path_plan[:, -1:]), axis=1)
        else:
            x_ref = self.static_path_plan[:, start_index:end_index]
        target_speed_row = np.full((1, x_ref.shape[1]), self.target_speed)
        return np.concatenate((x_ref, target_speed_row), axis=0)


def main(args=None):
    rclpy.init(args=args)
    planner = path_planning()
    tracker = path_tracking()
    executor = SingleThreadedExecutor()
    executor.add_node(planner)
    executor.add_node(tracker)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        tracker.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()