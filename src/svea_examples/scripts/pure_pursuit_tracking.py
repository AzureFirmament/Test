#! /usr/bin/env python3

import numpy as np
import pandas as pd
import math

from svea_core.interfaces import LocalizationInterface
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from svea_core import rosonic as rx


class pure_pursuit(rx.Node):

    r"""Pure Pursuit example script for SVEA.

    #**Background**

    This script implements a simple Pure Pursuit controller that follows a
    predefined path. The path is defined by a set of points, and the controller
    computes the steering angle and velocity to follow the path.

    The script also includes visualization of the goal and the path being
    followed.

    #**Preparation**

    TODO: Add instructions for setting up the teleoperation environment.

    #**Simulation**

    To run the Pure Pursuit example in simulation, you can use the following command:
    ```bash
    ros2 launch svea_examples floor2.xml is_sim:=true
    ```
    This launch file includes the following components, with example parameters:

        # Initial state of the robot (x, y, yaw, velocity)
        state:=[-7.4, -15.3, 0.9, 0.0] 
        # Points defining the path to follow. Each point is a string representation of a list.
        points:=['[-2.3,-7.1]','[10.5,11.7]','[5.7,15.0]','[-7.0,-4.0]'] 

    Attributes:
        points: List of points defining the path to follow.
        actuation: Actuation interface for sending control commands.
        localizer: Localization interface for receiving state information.
        goal_mark: ShowMarker for visualizing the goal.
        path: ShowPath for visualizing the path.
    """

    DELTA_TIME = 0.1
    TRAJ_LEN = 20

    bias_x = 1.5
    bias_y = 3.0

    points = rx.Parameter('[[-2.3, -7.1], [10.5, 11.7], [5.7, 15.0], [-7.0, -4.0]]')
    target_velocity = rx.Parameter(0.6)

    data_len = 500
    traj_data = np.zeros((int(data_len), 2), dtype=np.float32)
    index = 0

    start_flag = False
    
    # Interfaces
    
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    
    goal_marker = ShowMarker() # for goal visualization
    visual = ShowPath() # for path visualization

    def on_startup(self):
        """
        Initialize the Pure Pursuit controller and set up the path and goal.
        Controller is initialized with the target velocity and the points
        provided in the parameters. The current state is obtained from the
        localization interface, and the goal is set to the first point in the
        path.
        The trajectory is updated based on the current state and the goal.
        The controller is set to not finished initially, and a timer is created
        to call the loop method at regular intervals.
        """
        # Convert parameter to numerical list
        self._points = eval(self.points)

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        path = pd.read_csv(r'/svea_ws/src/svea_examples/params/tiha_path_data.csv',
                           header = 0)
        
        self.path = path.to_numpy()
        print(self.path.shape[0])
        for i in range(self.path.shape[0] - 1):
            print(i)
            if self.path[i][3] > 0 and self.path[i + 1][3] < 0:
                self.cusp_index = i
                print(f"Cusp index found at: {self.cusp_index}")
                break

        self.goal = [self.path[self.cusp_index][1], -self.path[self.cusp_index][0]]

        # self.curr = 0
        # self.goal = self._points[self.curr]
        self.goal_marker.place([*self.goal, 0.5], color='blue')
        # self.update_traj(x, y)
        self.controller.traj_x = self.path[:self.cusp_index + 1][1]
        self.controller.traj_y = -self.path[:self.cusp_index + 1][0]
        self.visual.publish_path(self.controller.traj_x, self.controller.traj_y)

        self.create_timer(self.DELTA_TIME, self.loop)

    def loop(self):
        """
        Main loop of the Pure Pursuit controller. It retrieves the current state
        from the localization interface, computes the steering and velocity
        commands using the controller, and sends these commands to the actuation
        interface.
        If the controller has finished following the path, it updates the goal
        and trajectory based on the next point in the path.
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if not self.start_flag:
            if (x - self.controller.traj_x[0])**2 + (y - self.controller.traj_y[0])**2 < 0.2**2:
                self.start_flag = True
                self.get_logger().info("Starting path following.")
                self.actuation.send_control(0,-2.0)
            else:
                self.actuation.send_control(0, -2.0)

        else:
            if self.index < self.data_len:
                self.get_logger().info("Record trajectory data")
                self.traj_data[self.index] = [x, y]
                self.index += 1
                if self.index == self.data_len:
                    df = pd.DataFrame(self.traj_data, columns=['x', 'y'])
                    df.to_csv(r'/svea_ws/src/svea_examples/params/traj_data.csv', index=False)
                    self.get_logger().info("Trajectory data saved to traj_data.csv")

            if self.controller.is_finished:
                self.get_logger().info("Path completed. Updating goal and trajectory.")
                # self.update_goal()
                # self.update_traj(x, y)
                self.update_path()

            target = self.select_target(state)
            steering, velocity = self.controller.compute_control(state, target)
            self.get_logger().info(f"target point on path: {self.controller.target}")
            # self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
            self.get_logger().info(f"goal: {self.goal}, current state: {state}")
            self.actuation.send_control(steering, -2.0)

    def update_goal(self):
        """
        Update the goal to the next point in the path. If the end of the path
        is reached, it wraps around to the beginning. The current index is
        incremented, and the goal marker is updated.
        """
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.goal_marker.place([*self.goal, 0.5], color='blue')

    def update_traj(self, x, y):
        """
        Update the trajectory based on the current state and the goal. It
        generates a linear trajectory from the current position to the goal
        position, and updates the controller's trajectory points.
        The trajectory is visualized using the ShowPath interface.
        """
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys
        self.visual.publish_path(xs,ys)

    def update_path(self):
        """
        Update the path based on the current state and the goal. It generates a
        linear path from the current position to the goal position, and updates
        the controller's path points.
        The path is visualized using the ShowPath interface.
        """
        self.controller.is_finished = False
        self.goal = [self.path[-1][1], -self.path[-1][0]]
        self.goal_marker.place([*self.goal, 0.5], color='blue')
        self.target_velocity = - self.target_velocity
        self.controller.traj_x = self.path[:self.cusp_index + 1][1]
        self.controller.traj_y = -self.path[:self.cusp_index + 1][0]
        self.visual.publish_path(self.controller.traj_x, self.controller.traj_y)

    def select_target(self, state):
        """
        Select the target point on the path based on the current state. It
        computes the distance from the current position to each point in the
        path, and selects the point that is within the look-ahead distance.
        If no point is found within the look-ahead distance, it selects the
        last point in the path.
        """
        x, y, yaw, vel = state
        min_dist = float('inf')
        target_index = None

        dx = [x - icx for icx in self.controller.traj_x]
        dy = [y - icy for icy in self.controller.traj_y]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        ind = d.index(min(d))
        Lf = 0.2
        dist = 0.0

        # search look ahead target point index
        while Lf > dist and (ind + 1) < len(self.controller.traj_x):
            dx = self.controller.traj_x[ind + 1] - self.controller.traj_x[ind]
            dy = self.controller.traj_y[ind + 1] - self.controller.traj_y[ind]
            dist += math.sqrt(dx ** 2 + dy ** 2)
            ind += 1

        return self.controller.traj_x[ind], self.controller.traj_y[ind]
if __name__ == '__main__':
    pure_pursuit.main()