import rclpy
from control_msgs.action import ParallelGripperCommand
from giskardpy.executor import Executor
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    RobotiqGripperActionServerTask,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from coraplex.alternative_motion_mappings import tiago_motion_mapping
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.world import World
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoGripperMotion


def robotiq_gripper_dataclass_test():
    rclpy_node = rclpy.create_node("tiago_demo_node")
    giskard = GiskardWrapper(node_handle=rclpy_node)

    task1 = TiagoGripperMotion(
        action_topic=Arms.BOTH,
        message_type=ParallelGripperCommand,
        target_position=GripperState.OPEN,
    )

    task2 = TiagoGripperMotion(
        action_topic=Arms.BOTH,
        message_type=ParallelGripperCommand,
        target_position=GripperState.CLOSED,
    )

    msc = MotionStatechart()
    msc.add_node(seq := Sequence([task1]))
    msc.add_node(EndMotion.when_true(seq))

    giskard.execute(msc)

    msc = MotionStatechart()
    msc.add_node(seq := Sequence([task2]))
    msc.add_node(EndMotion.when_true(seq))

    giskard.execute(msc)


if __name__ == "__main__":
    robotiq_gripper_dataclass_test()
