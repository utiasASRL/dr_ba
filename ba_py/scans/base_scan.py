from abc import ABC, abstractmethod
import numpy as np

class BaseScan(ABC):
    def __init__(self, pose, scan_id):
        self.pose = pose
        self.pose_2d = np.eye(3)
        self.pose_2d[0:2, 0:2] = pose[0:2, 0:2]
        self.pose_2d[0:2, 2] = pose[0:2, 3]
        self.id = scan_id

    def update_pose(self, new_pose):
        self.pose = new_pose
        self.pose_2d[0:2, 0:2] = new_pose[0:2, 0:2]
        self.pose_2d[0:2, 2] = new_pose[0:2, 3]

    @abstractmethod
    def interpolate(self, x, y, jac=False):
        pass

    @abstractmethod
    def check_coverage_at_point(self, x, y):
        pass