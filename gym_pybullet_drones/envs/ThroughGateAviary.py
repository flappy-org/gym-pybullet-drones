import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType


class ThroughGateAviary(BaseRLAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs = None,
                 initial_rpys = None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 120,
                 ctrl_freq: int = 30,
                 gui = False,
                 record = False,
                 obs: ObservationType = ObservationType.RGB,
                 act: ActionType = ActionType.VEL
                 ):
        self.EPISODE_LEN_SEC = 15
        self.GATES = self._defineGates()

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         img_fps=30)

        self.IMG_RES = np.array([640, 360])
        self.rgb = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        self.dep = np.ones((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]))
        self.seg = np.zeros((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0]))

        self.SPEED_LIMIT = 3.0

        self.next_gate_idx = 0
        self.prev_pos = None

    ################################################################################

    def _defineGates(self):
        positions = [
            (3,  0,    1.0),
            (6,  1.5,  1.5),
            (9,  0,    2.0),
            (12, -1.5,  2.5),
            (15,  0,    2.0),
            (18,  0,    1.0),
        ]
        yaws = [0, 15, -15, 15, -15, 0]

        gates = []
        for (x, y, z), yaw_deg in zip(positions, yaws):
            yaw_rad = np.deg2rad(yaw_deg)
            normal = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
            normal = normal / np.linalg.norm(normal)
            gates.append(dict(pos=np.array([x, y, z], dtype=np.float64),
                              yaw=yaw_deg,
                              normal=normal,
                              bar_ids=None))
        return gates

    ################################################################################

    def _buildGate(self, pos, yaw_deg):
        yaw_rad = np.deg2rad(yaw_deg)
        quat = p.getQuaternionFromEuler([0, 0, yaw_rad])
        color = [0.5, 0.5, 0.5, 1.0]

        bars = [
            dict(halfExtents=[0.13, 1.35, 0.3],  offset=[0,  0,  1.05]),
            dict(halfExtents=[0.13, 1.35, 0.3],  offset=[0,  0,  -1.05]),
            dict(halfExtents=[0.13, 0.3,  0.75], offset=[0, -1.05, 0]),
            dict(halfExtents=[0.13, 0.3,  0.75], offset=[0,  1.05, 0]),
        ]

        rot = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                        [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
                        [0,                0,               1]])

        bar_ids = []
        for bar in bars:
            world_offset = rot @ np.array(bar["offset"])
            world_pos = np.array(pos) + world_offset

            vs = p.createVisualShape(p.GEOM_BOX,
                                     halfExtents=bar["halfExtents"],
                                     rgbaColor=color,
                                     physicsClientId=self.CLIENT)
            cs = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=bar["halfExtents"],
                                        physicsClientId=self.CLIENT)
            body_id = p.createMultiBody(baseMass=0,
                                        baseVisualShapeIndex=vs,
                                        baseCollisionShapeIndex=cs,
                                        basePosition=world_pos,
                                        baseOrientation=quat,
                                        physicsClientId=self.CLIENT)
            bar_ids.append(body_id)

        return bar_ids

    ################################################################################

    def _getDroneImages(self, nth_drone, segmentation=False):
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])

        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT)
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=90.0,
            aspect=640.0 / 360.0,
            nearVal=self.L,
            farVal=1000.0)

        seg_flag = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=seg_flag,
            physicsClientId=self.CLIENT)

        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _observationSpace(self):
        return spaces.Box(low=0.0, high=1.0, shape=(1, 360, 640), dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        rgb, _, _ = self._getDroneImages(0, segmentation=False)
        gray = (0.299 * rgb[:, :, 0].astype(np.float32) +
                0.587 * rgb[:, :, 1].astype(np.float32) +
                0.114 * rgb[:, :, 2].astype(np.float32)) / 255.0
        return gray.astype(np.float32)[np.newaxis, :, :]

    ################################################################################

    def _computeReward(self):
        reward = 0.0
        curr_pos = self.pos[0].copy()
        prev_pos = self.prev_pos

        for i, gate in enumerate(self.GATES):
            gate_center = gate["pos"]
            gate_normal = gate["normal"]

            prev_side = np.dot(prev_pos - gate_center, gate_normal)
            curr_side = np.dot(curr_pos - gate_center, gate_normal)

            if prev_side < 0.0 and curr_side > 0.0:
                t = prev_side / (prev_side - curr_side)
                intersect = prev_pos + t * (curr_pos - prev_pos)
                proj = intersect - gate_center
                proj_plane = proj - np.dot(proj, gate_normal) * gate_normal

                world_up = np.array([0.0, 0.0, 1.0])
                gate_up = world_up - np.dot(world_up, gate_normal) * gate_normal
                gate_up = gate_up / (np.linalg.norm(gate_up) + 1e-8)
                gate_right = np.cross(gate_normal, gate_up)

                y_local = abs(np.dot(proj_plane, gate_right))
                z_local = abs(np.dot(proj_plane, gate_up))

                if y_local <= 0.75 and z_local <= 0.75:
                    if i == self.next_gate_idx:
                        reward += 200.0
                        self.next_gate_idx += 1
                    else:
                        reward -= 300.0

        if self.next_gate_idx < len(self.GATES):
            target = self.GATES[self.next_gate_idx]["pos"]
            curr_dist = np.linalg.norm(target - curr_pos)
            prev_dist = np.linalg.norm(target - prev_pos)
            reward += 5.0 * (prev_dist - curr_dist)

        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                       physicsClientId=self.CLIENT)
        for c in contacts:
            body_b = c[2]
            if body_b in self.ALL_BAR_IDS:
                reward -= 100.0
                break

        if self.next_gate_idx == len(self.GATES):
            reward += 500.0

        reward -= 0.5

        self.prev_pos = curr_pos.copy()
        return float(reward)

    ################################################################################

    def _computeTerminated(self):
        return self.next_gate_idx == len(self.GATES)

    ################################################################################

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[0]) > 25.0 or abs(state[1]) > 10.0 or state[2] > 5.0 or state[2] < 0.0:
            return True
        if abs(state[7]) > 0.6 or abs(state[8]) > 0.6:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        return {"next_gate": self.next_gate_idx, "total_gates": len(self.GATES)}

    ################################################################################

    def _addObstacles(self):
        self.ALL_BAR_IDS = set()
        for gate in self.GATES:
            gate["bar_ids"] = self._buildGate(gate["pos"], gate["yaw"])
            self.ALL_BAR_IDS.update(gate["bar_ids"])

    ################################################################################

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.next_gate_idx = 0
        self.prev_pos = self.pos[0].copy()
        return obs, info
