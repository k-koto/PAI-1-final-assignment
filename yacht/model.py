"""470級を1/6に縮小した簡易ヨットとSO-101アームのMuJoCoモデルを組み立てる。

座標系: x=船首方向, y=左舷(風上), z=上。
船体は台座にロール軸(x軸)のヒンジでつながるだけの土台で、前進はしない。
"""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SO101_XML = ROOT / "assets" / "so101" / "so101.xml"

# 470級(全長4.70m, 全幅1.69m, マスト6.78m)の1/6スケール
SCALE = 1 / 6
LENGTH = 4.70 * SCALE
BEAM = 1.69 * SCALE
DEPTH = 0.09
MAST_X = 0.08
MAST_H = 6.78 * SCALE
BOOM_Z = 0.13

BOAT_MASS = 2.0
BOAT_COM_Z = -0.25  # 船体の重心を下げ(センターボード相当)、アームの頭の重さによる倒れやすさを打ち消す
BOAT_COM_Y = -0.055  # 畳んだアームの重さと釣り合うよう、重心を少し風下へ
BOAT_IXX = 1.0
WEIGHT_MASS = 0.35  # アーム先端の重り(乗り手の体重の代わり)
ARM_MOUNT = (-0.10, BEAM / 2 - 0.035, 0.0)
SAIL_CE_HEIGHT = 0.42  # 帆の受風中心の高さ(風の矢印の表示用)


def _hull_mesh(n_st=24, n_sec=14):
    """船首が尖り、船尾がトランサムで切れた船体の閉じたメッシュを作る。"""
    us = np.linspace(0.0, 1.0, n_st)  # 0=船尾, 1=船首
    rings = []
    for u in us:
        if u < 0.45:
            hb = BEAM / 2 * (0.82 + 0.18 * np.sin(u / 0.45 * np.pi / 2))
        else:
            hb = BEAM / 2 * np.sqrt(max(1 - ((u - 0.45) / 0.55) ** 2, 0.0))
        hb = max(hb, 0.004)
        d = DEPTH * (0.55 + 0.45 * np.sin(min(u / 0.35, 1) * np.pi / 2))
        if u > 0.7:
            d *= 1 - 0.75 * ((u - 0.7) / 0.3) ** 2
        x = -LENGTH / 2 + u * LENGTH
        ring = []
        for phi in np.linspace(-np.pi / 2, np.pi / 2, n_sec):
            ring.append((x, hb * np.sin(phi), -d * np.cos(phi) ** 0.7))
        rings.append(ring)
    verts = [v for ring in rings for v in ring]
    faces = []
    for i in range(n_st - 1):
        for j in range(n_sec - 1):
            a, b = i * n_sec + j, i * n_sec + j + 1
            c, e = (i + 1) * n_sec + j, (i + 1) * n_sec + j + 1
            faces += [(a, b, c), (b, e, c)]
    # デッキ(上面)
    for i in range(n_st - 1):
        a, b = i * n_sec, i * n_sec + n_sec - 1
        c, e = (i + 1) * n_sec, (i + 1) * n_sec + n_sec - 1
        faces += [(a, c, b), (b, c, e)]
    # トランサム(船尾板)
    for j in range(1, n_sec - 1):
        faces.append((0, j + 1, j))
    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def _triangle_mesh(p0, p1, p2, thickness=0.002):
    """薄い三角柱(帆)のメッシュ。"""
    p = np.array([p0, p1, p2], dtype=float)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    n = n / np.linalg.norm(n) * thickness / 2
    verts = np.vstack([p + n, p - n])
    faces = [(0, 1, 2), (3, 5, 4), (0, 3, 1), (1, 3, 4), (1, 4, 2), (2, 4, 5), (2, 5, 0), (0, 5, 3)]
    return verts, np.array(faces)


def _add_mesh(spec, name, verts, faces):
    spec.add_mesh(name=name, uservert=verts.flatten().tolist(), userface=faces.flatten().tolist())


def build_spec():
    spec = mujoco.MjSpec()
    spec.modelname = "yacht470"
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    spec.option.timestep = 0.002
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.55, 0.75, 0.95], rgb2=[0.95, 0.97, 1.0], width=512, height=512)
    spec.add_texture(name="sea", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_RANDOM,
                     random=0.02, markrgb=[0.6, 0.8, 0.9],
                     rgb1=[0.05, 0.30, 0.50], rgb2=[0.06, 0.33, 0.53], width=512, height=512)
    spec.add_material(name="sea", textures=["", "sea"], texrepeat=[12, 12], reflectance=0.25)
    spec.add_material(name="hull", rgba=[0.97, 0.97, 0.97, 1], specular=0.6, shininess=0.8)
    spec.add_material(name="sail", rgba=[0.98, 0.98, 0.94, 0.92])
    spec.add_material(name="spar", rgba=[0.75, 0.76, 0.8, 1], specular=0.8)
    spec.add_material(name="foil", rgba=[0.15, 0.15, 0.2, 1])
    spec.add_material(name="crew", rgba=[0.95, 0.35, 0.1, 1])

    world = spec.worldbody
    world.add_light(pos=[1.5, 2.0, 4.0], dir=[-0.3, -0.4, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_geom(name="sea", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.1],
                   pos=[0, 0, -0.055], material="sea")
    world.add_camera(name="stern", pos=[-2.6, 0.0, 0.55], xyaxes=[0, -1, 0, 0.15, 0, 1])
    world.add_camera(name="quarter", pos=[-1.9, 1.6, 0.9], xyaxes=[-0.64, -0.77, 0, 0.25, -0.2, 0.95])

    hull_v, hull_f = _hull_mesh()
    _add_mesh(spec, "hull", hull_v, hull_f)
    main_v, main_f = _triangle_mesh([0, 0, BOOM_Z + 0.01], [0, 0, MAST_H - 0.02], [-0.40, 0, BOOM_Z + 0.01])
    _add_mesh(spec, "mainsail", main_v, main_f)
    jib_v, jib_f = _triangle_mesh([0.30, 0, 0.03], [0, 0, 0.78], [-0.06, 0, 0.08])
    _add_mesh(spec, "jib", jib_v, jib_f)

    boat = world.add_body(name="boat", pos=[0, 0, 0])
    boat.add_joint(name="heel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.05)
    boat.explicitinertial = True
    boat.mass = BOAT_MASS
    boat.ipos = [0, BOAT_COM_Y, BOAT_COM_Z]
    boat.inertia = [BOAT_IXX, BOAT_IXX, BOAT_IXX]
    boat.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hull", material="hull")
    # センターボードとラダー
    boat.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.003, 0.09], pos=[0.05, 0, -0.16], material="foil")
    boat.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.003, 0.06], pos=[-LENGTH / 2 - 0.01, 0, -0.07],
                  material="foil")
    # マスト・ブーム・帆。帆は風下(-y)へ少し出した状態
    boat.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.007, MAST_H / 2], pos=[MAST_X, 0, MAST_H / 2],
                  material="spar")
    rig = boat.add_body(name="mainrig", pos=[MAST_X, 0, 0], euler=[0, 0, -0.25])
    rig.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.005, 0.21], pos=[-0.21, 0, BOOM_Z],
                 euler=[0, np.pi / 2, 0], material="spar")
    rig.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mainsail", material="sail")
    jib = boat.add_body(name="jibrig", pos=[MAST_X, 0, 0], euler=[0, 0, -0.18])
    jib.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="jib", material="sail")
    boat.add_site(name="sail_ce", pos=[MAST_X - 0.12, 0, SAIL_CE_HEIGHT])
    mount = boat.add_frame(pos=list(ARM_MOUNT), euler=[0, 0, np.pi / 2])

    arm_spec = mujoco.MjSpec.from_file(str(SO101_XML))
    # 親モデルの設定に合わせ、アタッチ時の競合警告を出さない
    for key in ("timestep", "iterations", "ls_iterations", "impratio", "integrator", "cone"):
        setattr(arm_spec.option, key, getattr(spec.option, key))
    mount.attach_body(arm_spec.body("base"), "arm/", "")
    # 重りを持って腕を水平に伸ばせるよう、サーボのトルク上限を引き上げる
    for act in spec.actuators:
        act.forcerange = [-8.0, 8.0]
    # 先端の重り(乗り手の体重の代わり)
    grip = spec.body("arm/gripper")
    w = grip.add_body(name="weight", pos=[0, 0, -0.06])
    w.explicitinertial = True
    w.mass = WEIGHT_MASS
    w.inertia = [2e-4, 2e-4, 2e-4]
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03], material="crew")
    return spec


def build_model():
    spec = build_spec()
    model = spec.compile()
    return model
