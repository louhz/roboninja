# random object initial pose in genesis with different x, y and rotation on the ground( a certain size of the table)



# you can assume you have the digital asset  of a small table, the strawberry must on the table, and the table can also move around


# the table is from[-0.5,-0.5] to [0.5,0.5] meters



import numpy as np
import genesis as gs

def sample_episode_pose(
    seed: int,
    table_world_min=(-0.5, -0.5),
    table_world_max=( 0.5,  0.5),
    table_top_half_extents=(0.5, 0.5),   # tabletop corners: [-0.5,-0.5]..[0.5,0.5]
    strawberry_margin=0.03,              # keep away from edges
    table_z=0.0,                         # table base z
    table_top_z=0.75,                    # tabletop height (set this to your asset)
    strawberry_z_offset=0.02,            # small lift to avoid penetration
):
    rng = np.random.default_rng(seed)

    # Table pose (world)
    tx = rng.uniform(table_world_min[0], table_world_max[0])
    ty = rng.uniform(table_world_min[1], table_world_max[1])
    table_yaw_deg = rng.uniform(-180.0, 180.0)

    # Strawberry pose (sample on tabletop in table-local coords)
    hx, hy = table_top_half_extents
    lx = rng.uniform(-hx + strawberry_margin, hx - strawberry_margin)
    ly = rng.uniform(-hy + strawberry_margin, hy - strawberry_margin)

    # Rotate local offset by table yaw -> world offset
    yaw = np.deg2rad(table_yaw_deg)
    ox = lx * np.cos(yaw) - ly * np.sin(yaw)
    oy = lx * np.sin(yaw) + ly * np.cos(yaw)

    # Strawberry world pose
    sx = tx + ox
    sy = ty + oy
    sz = table_z + table_top_z + strawberry_z_offset

    strawberry_yaw_deg = rng.uniform(-180.0, 180.0)

    return (tx, ty, table_z), (0.0, 0.0, table_yaw_deg), (sx, sy, sz), (0.0, 0.0, strawberry_yaw_deg)


def build_scene_one_episode(seed: int, table_mesh: str, strawberry_mesh: str):
    gs.init(seed=seed, backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -10.0)),
        show_viewer=True,
    )
    plane = scene.add_entity(gs.morphs.Plane())

    table_pos, table_euler, straw_pos, straw_euler = sample_episode_pose(seed)

    table = scene.add_entity(
        gs.morphs.Mesh(
            file=table_mesh,
            pos=table_pos,
            euler=table_euler,   # degrees, extrinsic x-y-z :contentReference[oaicite:3]{index=3}
            fixed=True,          # keep table static (set False if you truly want it dynamic) :contentReference[oaicite:4]{index=4}
        )
    )
    strawberry = scene.add_entity(
        gs.morphs.Mesh(
            file=strawberry_mesh,
            pos=straw_pos,
            euler=straw_euler,
            fixed=False,
        )
    )

    scene.build()
    return scene
