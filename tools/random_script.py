import shutil

source_dir = "/scratch/tshu2/datasets/vision/spoc/unit/obj_nav_type_house_037777_episode_0"
base_name = '/scratch/tshu2/datasets/vision/spoc/unit/obj_nav_type_house_037777_episode_'

for i in range(1, 41):
    dest_dir = f"{base_name}{i}"
    shutil.copytree(source_dir, dest_dir)
    print(f"Created: {dest_dir}")
