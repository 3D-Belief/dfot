import shutil
import os
from pathlib import Path

source_dir = Path(os.environ.get("SPOC_SOURCE_DIR", "data/spoc/unit/obj_nav_type_house_037777_episode_0"))
output_base_dir = Path(os.environ.get("SPOC_OUTPUT_BASE_DIR", "data/spoc/unit"))
episode_prefix = os.environ.get("SPOC_EPISODE_PREFIX", "obj_nav_type_house_037777_episode_")

for i in range(1, 41):
    dest_dir = output_base_dir / f"{episode_prefix}{i}"
    shutil.copytree(source_dir, dest_dir)
    print(f"Created: {dest_dir}")
