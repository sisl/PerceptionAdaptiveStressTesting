from tqdm import tqdm
import pickle
from pathlib import Path
from nuscenes import NuScenes
from sequential_perception.datasets import NuScenesScene


if __name__ == '__main__':
    dataroot = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval'
    version = 'v1.0-trainval'
    output_root = '/mnt/hdd/data/sets/nuscenes/v1.0-trainval/exported_scenes'
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    for i in tqdm(range(len(nusc.scene))):
        scene_record = nusc.scene[i]
        nusc_scene = NuScenesScene(dataroot, nusc, scene_record['token'])
        fname = scene_record['token'] + '.pkl'
        with open(output_path / fname, 'wb') as f:
            pickle.dump(nusc_scene, f)