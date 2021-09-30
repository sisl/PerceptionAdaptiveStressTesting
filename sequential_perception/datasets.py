from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcdet.config import cfg, cfg_from_yaml_file


class PCDetNuScenesDataset:
    def __init__(self, nuscenes, pcdet_infos):
        self.nuscenes = nuscenes
        self.pcdet_infos = pcdet_infos
        self.token_to_idx_map = self._get_reverse_sample_map()
        return

    def _get_reverse_sample_map(self):
        reverse_map = {}
        for idx, data_dict in enumerate(self.pcdet_infos):
            sample_token = data_dict['metadata']['token']
            reverse_map[sample_token] = idx
        
        return reverse_map

    def get_data_for_sample(self, sample_token: str):
        idx = self.token_to_idx_map[sample_token]
        data_dict = self.pcdet_dataset[idx]
        return data_dict