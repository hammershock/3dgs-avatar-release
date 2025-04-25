from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .mydataset import MyDataset

def load_dataset(cfg, split='train'):
    if cfg.name == 'zjumocap':
        return ZJUMoCapDataset(cfg, split=split)
    elif cfg.name == 'people_snapshot':
        return PeopleSnapshotDataset(cfg, split=split)
    elif cfg.name == 'mydataset':
        return MyDataset(cfg, split=split)
    raise ValueError("Unknown dataset")