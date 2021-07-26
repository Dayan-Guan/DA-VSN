import pathlib

project_root = pathlib.Path(__file__).resolve().parents[2]
project_root_ADVENT = pathlib.Path(__file__).resolve().parents[3] / 'ADVENT'

__all__ = ['project_root', 'project_root_ADVENT']
