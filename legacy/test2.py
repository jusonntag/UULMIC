from preprocessing import Preprocessor
from pathlib import Path

Preprocessor(
    data_dir =  Path.cwd() / 'data' / 'raw'
)