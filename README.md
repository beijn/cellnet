# Cell Counting Collaboration
at labs in Tartu, Göttingen and Cambridge

Localizing Cells in Phase-Contrast Microscopy Images using Sparse and Noisy Center-Point Annotations

Based on [Benjamin Eckhardt's Bachelor's Thesis in Computer Science](https://github.com/beijn/bachelor-thesis).

## Installation 

`pip install git+https://github.com/beijn/cellnet`

## Usage

```python
from cellnet import init_model, count

model = init_model(<version>)

counts = count(<[image_file_descriptors]>, model)
```

`<version>`
- `'latest'`: will download the latest model 
- `None`: will use what ever is already in the cache or default to `'latest'`
- any other string will download that version from [releases](https://github.com/beijn/cellnet/releases) 
