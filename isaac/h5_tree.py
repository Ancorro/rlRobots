"""Print an HDF5 dataset's layout: top-level attrs, demo count, and one demo's datasets with shapes.

  python h5_tree.py file.hdf5 [demo]
"""
import sys

import h5py

f = h5py.File(sys.argv[1], "r")
print("attrs:", {k: str(v)[:300] for k, v in f.attrs.items()})
d = f["data"]
print("data attrs:", {k: str(v)[:600] for k, v in d.attrs.items()})
names = list(d.keys())
print(len(names), "demos:", names[:5])
e = d[sys.argv[2] if len(sys.argv) > 2 else names[0]]
print("demo attrs:", dict(e.attrs))
e.visititems(lambda n, o: print(" ", n, o.shape, o.dtype) if isinstance(o, h5py.Dataset) else None)
