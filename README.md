  # ERASOR2 mapgen fails on frames larger than 500k points

  ## Symptom

  `mapgen` aborts after loading poses:

  ```text
  Total 20 poses are loaded
  terminate called after throwing an instance of 'std::invalid_argument'
  what(): Something's wrong! The numbers of points are not matched to each other

  ## Root Cause

  The ERASOR2 SemanticKITTILoader::loadCloud() implementation uses a fixed-size float buffer:

  std::vector<float> buffer(2000000);

  SemanticKITTI .bin files store four float32 values per point x, y, z, intensity, so this loader can read at most 500,000
  points per frame.

  When a frame contains more than 500,000 points, the loaded cloud is truncated, but the corresponding .label file still
  contains labels for the full frame. The point/label size check then fails before ERASOR2 reaches its configured voxel
  downsampling step.

  ## Notes

  dataloader.voxel_size: 0.2 is applied after loading. It does not prevent this failure because the crash happens while
  checking the raw loaded cloud against the raw label file.
