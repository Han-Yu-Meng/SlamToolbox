# ERASOR2 mapgen fails on frames larger than 500k points

## Symptom

`mapgen` aborts after loading poses:

```text
Total 20 poses are loaded
terminate called after throwing an instance of 'std::invalid_argument'
what(): Something's wrong! The numbers of points are not matched to each other
```

## Root Cause

The ERASOR2 `SemanticKITTILoader::loadCloud()` implementation uses a fixed-size
float buffer:

```cpp
std::vector<float> buffer(2000000);
```

SemanticKITTI `.bin` files store four `float32` values per point
`x, y, z, intensity`, so this loader can read at most 500,000 points per frame.

When a frame contains more than 500,000 points, the loaded cloud is truncated,
but the corresponding `.label` file still contains labels for the full frame.
The point/label size check then fails before ERASOR2 reaches its configured
voxel downsampling step.

## Example

The Airport_1F dataset has 20 poses and 20 scan/label files, but many frames
contain more than 500,000 points:

```text
000000 bin_pts=932577 labels=932577
000010 bin_pts=955533 labels=955533
000019 bin_pts=466632 labels=466632
```

`000019` is below the loader limit, but most earlier frames exceed it.

## Notes

`dataloader.voxel_size: 0.2` is applied after loading. It does not prevent this
failure because the crash happens while checking the raw loaded cloud against
the raw label file.

Possible fixes are:

- Read the `.bin` file dynamically based on file size in ERASOR2.
- Preprocess generated KITTI frames so each `.bin` and matching `.label` stay
  below the loader limit.
