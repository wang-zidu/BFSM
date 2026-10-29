# Local geometry PCA construction

Construct a **new institutional FSMM and TMM** from your own already aligned, corresponding face/skull meshes. This module does not replace the supplied model files and does not load the original cohort. Keep inputs, the manifest and outputs local under your institution's data rules.

The calculation follows the original construction scripts:

```
FSMM sample = flatten(concatenate(face_vertices, skull_vertices))
TMM sample  = flatten(skull_vertices[pairs[:, 1]] - face_vertices[pairs[:, 0]])
shapeMU     = PCA.mean_[None, :]
shapeBase   = PCA.components_.T * sqrt(PCA.explained_variance_)[None, :]
geometry    = shapeMU + shapeBase @ coefficients
```

FSMM face and skull use one shared coefficient vector. TMM is independently fitted to paired displacement vectors, with the **skull minus face** sign. PCA uses full SVD and sample variance with denominator `N - 1`. A coefficient of `+3` means three standard deviations along that mode; the basis already contains the standard-deviation scale.

## Inputs

1. Triangular OBJ meshes or `N×3` NPY vertex arrays for each paired sample. OBJ topology must exactly match the fixed template, including vertex and triangle order. Materials, texture and per-vertex colors are ignored. NPY has no connectivity to check, so the canonical vertex-order contract must be established upstream.
2. A template NPZ with `face_tri`, `skull_tri`, scalar `face_vertex_count`, scalar `skull_vertex_count`, and `face_to_skull_ind_new` (`P×2` integer face/skull indices). Optional `region_*` arrays are fixed face vertex indices. Optional `face_buf`/`point_buf` normal buffers are validated and copied; if absent they are generated deterministically from `face_tri`. `face_buf` equals `face_tri`. Each `point_buf` row lists all incident face indices once; padding is the face count `F`, selecting the zero normal appended by the original loader. Supplied adjacency order is preserved. The construction routine copies only these geometry/template fields, never weights, textures or sample records. The supplied `assets/models/fsmm_200.npz` can be used as this template if your data uses exactly its topology and pairing.
3. A private TOML manifest (legacy JSON is also accepted). Paths are absolute or relative to the manifest. For example:

```toml
[contract]
same_topology_and_vertex_order = true
paired_common_coordinate_frame = true
all_samples_prealigned = true
alignment = "external_preprocessing_no_alignment_in_this_cli"
axis_convention = "x_lateral_y_superior_z_anterior"
units = "mm"

[[samples]]
face = "case_a/topology_face.obj"
skull = "case_a/topology_skull.obj"

[[samples]]
face = "case_b/topology_face.obj"
skull = "case_b/topology_skull.obj"
```

The two-row example illustrates the manifest format; the component count is chosen for your own cohort. All cases must be in the same units, template frame and alignment convention; each face and skull must remain mutually aligned. This command deliberately performs no hidden registration, independent recentering, scale normalization or mirroring. It can verify shapes, indices and finite values, but cannot prove semantic vertex correspondence or a truthful coordinate contract from numerical arrays alone. Complete the registration pipeline first.

## Run

From the repository root, using the configured environment with NumPy and scikit-learn, set `--components` to your chosen count (50 below is an example):

```bash
python -m bfsm_topology.construction.build \
  --manifest /your/private/aligned_pairs.toml \
  --template assets/models/fsmm_200.npz \
  --components 50 \
  --output /your/private/new_geometry_models
```

Use `--dtype float64` if double-precision fitting is required; the default is float32. Full SVD and the two PCA input matrices require memory proportional to cohort size times geometry dimensions. This CLI targets the original small-cohort regime, rather than implementing an undocumented incremental approximation.

For `N` paired samples, choose `K` components with `1 <= K <= N - 1` and `K` no greater than the numerical rank of either centered FSMM/TMM PCA input matrix. The supplied models have 200 components; institutional models are not fixed to that count. Using `fsmm_200.npz` as the topology/pairing template does not fix the new model's component count. A mirrored mesh is not automatically an additional independent identity. No mirroring occurs here; if used upstream, reflection and left/right reindexing must already preserve all correspondence. Rank deficiency causes an error, not zero padding or invented basis columns. PCA signs can differ across library versions while describing the same subspace; the construction function returns the versions and rank tolerance to its caller.

Outputs are `fsmm.npz` and `tmm.npz`. They contain the geometry means, scaled bases, topology, pairing indices and region fields. Texture models are not fitted by this construction routine. Output models are not automatically copied over supplied models. Existing output artifacts are never silently overwritten.

Model construction accepts other consistent topologies as a general mathematical model, but this does **not** make the original image/TMM pipeline topology-independent. Its face/skull decoder and associated BFM inputs expect the standard 35709-face/23207-skull ordering and compatible triangles, pair indices and regions. The generated normal buffers close the geometry-only loader requirements for that standard topology; an arbitrary-topology PCA model still requires its own compatible downstream implementation. Texture reconstruction remains outside this geometry-only construction routine.

## Reproducible non-patient check

```bash
python -m bfsm_topology.construction.synthetic_smoke --output /your/local/new_synthetic_check
```

This generates six labelled synthetic cube pairs and invokes the actual construction CLI for float32 and float64. It independently evaluates the original scikit-learn equations, compares means and scaled bases, verifies the `skull - face` construction, checks generated/retained normal buffers, and rejects insufficient sample count, zero rank, mismatched OBJ topology or incomplete adjacency. No patient data, model weights, PyTorch or GPU are needed. This checks the PCA calculation and input handling.
