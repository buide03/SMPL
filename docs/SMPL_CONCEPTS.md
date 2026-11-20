# SMPL concepts

SMPL is a parametric human body model with a fixed template topology. Vertices store XYZ locations. Each face lists three zero-based vertex indices and its order defines triangle winding. A standard SMPL mesh has 6,890 vertices and 13,776 faces.

Shape β is a vector of coefficients for a learned deformation basis. The selected coefficients jointly change the template surface and affect joint locations. Individual coefficients generally change several aspects of shape together.

Pose θ is the set of body joint rotations and the root orientation. The 23 body rotations are local to their respective parents. The canonical mesh uses neutral joint rotations at the selected shape. The posed mesh uses pose-dependent corrections and linear blend skinning, which transforms each vertex by a weighted combination of joint transforms. The weights and joint hierarchy come from the SMPL model asset.

Camera placement controls how the posed mesh is seen. Camera coordinates are measured relative to a camera frame. World coordinates use a scene-wide reference frame. Calibrated extrinsics relate those frames, while HMR2 crop camera parameters describe the model's predicted placement for projection in a crop. See [coordinate systems](COORDINATE_SYSTEMS.md) and the [SMPL model project](https://smpl.is.tue.mpg.de/).
