export const defaultParams = {
  thickness_ratio: 0.12,
  camber: 0.04,
  camber_position: 0.4,
  leading_edge_radius: 0.02,
  trailing_edge_angle: 15,
  aspect_ratio: 8,
  taper_ratio: 0.5,
  sweep_angle: 20,
  twist_angle: 0,
  dihedral_angle: 3,
  mach: 0.5,
  reynolds: 1e6,
  alpha: 5,
  beta: 0,
  altitude: 0,
};

export const sweepOptions = [
  { key: "alpha", label: "Angle of attack", from: -5, to: 15 },
  { key: "mach", label: "Mach number", from: 0.1, to: 0.9 },
  { key: "reynolds", label: "Reynolds", from: 1e5, to: 1e7 },
  { key: "thickness_ratio", label: "Thickness", from: 0.06, to: 0.24 },
  { key: "sweep_angle", label: "Sweep", from: 0, to: 45 },
];