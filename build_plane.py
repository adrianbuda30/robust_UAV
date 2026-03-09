import numpy as np
import casadi as ca


# Helpers: CasADi <-> NumPy

def to_num(x):

    """Convert CasADi (SX/MX/DM) to float/ndarray, leave NumPy/float as is."""
    if isinstance(x, (ca.SX, ca.MX, ca.DM)):
        a = x.full()
        return float(a) if a.size == 1 else np.asarray(a).squeeze()
    if np.isscalar(x):
        return float(x)
    return np.asarray(x, dtype=float)

def to_vec3(x, y, z):
    """3 scalars (possibly CasADi) -> 1D float NumPy array of length 3."""
    return np.array([to_num(x), to_num(y), to_num(z)], dtype=float)


# Define class for plane geometry

class plane:
    def __init__(self):
        self.segment = {}
        self.mass_properties = {}

    def add_segment(self, name, profile):
        if not isinstance(profile, dict):
            raise TypeError("Profile must be a dictionary.")
        self.segment[name] = profile

    def add_mass_properties(self, profile):
        if not isinstance(profile, dict):
            raise TypeError("Profile must be a dictionary.")
        self.mass_properties = profile

    def get_mass_properties(self, key=None):
        if key is None or key not in self.mass_properties:
            return self.mass_properties
        return self.mass_properties[key]

    def get_segment(self, name):
        return self.segment.get(name, None)

    def update_segment(self, name, updates):
        if name in self.segment:
            self.segment[name].update(updates)
        else:
            raise KeyError(f"Segment '{name}' not found.")

    def __repr__(self):
        return f"Plane with segments: {list(self.segment.keys())}, mass_properties: {self.mass_properties}"

if __name__ == "__main__":
    main()

