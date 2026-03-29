"""Generate test meshes: subdivided tetrahedron (sphere) and ellipsoid."""
import math

def normalize(v):
    l = math.sqrt(sum(x*x for x in v))
    return tuple(x/l for x in v) if l > 1e-15 else v

def scale(v, sx, sy, sz):
    return (v[0]*sx, v[1]*sy, v[2]*sz)

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2)

def subdivide_sphere(verts, faces, n_subdivisions):
    """Loop subdivision projected onto unit sphere."""
    for _ in range(n_subdivisions):
        edge_midpoints = {}
        new_verts = list(verts)
        new_faces = []
        
        for f in faces:
            mids = []
            for i in range(3):
                a, b = f[i], f[(i+1)%3]
                key = (min(a,b), max(a,b))
                if key not in edge_midpoints:
                    mp = midpoint(verts[a], verts[b])
                    mp = normalize(mp)  # project to sphere
                    edge_midpoints[key] = len(new_verts)
                    new_verts.append(mp)
                mids.append(edge_midpoints[key])
            
            # 4 sub-triangles
            a, b, c = f
            m_ab, m_bc, m_ca = mids
            new_faces.append((a, m_ab, m_ca))
            new_faces.append((m_ab, b, m_bc))
            new_faces.append((m_ca, m_bc, c))
            new_faces.append((m_ab, m_bc, m_ca))
        
        verts = new_verts
        faces = new_faces
    return verts, faces

def write_obj(filename, verts, faces):
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Wrote {filename}: {len(verts)} vertices, {len(faces)} faces")

# Regular tetrahedron vertices
tet_verts = [
    normalize((1, 1, 1)),
    normalize((1, -1, -1)),
    normalize((-1, 1, -1)),
    normalize((-1, -1, 1)),
]
tet_faces = [
    (0, 1, 2),
    (0, 3, 1),
    (0, 2, 3),
    (1, 3, 2),
]

# Subdivided sphere (5 levels → 4*4^5 = 4096 faces)
sphere_v, sphere_f = subdivide_sphere(tet_verts, tet_faces, 5)
write_obj("sphere.obj", sphere_v, sphere_f)

# Ellipsoid: stretch the sphere by (2, 1, 1) → kappa differs along axes
ellipsoid_v = [scale(v, 2.0, 1.0, 1.0) for v in sphere_v]
write_obj("ellipsoid.obj", ellipsoid_v, sphere_f)

# Torus: explicit parameterization (clear curvature structure)
R, r = 2.0, 0.7  # major and minor radii
nu, nv_t = 80, 40  # resolution
torus_v = []
torus_f = []
for i in range(nu):
    for j in range(nv_t):
        theta = 2*math.pi * i / nu
        phi = 2*math.pi * j / nv_t
        x = (R + r*math.cos(phi)) * math.cos(theta)
        y = (R + r*math.cos(phi)) * math.sin(theta)
        z = r * math.sin(phi)
        torus_v.append((x, y, z))

for i in range(nu):
    for j in range(nv_t):
        a = i * nv_t + j
        b = i * nv_t + (j+1) % nv_t
        c = ((i+1) % nu) * nv_t + (j+1) % nv_t
        d = ((i+1) % nu) * nv_t + j
        torus_f.append((a, b, c))
        torus_f.append((a, c, d))

write_obj("torus.obj", torus_v, torus_f)
