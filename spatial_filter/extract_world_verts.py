"""
extract_world_verts.py — Blender headless script.

Exports world-space vertex coordinates and face centroids for all mesh objects
(or a specified subset) in the Virtual Worm blend file.

Run:
  /Applications/Blender.app/Contents/MacOS/blender \
      --background Virtual_Worm_February_2012.blend \
      --python extract_world_verts.py

Outputs four files into data/nml_vertices/:
  blend_world_verts.csv      — one row per vertex: mesh, vid, x, y, z
  blend_world_face_cents.csv — one row per face centroid: mesh, x, y, z
  blend_world_faces.csv      — one row per face: mesh, fid, v0, v1, v2[, v3]
                               vertex indices into blend_world_verts (vid column)
                               used for dense bilinear/barycentric surface sampling
  blend_world_centroids.csv  — one row per object: mesh, cx, cy, cz, n_verts
"""

import bpy
import csv
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "nml_vertices")
os.makedirs(OUT_DIR, exist_ok=True)

VERTS_CSV      = os.path.join(OUT_DIR, "blend_world_verts.csv")
FACE_CENTS_CSV = os.path.join(OUT_DIR, "blend_world_face_cents.csv")
FACES_CSV      = os.path.join(OUT_DIR, "blend_world_faces.csv")
CENTROIDS_CSV  = os.path.join(OUT_DIR, "blend_world_centroids.csv")

# Objects to export full vertex clouds (intestine + calibration neurons).
# Empty set = export everything.
EXPORT_FULL = {
    # Intestine cells (world-space meshes)
    "int1DL", "int1DR", "int1VL", "int1VR",
    "int2D",  "int2V",
    "int3D",  "int3V",
    "int4D",  "int4V",
    "int5L",  "int5R",
    "int6L",  "int6R",
    "int7L",  "int7R",
    "int8L",  "int8R",
    "int9L",  "int9R",
    # Body boundary
    "Cuticle",
    "Seam_Cells_Left", "Seam_Cells_Right",
    # Anatomical landmarks
    "Pharyn-Intest-Valve",
    # Calibration neurons (soma meshes, world-space)
    "CANL", "CANR",
    "RICL", "RICR",
    "ADEL", "ADER",
    "ADFL", "ADFR",
    "ASIL", "ASIR",
    "AVM",
    "HSNL", "HSNR",
    "NSML", "NSMR",
    "PDEL", "PDER",
    "RIML", "RIMR",
    "RIH",  "RIR",
    "CEPDL","CEPDR","CEPVL","CEPVR",
    "VC4",  "VC5",
}

def world_verts(obj):
    """Return list of (vid, x, y, z) for every vertex in world space."""
    wm = obj.matrix_world
    return [(v.index, *((wm @ v.co).to_tuple())) for v in obj.data.vertices]

def face_centroids(obj):
    """Return list of (x, y, z) for the world-space centroid of every face."""
    wm   = obj.matrix_world
    mesh = obj.data
    cents = []
    for poly in mesh.polygons:
        vcos = [(wm @ mesh.vertices[vi].co) for vi in poly.vertices]
        n    = len(vcos)
        cx   = sum(v.x for v in vcos) / n
        cy   = sum(v.y for v in vcos) / n
        cz   = sum(v.z for v in vcos) / n
        cents.append((cx, cy, cz))
    return cents

def face_vertex_lists(obj):
    """
    Return list of (fid, v0, v1, v2[, v3]) — per-face vertex indices.
    Indices match the vid column in blend_world_verts.csv.
    Triangles emit 3 indices; quads emit 4.
    """
    rows = []
    for fid, poly in enumerate(obj.data.polygons):
        vids = list(poly.vertices)   # 3 or 4 vertex indices
        rows.append((fid, *vids))
    return rows

print("Scanning scene objects...", flush=True)

verts_rows      = []
face_cent_rows  = []
face_rows       = []
centroid_rows   = []

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    name = obj.name
    wv   = world_verts(obj)
    n    = len(wv)
    if n == 0:
        continue

    cx = sum(v[1] for v in wv) / n
    cy = sum(v[2] for v in wv) / n
    cz = sum(v[3] for v in wv) / n
    centroid_rows.append((name, cx, cy, cz, n))

    if (not EXPORT_FULL) or (name in EXPORT_FULL):
        for vid, vx, vy, vz in wv:
            verts_rows.append((name, vid, vx, vy, vz))
        for fx, fy, fz in face_centroids(obj):
            face_cent_rows.append((name, fx, fy, fz))
        for face_entry in face_vertex_lists(obj):
            face_rows.append((name, *face_entry))

    print(f"  {name}: {n} verts, centroid=({cx:.3f}, {cy:.3f}, {cz:.3f})", flush=True)

# Write full vertex cloud
with open(VERTS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "vid", "x", "y", "z"])
    w.writerows(verts_rows)

# Write face centroids
with open(FACE_CENTS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "x", "y", "z"])
    w.writerows(face_cent_rows)

# Write face topology (variable columns: tri=5, quad=6)
with open(FACES_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "fid", "v0", "v1", "v2", "v3"])
    for row in face_rows:
        # pad triangles to 4 vids with empty string
        padded = list(row) + [""] * (6 - len(row))
        w.writerow(padded[:6])

# Write per-object centroid summary (all objects)
with open(CENTROIDS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "cx", "cy", "cz", "n_verts"])
    w.writerows(centroid_rows)

print(f"\nWrote {len(verts_rows):,} vertex rows       → {VERTS_CSV}", flush=True)
print(f"Wrote {len(face_cent_rows):,} face centroid rows → {FACE_CENTS_CSV}", flush=True)
print(f"Wrote {len(face_rows):,} face topology rows  → {FACES_CSV}", flush=True)
print(f"Wrote {len(centroid_rows)} object centroids    → {CENTROIDS_CSV}", flush=True)
