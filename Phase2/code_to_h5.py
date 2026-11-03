import numpy as np
import re


def _loop_sort_key(loop_name: str) -> tuple:
    m = re.match(r"^loop(\d+)_(\d+)$", loop_name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (10**9, 0)


def code_to_h5(code_str):

    BOOLEAN_MAP_INV = {'New': 0, 'Joining': 1, 'Cutting': 2, 'Intersection': 3}
    EXTENT_MAP_INV = {'One-sided': 0, 'Symmetric': 1, 'Two-sided': 2}
    
    def encode_coord(val):
        return val + 128
    
    def encode_deg_360(val):
        return int(round((float(val) / 360.0) * 256.0))
    
    def encode_deg_180(val):
        return int(round(((float(val) + 180.0) / 360.0) * 256.0))
    
    def encode_scale(val):
        return int(round((float(val) / 2.0) * 255.0))

    def create_row(cmd_type):
        row = np.full(17, -1, dtype=int)
        row[0] = cmd_type
        return row

    lines = code_str.strip().split('\n')
    
    loops_data = {}

    sketches_map = {}

    final_sequence = []

    re_loop_decl = re.compile(r"(\w+)\s*=\s*Loop\(\)")
    re_line = re.compile(r"(\w+)\.Line\(endpoint=\((-?\d+),\s*(-?\d+)\)\)")
    re_arc = re.compile(r"(\w+)\.Arc\(endpoint=\((-?\d+),\s*(-?\d+)\),\s*degrees=([\d\.]+),\s*counterclockwise=(True|False)\)")
    re_circle = re.compile(r"(\w+)\.Circle\(center=\((-?\d+),\s*(-?\d+)\),\s*radius=(\d+)\)")
    
    re_add = re.compile(r"(\w+)\.add\((\w+)\)")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if "End of code" in line:
                final_sequence.append(create_row(3))
            continue

        m = re_loop_decl.search(line)
        if m:
            loop_name = m.group(1)
            loops_data[loop_name] = []
            continue

        m = re_line.search(line)
        if m:
            loop_name, x, y = m.group(1), int(m.group(2)), int(m.group(3))
            row = create_row(0)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            loops_data[loop_name].append(row)
            continue

        m = re_arc.search(line)
        if m:
            loop_name = m.group(1)
            x, y = int(m.group(2)), int(m.group(3))
            deg = float(m.group(4))
            ccw = 1 if m.group(5) == 'True' else 0
            
            row = create_row(1)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            row[3] = encode_deg_360(deg)
            row[4] = ccw
            loops_data[loop_name].append(row)
            continue


        m = re_circle.search(line)
        if m:
            loop_name = m.group(1)
            x, y = int(m.group(2)), int(m.group(3))
            r = int(m.group(4))
            
            row = create_row(2)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            row[5] = r                   
            
            loops_data[loop_name].append(row)
            continue

        m = re_add.search(line)
        if m:
            sketch_name, loop_name = m.group(1), m.group(2)
            if sketch_name not in sketches_map:
                sketches_map[sketch_name] = []
            sketches_map[sketch_name].append(loop_name)
            continue


        if line.startswith("Extrude("):

            sketch_match = re.search(r"sketch=(\w+)", line)
            if not sketch_match: continue
            sketch_name = sketch_match.group(1)
 
            if sketch_name in sketches_map:
                for loop_name in sketches_map[sketch_name]:
                    final_sequence.append(create_row(4))
                    final_sequence.extend(loops_data.get(loop_name, []))

            row = create_row(5)

            ori_m = re.search(r"orientation_degrees=\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", line)
            if ori_m:
                row[6] = encode_deg_180(ori_m.group(1))
                row[7] = encode_deg_180(ori_m.group(2))
                row[8] = encode_deg_180(ori_m.group(3))

            org_m = re.search(r"origin=\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", line)
            if org_m:
                row[9] = encode_coord(int(org_m.group(1)))
                row[10] = encode_coord(int(org_m.group(2)))
                row[11] = encode_coord(int(org_m.group(3)))
                
            scale_m = re.search(r"scale=([\d\.]+)", line)
            if scale_m:
                row[12] = encode_scale(scale_m.group(1))
                
            dist_m = re.search(r"extrude_distance=\((-?\d+),\s*(-?\d+)\)", line)
            if dist_m:
                row[13] = encode_coord(int(dist_m.group(1)))
                row[14] = encode_coord(int(dist_m.group(2)))
                
            bool_m = re.search(r"boolean_type='(\w+)'", line)
            if bool_m:
                row[15] = BOOLEAN_MAP_INV.get(bool_m.group(1), 0)
                
            extent_m = re.search(r"extent_type='([\w\-]+)'", line)
            if extent_m:
                row[16] = EXTENT_MAP_INV.get(extent_m.group(1), 0)
            
            final_sequence.append(row)

    return np.array(final_sequence)


def code_to_h5_partial(code_str):
    BOOLEAN_MAP_INV = {'New': 0, 'Joining': 1, 'Cutting': 2, 'Intersection': 3}
    EXTENT_MAP_INV = {'One-sided': 0, 'Symmetric': 1, 'Two-sided': 2}

    def encode_coord(val):
        return val + 128

    def encode_deg_360(val):
        return int(round((float(val) / 360.0) * 256.0))

    def encode_deg_180(val):
        return int(round(((float(val) + 180.0) / 360.0) * 256.0))

    def encode_scale(val):
        return int(round((float(val) / 2.0) * 255.0))

    def create_row(cmd_type):
        row = np.full(17, -1, dtype=int)
        row[0] = cmd_type
        return row

    lines = code_str.strip().split("\n")

    loops_data = {}

    sketches_map = {}

    final_sequence = []

    emitted_loops = set()

    re_loop_decl = re.compile(r"(\w+)\s*=\s*Loop\(\)")
    re_line = re.compile(r"(\w+)\.Line\(endpoint=\((-?\d+),\s*(-?\d+)\)\)")
    re_arc = re.compile(
        r"(\w+)\.Arc\(endpoint=\((-?\d+),\s*(-?\d+)\),\s*degrees=([\d\.]+),\s*counterclockwise=(True|False)\)"
    )
    re_circle = re.compile(r"(\w+)\.Circle\(center=\((-?\d+),\s*(-?\d+)\),\s*radius=(\d+)\)")

    re_add = re.compile(r"(\w+)\.add\((\w+)\)")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if "End of code" in line:
                final_sequence.append(create_row(3))
            continue

        m = re_loop_decl.search(line)
        if m:
            loop_name = m.group(1)
            loops_data[loop_name] = []
            continue

        m = re_line.search(line)
        if m:
            loop_name, x, y = m.group(1), int(m.group(2)), int(m.group(3))
            row = create_row(0)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            loops_data[loop_name].append(row)
            continue

        m = re_arc.search(line)
        if m:
            loop_name = m.group(1)
            x, y = int(m.group(2)), int(m.group(3))
            deg = float(m.group(4))
            ccw = 1 if m.group(5) == 'True' else 0

            row = create_row(1)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            row[3] = encode_deg_360(deg)
            row[4] = ccw
            loops_data[loop_name].append(row)
            continue

        m = re_circle.search(line)
        if m:
            loop_name = m.group(1)
            x, y = int(m.group(2)), int(m.group(3))
            r = int(m.group(4))

            row = create_row(2)
            row[1] = encode_coord(x)
            row[2] = encode_coord(y)
            row[5] = r

            loops_data[loop_name].append(row)
            continue

        m = re_add.search(line)
        if m:
            sketch_name, loop_name = m.group(1), m.group(2)
            if sketch_name not in sketches_map:
                sketches_map[sketch_name] = []
            sketches_map[sketch_name].append(loop_name)
            continue

        if line.startswith("Extrude("):

            sketch_match = re.search(r"sketch=(\w+)", line)
            if not sketch_match:
                continue
            sketch_name = sketch_match.group(1)

            if sketch_name in sketches_map:
                for loop_name in sketches_map[sketch_name]:
                    emitted_loops.add(loop_name)
                    final_sequence.append(create_row(4))
                    final_sequence.extend(loops_data.get(loop_name, []))

            row = create_row(5)

            ori_m = re.search(r"orientation_degrees=\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", line)
            if ori_m:
                row[6] = encode_deg_180(ori_m.group(1))
                row[7] = encode_deg_180(ori_m.group(2))
                row[8] = encode_deg_180(ori_m.group(3))

            org_m = re.search(r"origin=\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", line)
            if org_m:
                row[9] = encode_coord(int(org_m.group(1)))
                row[10] = encode_coord(int(org_m.group(2)))
                row[11] = encode_coord(int(org_m.group(3)))

            scale_m = re.search(r"scale=([\d\.]+)", line)
            if scale_m:
                row[12] = encode_scale(scale_m.group(1))

            dist_m = re.search(r"extrude_distance=\((-?\d+),\s*(-?\d+)\)", line)
            if dist_m:
                row[13] = encode_coord(int(dist_m.group(1)))
                row[14] = encode_coord(int(dist_m.group(2)))

            bool_m = re.search(r"boolean_type='(\w+)'", line)
            if bool_m:
                row[15] = BOOLEAN_MAP_INV.get(bool_m.group(1), 0)

            extent_m = re.search(r"extent_type='([\w\-]+)'", line)
            if extent_m:
                row[16] = EXTENT_MAP_INV.get(extent_m.group(1), 0)

            final_sequence.append(row)

    for loop_name in sorted(loops_data.keys(), key=_loop_sort_key):
        if loop_name in emitted_loops:
            continue
        curve_rows = loops_data.get(loop_name, [])
        if not curve_rows:
            continue
        final_sequence.append(create_row(4))
        final_sequence.extend(curve_rows)

    return np.array(final_sequence)