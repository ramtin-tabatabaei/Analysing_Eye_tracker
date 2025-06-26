import cv2
import numpy as np
import pandas as pd
import json
from copy import copy

import cv2
import cv2.aruco as aruco
import numpy as np
import json

def undistort_point(pt, K, D):
    """Undistort a single (x,y) point into image space."""
    arr = np.array(pt, dtype=np.float32).reshape(1,1,2)
    u = cv2.undistortPoints(arr, K, D, P=K)
    return float(u[0,0,0]), float(u[0,0,1])

def undistorted_marker_centers(corners, ids, K, D, valid_ids):
    pts = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        if mid not in valid_ids:
            continue
        c = marker_corners.reshape(4,2).mean(axis=0)
        pts[mid] = undistort_point((c[0], c[1]), K, D)
    return pts

def predict_missing_corners(centers):
    predictors = {
        22: [(21,17), ( 8,23)],
        17: [(21,22), (15,14)],
        14: [(15,17), (24,23)],
        23: [( 8,22), (24,14)],
    }
    for corner, rules in predictors.items():
        if corner in centers:
            continue
        preds = []
        for mid, other in rules:
            if mid in centers and other in centers:
                mx,my = centers[mid]
                ox,oy = centers[other]
                preds.append((2*mx - ox, 2*my - oy))
        if preds:
            xs, ys = zip(*preds)
            centers[corner] = (sum(xs)/len(xs), sum(ys)/len(ys))
    return centers

def is_inside_quad(quad_pts, pt):
    contour = np.array(quad_pts, dtype=np.float32).reshape(-1,1,2)
    return cv2.pointPolygonTest(contour, pt, False)

def is_inside_tangram(detected, test_pt_und):

    centers = predict_missing_corners(detected.copy())
    corner_ids = [22,17,14,23]
    missing = [c for c in corner_ids if c not in centers]
    if missing:
        return

    quad = [centers[c] for c in corner_ids]
    res = is_inside_quad(quad, test_pt_und)
    if res>0:
        return 1 
    else:
        return

# def check_robot_pieces(corners, ids, K, D, vis, test_pt):
def is_inside_robot_pieces(centers_all, test_pt_und):
    corner_ids = [22, 8, 19, 20]  # in CCW or CW order around the shape

    # 1) collect detected
    centers = {k: centers_all[k] for k in corner_ids if k in centers_all}

    # 2) see what’s missing
    missing = [c for c in corner_ids if c not in centers]
    if len(missing) == 1:
        # predict the single missing one
        m = missing[0]
        idx = corner_ids.index(m)
        opp = corner_ids[(idx+2) % 4]
        nbr1 = corner_ids[(idx+1) % 4]
        nbr2 = corner_ids[(idx-1) % 4]

        if all(x in centers for x in (opp, nbr1, nbr2)):
            # p_m = p_nbr1 + p_nbr2 - p_opp
            x = centers[nbr1][0] + centers[nbr2][0] - centers[opp][0]
            y = centers[nbr1][1] + centers[nbr2][1] - centers[opp][1]
            centers[m] = (x, y)
        else:
            return

    elif missing:
        # 0 missing is ideal, >1 missing is underdetermined
        return

    # 3) build hull & test
    pts = np.array([centers[c] for c in corner_ids], dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1,2)
    # quad = [tuple(pt) for pt in hull]

    res = cv2.pointPolygonTest(hull.reshape(-1,1,2), test_pt_und, False)
    if res>0:
        return 1 
    else:
        return

    

# def check_end_effector(corners, ids, K, D, vis, test_pt):
def is_inside_end_effector(centers, test_pt_und):
    """
    - Finds ArUco marker 16.
    - Undistorts both its center and the test point.
    - If their distance < threshold, we call it “inside”.
    - Draws the marker center (blue), test_pt (red), and a green circle of radius=threshold.
    """
    # rx=120

    try:
        center16 = centers[16]
    except KeyError:
        center16 = None


    if center16 is None:
        return
    x16, y16 = center16

    # 2) Try to undistort markers 9 and 8
    # 1) Build a dict of matches:
    found = {ref: centers[ref] for ref in (9, 8) if ref in centers}

    # 2a) If you want the first match out of 9, then 8:
    found_number = next((ref for ref in (9, 8) if ref in centers), None)

    # 3) Decide rx, ry
    use_small = False
    if not found:
        # none detected
        use_small = True
    else:
        # if both present and x16 < both their x’s
        # if 9 in found and 8 in found:
        if x16 < found[found_number][0]:
            use_small = True

    if use_small:
        x16 = x16 + 50
        rx, ry = 120, 180
    else:
        x16 = x16 + 30
        rx, ry = 120, 230


    # 4) Undistort test point
    tx = test_pt_und[0]
    ty = test_pt_und[1]

    # 5) Ellipse inside-test
    dx, dy = tx - x16, ty - y16
    val = (dx/rx)**2 + (dy/ry)**2
    inside = val <= 1.0

    if inside :
        return 1 
    else:
        return

   


# def check_user_pieces(corners, ids, K, D, vis, test_pt):
def is_inside_user_pieces(centers, test_pt_und):
    """
    - Top‐left corner: marker 23
    - Top‐right corner: marker 14
    - Marker 24 is the midpoint of 23–14 along the top edge
    - Bottom‐left/right share the same x as 23/14 and y at image bottom
    """

    # 1) Undistort any of {23, 14, 24} that you see
    valid_ids = [23, 14, 24]
    # tops = undistorted_marker_centers(corners, ids, K, D, valid_ids)
    tops = {k: centers[k] for k in valid_ids if k in centers}


    # 2) See which of 23/14 is missing
    missing = [c for c in (23,14) if c not in tops]
    if len(missing) == 1:
        miss = missing[0]
        other = 23 if miss == 14 else 14

        # can only predict if you have the midpoint 24 and the other corner
        if 24 in tops and other in tops:
            x24, y24 = tops[24]
            xo, yo   = tops[other]
            # midpoint rule: 24 = (23 + 14) / 2
            # => missing = 2*24 - other
            xm = 2*x24 - xo
            ym = 2*y24 - yo
            tops[miss] = (xm, ym)
        else:
            return

    elif missing:
        # none missing is ideal; >1 means underdetermined
        return

    # 3) Build the quad using 23,14 (now both present)
    h = 1200
    x23, y23 = tops[23]
    x14, y14 = tops[14]
    y_bot = h - 1

    quad = [
        (x23, y23),      # top‐left
        (x14, y14),      # top‐right
        (x14, y_bot),    # bottom‐right
        (x23, y_bot)     # bottom‐left
    ]

    # 4) Undistort test point and test
    contour = np.array(quad, dtype=np.float32).reshape(-1,1,2)
    res = cv2.pointPolygonTest(contour, test_pt_und, False)

    if res>0:
        return 1 
    else:
        return

    
# def check_robot_head(corners, ids, K, D, vis, test_pt):
def is_inside_robot_head(centers, test_pt_und):
    """
    - Tries in order: marker 26, then 28, then 25, then 27.
    - For whichever one you find first, uses its ID to pick:
        * an x-offset to apply before drawing/testing
        * the ellipse radii (rx, ry)
    - Then undistorts the test_pt, tests (dx/rx)^2+(dy/ry)^2 <= 1,
      and draws an axis-aligned oval plus annotations.
    """

    # 1) Find the first matching ID
    candidate_ids = [26, 28, 25, 27]
    found_id = None
    center = None

    for target in candidate_ids:
        for  mid in centers:
            if mid == target:
                center = centers[mid]
                found_id = target
                break
        if found_id is not None:
            break

    if found_id is None:
        return

    x, y = center

    # 2) Set up per-ID offsets and radii
    #    format: id: (x_offset, rx, ry)
    params = {
        26: ( 30, 100),
        28: ( -30, 100),
        25: ( 70, -150),
        27: ( -100, -150),
    }
    rx = 140
    ry = 120
    x_off, y_off = params[found_id]
    x_eff = x + x_off
    y_eff = y + y_off

    # 3) Undistort the test point
    tx = test_pt_und[0]
    ty = test_pt_und[1]

    # 4) Ellipse inclusion test
    dx, dy = tx - x_eff, ty - y_eff
    val = (dx/rx)**2 + (dy/ry)**2
    inside = val <= 1.0
    status = "inside" if inside else "outside"

    if inside:
        return 1 
    else:
        return


def is_inside_robot_body(centers, test_pt):

    # helper to build & test a rectangle, given four undistorted pts
    def _test_and_draw(pts4):
        # pts4 = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] in order
        contour = np.array(pts4, dtype=np.float32).reshape(-1,1,2)
        inside = cv2.pointPolygonTest(contour, test_pt, False) > 0
        if inside:
            return 1 
        else:
            return


    # 1) all four corners exist?
    if all(m in centers for m in (25,27,10,9)):
        pts = []
        for m in (25,27,10,9):
            x, y = centers[m]
            if m == 10:
                x += 40
            elif m == 9:
                x -= 40
            centers[m] = (x, y)     # reassign the updated tuple
            pts.append(centers[m])
        
        return _test_and_draw(pts)
    

    # 2) 25 & 27 exist, but one of 9/10 missing?
    if 25 in centers and 27 in centers:
        avail = [m for m in (9,18,10) if m in centers]
        if avail:
            # pick the first available for y
            y_bot = centers[avail[0]][1]
            x25,y25 = centers[25]
            x27,y27 = centers[27]
            pts = [
                (x25, y25),      # top-left
                (x27, y27),      # top-right
                (x27, y_bot),    # bot-right
                (x25, y_bot),    # bot-left
            ]

            return _test_and_draw(pts)

    # 3) either (25,10) or (27,9)?
    if 16 not in centers:
        return
    
    check_points = [25, 8, 9]
    check_point = None

    for mid in check_points:
        if mid in centers:
            check_point = mid
            break

    if check_point is None:
        return    

    x16 = centers[16][0]
    #   try (25,10)
    if x16 < centers[check_point][0]:
        
        if 25 in centers and 10 in centers:
            x1,y1 = centers[25]   # p1
            x3,y3 = centers[10]   # p3
            x3 = x3 + 50
            # compute p2 and p4
            pts = [
                (x1, y1),      # top-left
                (x3, y1),      # top-right
                (x3, y3),      # bottom-right
                (x1, y3),      # bottom-left
            ]
            return _test_and_draw(pts)
        
        #   try (27,9)
        if 27 in centers and 9 in centers:
            x1,y1 = centers[27]   # p1
            x3,y3 = centers[9]    # p3
            x3 = x3 -50
            # compute p2 and p4
            pts = [
                (x1, y1),
                (x3, y1),
                (x3, y3),
                (x1, y3),
            ]
            return _test_and_draw(pts)
            
        if 9 in centers and 10 in centers:
            x1,y1 = centers[9]   # p1
            x2,y2 = centers[10]   # p3
            x1 -=  70
            x2 +=  70
            # compute p2 and p4
            pts = [
                (x1, 0),      # top-left
                (x2, 0),      # top-right
                (x2, y2),      # bottom-right
                (x1, y1),      # bottom-left
            ]
            return _test_and_draw(pts)
        

def detect_place_attention(
    tangram,
    robot_body,
    robot_pieces,
    user_pieces,
    end_effector,
    robot_head,
):
    # map each argument’s name to its value
    attention_map = {
        'end_effector': end_effector,
        'robot_head': robot_head,
        'tangram': tangram,
        'robot_pieces': robot_pieces,
        'robot_body': robot_body,
        'user_pieces': user_pieces
    }

    # find and return the first name whose value is 1
    for name, val in attention_map.items():
        if val == 1:
            return name  # or return val if you want to return the number itself

    # if none are 1
    return "No detected places"

            



def analyze_csv(csv_path, json_path, out_path):
    df = pd.read_csv(csv_path)

    with open(json_path, 'r') as f:
        data = json.load(f)
    K = np.array(data['camera_matrix'])
    D = np.array(data['distortion_coefficients'])

    # override if needed:
    D = np.zeros((5, 1), dtype=float)
    K = np.eye(3, dtype=float)

    results = []
    for j, row in df.iterrows():
        # build centers
        centers = {}
        cols = row.index.tolist()
        for i in range(4, len(cols), 3):
            idc, xc, yc = cols[i], cols[i+1], cols[i+2]
            if pd.isna(row[idc]): 
                continue
            mid = int(row[idc])
            x, y = float(row[xc]), float(row[yc])
            centers[mid] = undistort_point((x,y), K, D)
        # undistort test_pt
        tx = float(row['gaze_x_px'])
        ty = float(row['gaze_y_px'])
        test_pt = undistort_point((tx,ty), K, D)

        place = detect_place_attention(is_inside_tangram(centers, test_pt), is_inside_robot_body(centers, test_pt), is_inside_robot_pieces(centers, test_pt), is_inside_user_pieces(centers, test_pt),
                                is_inside_end_effector(centers, test_pt), is_inside_robot_head(centers, test_pt))

    


        results.append({
            'timestamp_ns':         row['timestamp_ns'],
            'Video_time': row["Video_time"],
            'Place of Interest':      place 
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote results to {out_path!r}")

if __name__ == "__main__":
    import sys
    for i in range (1, 55):
        if i ==9 or i ==3:
            continue
        csv_path  = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/Video_Gaze_Aruco.csv'
        json_path = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/scene_camera.json'
        out_path  = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {i}/region_checks.csv'  # e.g. "region_checks.csv"
        analyze_csv(csv_path, json_path, out_path)
