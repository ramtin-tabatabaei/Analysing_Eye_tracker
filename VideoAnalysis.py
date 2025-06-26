import cv2
import cv2.aruco as aruco
import numpy as np
import json
import pandas as pd

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

def check_tangram_figure(corners, ids, K, D, vis, test_pt):
    all_ids = {22,21,17,15,14,24,23,8}
    detected = undistorted_marker_centers(corners, ids, K, D, all_ids)

    # draw detected
    for mid, (ux,uy) in detected.items():
        cv2.circle(vis, (int(ux),int(uy)), 4, (255,0,0), -1)
        cv2.putText(vis, str(mid), (int(ux)+5,int(uy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    centers = predict_missing_corners(detected.copy())
    corner_ids = [22,17,14,23]
    missing = [c for c in corner_ids if c not in centers]
    print(missing)
    if missing:
        print(f"[Tangram] Missing corners {missing}, skipping.")
        return

    # undistort test_pt
    test_pt_und = undistort_point(test_pt, K, D)

    quad = [centers[c] for c in corner_ids]
    print(quad)
    res = is_inside_quad(quad, test_pt_und)
    status = "inside" if res>0 else ("on edge" if res==0 else "outside")
    # print(f"[Tangram] Point {test_pt}→{test_pt_und} is {status}.")

    # draw quad
    for i in range(4):
        p1 = tuple(map(int, quad[i]))
        p2 = tuple(map(int, quad[(i+1)%4]))
        cv2.line(vis, p1, p2, (0,255,0) if res>0 else (0,0,255), 2)

    # col = (0,255,0) if res>0 else (0,0,255)
    # cv2.circle(vis, tuple(map(int,test_pt_und)), 6, col, -1)

def check_robot_pieces(corners, ids, K, D, vis, test_pt):
    corner_ids = [22, 8, 19, 20]  # in CCW or CW order around the shape
    centers = {}

    # 1) collect detected
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        mid = int(marker_id)
        if mid in corner_ids:
            c = marker_corners.reshape(4,2).mean(axis=0)
            centers[mid] = undistort_point((c[0], c[1]), K, D)
            ux,uy = centers[mid]
            cv2.circle(vis, (int(ux),int(uy)), 5, (255,0,0), -1)
            cv2.putText(vis, str(mid), (int(ux)+5,int(uy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

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
            print(f"[Robot] Predicted missing marker {m} at ({x:.1f},{y:.1f})")
        else:
            print(f"[Robot] Not enough detected to predict {m}, skipping.")
            return

    elif missing:
        # 0 missing is ideal, >1 missing is underdetermined
        print(f"[Robot] Missing markers {missing}, cannot predict >1 point, skipping.")
        return

    # 3) build hull & test
    pts = np.array([centers[c] for c in corner_ids], dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1,2)
    quad = [tuple(pt) for pt in hull]

    test_pt_und = undistort_point(test_pt, K, D)
    res = cv2.pointPolygonTest(hull.reshape(-1,1,2), test_pt_und, False)
    status = "inside" if res>0 else ("on edge" if res==0 else "outside")
    # print(f"[Robot] Point {test_pt}→{test_pt_und} is {status}.")

    # 4) draw quad & point
    for i in range(len(quad)):
        p1 = tuple(map(int, quad[i]))
        p2 = tuple(map(int, quad[(i+1)%len(quad)]))
        cv2.line(vis, p1, p2, (0,255,0) if res>0 else (0,0,255), 2)

    # color = (0,255,0) if res>0 else (0,0,255)
    

def check_end_effector(corners, ids, K, D, vis, test_pt):
    """
    - Finds ArUco marker 16.
    - Undistorts both its center and the test point.
    - If their distance < threshold, we call it “inside”.
    - Draws the marker center (blue), test_pt (red), and a green circle of radius=threshold.
    """
    # rx=120
    # ry=180
    # 1) Undistort marker 16 center
    center16 = None
    for mc, mid in zip(corners, ids.flatten()):
        if mid == 16:
            c = mc.reshape(4,2).mean(axis=0)
            center16 = undistort_point((c[0], c[1]), K, D)
            break
    if center16 is None:
        print("[User] Marker 18 not found.")
        return
    x16, y16 = center16

    # 2) Try to undistort markers 9 and 8
    found = {}
    for ref in (9, 8):
        for mc, mid in zip(corners, ids.flatten()):
            if mid == ref:
                c = mc.reshape(4,2).mean(axis=0)
                found[ref] = undistort_point((c[0], c[1]), K, D)
                found_number = ref
                break

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

    print(f"[User] Using rx={rx}, ry={ry} (use_small={use_small})")

    # 4) Undistort test point
    tx, ty = undistort_point(test_pt, K, D)

    # 5) Ellipse inside-test
    dx, dy = tx - x16, ty - y16
    val = (dx/rx)**2 + (dy/ry)**2
    inside = val <= 1.0
    status = "inside" if inside else "outside"
    print(f"[User] Ellipse test: {val:.3f} → {status}")

    # 6) Draw
    # ellipse
    cv2.ellipse(vis,
                (int(x16), int(y16)),
                (rx, ry),
                0, 0, 360,
                (0,255,0) if inside else (0,0,255), 2)
    # marker 18
    cv2.circle(vis, (int(x16), int(y16)), 5, (255,0,0), -1)
    cv2.putText(vis, "16", (int(x16)+6,int(y16)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    # optionally draw 9 and 8 if present
    for ref, (xr, yr) in found.items():
        cv2.circle(vis, (int(xr), int(yr)), 5, (255,255,0), -1)
        cv2.putText(vis, str(ref), (int(xr)+6,int(yr)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    # test point
    # cv2.circle(vis, (int(tx), int(ty)), 6, col, -1)


def check_user_pieces(corners, ids, K, D, vis, test_pt):
    """
    - Top‐left corner: marker 23
    - Top‐right corner: marker 14
    - Marker 24 is the midpoint of 23–14 along the top edge
    - Bottom‐left/right share the same x as 23/14 and y at image bottom
    """

    # 1) Undistort any of {23, 14, 24} that you see
    valid_ids = {23, 14, 24}
    tops = undistorted_marker_centers(corners, ids, K, D, valid_ids)

    # draw detected tops
    for mid, (ux, uy) in tops.items():
        cv2.circle(vis, (int(ux), int(uy)), 5, (255,0,0), -1)
        cv2.putText(vis, str(mid),
                    (int(ux)+5, int(uy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

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
            print(f"[User] Predicted marker {miss} at ({xm:.1f},{ym:.1f})")
            # draw prediction in cyan
            cv2.circle(vis, (int(xm), int(ym)), 5, (255,255,0), -1)
            cv2.putText(vis, str(miss),
                        (int(xm)+5, int(ym)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        else:
            print(f"[User] Cannot predict {miss}: need markers 24 and {other}.")
            return

    elif missing:
        # none missing is ideal; >1 means underdetermined
        print(f"[User] Missing top‐corners {missing}, skipping.")
        return

    # 3) Build the quad using 23,14 (now both present)
    h = vis.shape[0]
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
    test_pt_und = undistort_point(test_pt, K, D)
    contour = np.array(quad, dtype=np.float32).reshape(-1,1,2)
    res = cv2.pointPolygonTest(contour, test_pt_und, False)
    status = "inside" if res>0 else ("on edge" if res==0 else "outside")
    # print(f"[User] Point {test_pt}→{test_pt_und} is {status} the user‐piece quad.")

    # 5) Draw quad & test‐point
    for i in range(4):
        p1 = tuple(map(int, quad[i]))
        p2 = tuple(map(int, quad[(i+1)%4]))
        cv2.line(vis, p1, p2, (0,255,0) if res>0 else (0,0,255), 2)


    # col = (0,255,0) if res>0 else (0,0,255)
    # cv2.circle(vis,
    #            (int(test_pt_und[0]), int(test_pt_und[1])),
    #            7, col, -1)
    
def check_robot_head(corners, ids, K, D, vis, test_pt):
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
        for mc, mid in zip(corners, ids.flatten()):
            if mid == target:
                c = mc.reshape(4,2).mean(axis=0)
                center = undistort_point((c[0], c[1]), K, D)
                found_id = target
                break
        if found_id is not None:
            break

    if found_id is None:
        print("[EndEff] None of IDs", candidate_ids, "found—skipping.")
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

    # print(f"[EndEff] Found ID {found_id}. x={x:.1f} → x_eff={x_eff:.1f}, rx={rx}, ry={ry}")

    # 3) Undistort the test point
    tx, ty = undistort_point(test_pt, K, D)

    # 4) Ellipse inclusion test
    dx, dy = tx - x_eff, ty - y_eff
    val = (dx/rx)**2 + (dy/ry)**2
    inside = val <= 1.0
    status = "inside" if inside else "outside"
    # print(f"[EndEff] Ellipse test: {val:.3f} → {status}")

    # 5) Draw on vis
    #    a) ellipse centered at (x_eff, y)
    cv2.ellipse(vis,
                (int(x_eff), int(y_eff)),
                (rx, ry),
                0, 0, 360,
                (0,255,0) if inside else (0,0,255), 2)

    #    b) marker center (blue)
    cv2.circle(vis, (int(x), int(y)), 5, (255,0,0), -1)
    cv2.putText(vis, str(found_id), (int(x)+6, int(y)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    #    c) test point (green/red)
    # col = (0,255,0) if inside else (0,0,255)
    # cv2.circle(vis, (int(tx), int(ty)), 6, col, -1)
    # cv2.line(vis,
    #          (int(x_eff), int(y_eff)),
    #          (int(tx), int(ty)),
    #          col, 1)
    

def check_robot_body(corners, ids, K, D, vis, test_pt):
    """
    1. If {25,27,10,9} all exist: quad from those four, test point inside.
    2. Elif 25 and 27 exist but one of {9,10} is missing:
         • top corners = 25,27
         • bottom corners: same x’s, y from whichever of {9,18,10} is present
    3. Elif either (25,10) or (27,9) exist:
         • check x16 < x25 (for 25,10) or x16 < x27 (for 27,9)
         • if so, draw that rectangle & test
    4. Else: skip.
    """
    # helper to build & test a rectangle, given four undistorted pts
    def _test_and_draw(pts4, label):
        # pts4 = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] in order
        contour = np.array(pts4, dtype=np.float32).reshape(-1,1,2)
        # undistort test_pt
        tx, ty = undistort_point(test_pt, K, D)
        inside = cv2.pointPolygonTest(contour, (tx,ty), False) > 0
        color = (0,255,0) if inside else (0,0,255)
        # draw
        for i in range(4):
            p1 = tuple(map(int, pts4[i]))
            p2 = tuple(map(int, pts4[(i+1)%4]))
            cv2.line(vis, p1, p2, color, 2)
        # cv2.circle(vis, (int(tx),int(ty)), 6, color, -1)
        # print(f"[Complex:{label}] Point inside? {inside}")

    # undistort all markers in question
    want = [9,10,25,27,16,18]
    centers = {}
    for mc, mid in zip(corners, ids.flatten()):
        if mid in want:
            c = mc.reshape(4,2).mean(axis=0)
            centers[mid] = undistort_point((c[0],c[1]), K, D)

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
            # print(f"Adjusted center[{m}].x = {x}")
            pts.append(centers[m])
        _test_and_draw(pts, "A(25,27,10,9)")
        return

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
            _test_and_draw(pts, "B(25,27 + y)")
            return

    # 3) either (25,10) or (27,9)?
    if 16 not in centers:
        print("[Complex] Marker 16 missing, skipping phase 3.")
        return
    
    check_points = [25, 8, 9]
    check_point = None

    for mid in check_points:
        if mid in centers:
            check_point = mid
            break

    if check_point is None:
        print("Neither 25 nor 8 was found.")
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
            _test_and_draw(pts, "C(25,10)")
            return
        
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
            _test_and_draw(pts, "D(27,9)")
            return
        
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
            _test_and_draw(pts, "C(25,10)")
            return

    print("[Complex] No valid configuration found, skipping.")


def detect_and_test_with_visuals(video_path, json_path,
                                 frame_no=30, test_pt=(500,500)):
    with open(json_path, 'r') as f:
        data = json.load(f)
    K = np.array(data['camera_matrix'])
    D = np.array(data['distortion_coefficients'])

    D = np.zeros((5, 1), dtype=float)
    K = np.eye(3, dtype=float)


    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no-1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: can't read frame {frame_no}")
        return

    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    params     = aruco.DetectorParameters()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None:
        print("No markers found.")
        return

    vis = frame.copy()
    aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(255,0,0))
    # cv2.circle(vis, test_pt, 6, (0,0,255), -1)  # raw test‐point for reference

    check_tangram_figure(corners, ids, K, D, vis, test_pt)
    check_robot_pieces(corners, ids, K, D, vis, test_pt)
    check_user_pieces(corners, ids, K, D, vis, test_pt)
    check_end_effector(corners, ids, K, D, vis, test_pt)
    check_robot_head(corners, ids, K, D, vis, test_pt)
    check_robot_body(corners, ids, K, D, vis, test_pt)

    cv2.imshow("Detected + Predicted", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def process_video(video_path, json_path, gaze_data):
def process_video(video_path, json_path, gaze_data, detected_place_data):

    # 1) load intrinsics once
    with open(json_path, 'r') as f:
        data = json.load(f)
    K = np.array(data['camera_matrix'])
    D = np.array(data['distortion_coefficients'])

    D = np.zeros((5, 1), dtype=float)
    K = np.eye(3, dtype=float)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # if frame_index < 23000:
        #     continue

        vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        params     = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

        test_pt = (gaze_data['gaze_x_px'][frame_index],gaze_data['gaze_y_px'][frame_index])
        cv2.circle(vis, test_pt, 15, (0,0,255), 10)
        cv2.putText(vis, f"Frame Number: {frame_index:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        place = detected_place_data['Place of Interest'][frame_index]
        cv2.putText(vis, f"Place: {place}", (1000, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # if place != "No detected places":
        #     cv2.putText(vis, f"Place: {place}", (test_pt[0], test_pt[1]),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        if ids is not None:
            # draw raw detections + test‐point
            aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(255,0,0))

            # call each of your check_* routines
            check_tangram_figure(corners, ids, K, D, vis, test_pt)
            check_robot_pieces(  corners, ids, K, D, vis, test_pt)
            check_user_pieces(   corners, ids, K, D, vis, test_pt)
            check_end_effector(  corners, ids, K, D, vis, test_pt)
            check_robot_head(    corners, ids, K, D, vis, test_pt)
            check_robot_body(    corners, ids, K, D, vis, test_pt)

        cv2.imshow("Detected + Predicted", vis)
        # wait 30ms → ~33fps. Press 'q' to quit.
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pariticipant_number = 2
    video_file = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {pariticipant_number}/Neon Scene Camera v1 ps1.mp4'
    json_file = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {pariticipant_number}/scene_camera.json'
    gaze_file = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {pariticipant_number}/Video_Gaze_Aruco.csv'
    detected_placed_file = f'/Volumes/R@mtin/Year 2/Experiment/EyeTracker/Data/Participant {pariticipant_number}/region_checks_filled.csv'

    gaze_data = pd.read_csv(gaze_file)

    detected_placed_data = pd.read_csv(detected_placed_file)

    # process_video(video_file, json_file, gaze_data)

    process_video(video_file, json_file, gaze_data, detected_placed_data)
