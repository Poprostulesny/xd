import cv2
import numpy as np
import sys

def read_im(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im

def align_orb(haystack_color, needle_color, min_matches=10):
    img1_gray = cv2.cvtColor(haystack_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(needle_color, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < min_matches:
        return None

    src_pts = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return H

def warp_to_ref(img_to_warp, H, ref_shape):
    h, w = ref_shape[:2]
    warped = cv2.warpPerspective(img_to_warp, H, (w, h), flags=cv2.INTER_LINEAR)
    return warped

def simple_resize(img, ref_shape):
    h, w = ref_shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def diff_and_mark(imgA, imgB, out_mask='diff_mask.png', out_marked='diff_marked.png',
                  thresh_val=45, min_area=1200):
    # imgs must be same size
    h_img, w_img = imgA.shape[:2]
    diff = cv2.absdiff(imgA, imgB)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=4)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked = imgA.copy()
    
    # Filtruj kontury: tylko w prawej tylnej części
    # Skrzynka jest w prawej części (x > 0.6 * szerokość)
    # i w środkowej wysokości (0.3 < y < 0.65 * wysokość)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        
        center_x = x + w/2
        center_y = y + h/2
        
        if center_x > w_img * 0.6 and 0.3 * h_img < center_y < 0.65 * h_img:
            cv2.rectangle(marked, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imwrite(out_mask, th)
    cv2.imwrite(out_marked, marked)
    return th, marked

def main(pathA, pathB):
    A = read_im(pathA)
    B = read_im(pathB)

    # limit max size to speed up (optional)
    max_dim = 2000
    for img in (A, B):
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            if img is A:
                A = cv2.resize(A, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                B = cv2.resize(B, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # try ORB+homography alignment
    H = align_orb(A, B)
    if H is not None:
        try:
            B_aligned = warp_to_ref(B, H, A.shape)
            print("Alignment: homography ok")
        except Exception:
            B_aligned = simple_resize(B, A.shape)
            print("Alignment: homography failed during warp, fallback to resize")
    else:
        # fallback: resize B to A
        B_aligned = simple_resize(B, A.shape)
        print("Alignment: not enough feature matches, used simple resize")

    diff_mask, marked = diff_and_mark(A, B_aligned,
                                      out_mask='diff_mask.png',
                                      out_marked='diff_marked.png',
                                      thresh_val=45,
                                      min_area=1200)
    print("Saved: diff_mask.png, diff_marked.png")

if __name__ == '__main__':
    img1 = "data_oriented/czyste/48001F003202511180025 czarno.bmp"
    img2 = "data_oriented/brudne/48001F003202511190032 czarno.bmp"
    main(img2,img1)