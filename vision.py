import io, cv2, numpy as np
from PIL import Image
from typing import List, Tuple
from models import Element, Finding, Capture

def load_image_bytes(b: bytes):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def align_thermal_to_rgb(thermal_bgr: np.ndarray, rgb_bgr: np.ndarray) -> np.ndarray:
    # Feature-match homography; fallback to resize
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), None)
        if des1 is None or des2 is None: raise ValueError
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if m.distance < 0.7*n.distance]
        if len(good) < 8: raise ValueError
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return cv2.warpPerspective(thermal_bgr, H, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    except Exception:
        return cv2.resize(thermal_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

def estimate_deltaT_map(thermal_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gmin, gmax = np.percentile(gray, [1, 99])
    norm = np.clip((gray - gmin) / max(1e-6, (gmax - gmin)), 0, 1)
    delta = norm - np.median(norm)  # relative temp difference
    return delta  # negative = cooler

def detect_elements_stub(rgb_bgr: np.ndarray) -> List[Element]:
    h,w,_ = rgb_bgr.shape
    box = (int(w*0.3), int(h*0.25), int(w*0.4), int(h*0.35))
    return [Element(id="win_1", type="window", box=box, side="front")]

def _connected_masks(binary: np.ndarray, min_area=200):
    n, labels = cv2.connectedComponents(binary)
    masks=[]
    for i in range(1, n):
        m = (labels==i).astype(np.uint8)
        if m.sum() >= min_area: masks.append(m)
    return masks

def detect_draughts(delta: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    gy, gx = np.gradient(delta)
    mag = np.hypot(gx, gy)
    cold = ((delta < -0.1) & (mag > 0.05)).astype(np.uint8)*255
    cold = cv2.medianBlur(cold, 5)
    out=[]
    for m in _connected_masks(cold, 200):
        strength = float(abs(delta[m>0]).mean())
        if strength > 0.08: out.append((m, strength))
    return out

def detect_bridges(delta: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    cold = (delta < -0.08).astype(np.uint8)*255
    edges = cv2.Canny(cold, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=5)
    if lines is None: return []
    line_mask = np.zeros_like(cold)
    total = 0
    for x1,y1,x2,y2 in lines[:,0]:
        cv2.line(line_mask, (x1,y1), (x2,y2), 255, 2)
        total += np.hypot(x2-x1, y2-y1)
    if total < 120: return []
    return [(line_mask, 0.1 + total/1000.0)]

def detect_insulation_gaps(delta: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    blobs = (delta < -0.12).astype(np.uint8)*255
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts,_ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400: continue
        mask = np.zeros_like(blobs)
        cv2.drawContours(mask, [c], -1, 255, -1)
        strength = float(abs(delta[mask>0]).mean())
        out.append((mask, strength))
    return out

def overlap(mask: np.ndarray, box) -> float:
    x,y,w,h = box
    sub = mask[y:y+h, x:x+w]
    return sub.astype(bool).mean() if sub.size else 0.0

def impact_from_score(s: float) -> str:
    return "large" if s>0.2 else "medium_large" if s>0.12 else "medium"

def confidence_from_env(deltaT_env: float, clarity: float) -> str:
    base = "low" if (deltaT_env is None or deltaT_env < 6) else "medium" if deltaT_env < 10 else "high"
    return "medium" if (clarity<0.1 and base=="high") else base

def analyze(rgb_bytes: bytes, thermal_bytes: bytes, inside: bool, t_in: float, t_out: float):
    rgb = load_image_bytes(rgb_bytes)
    thermal = load_image_bytes(thermal_bytes)
    warped = align_thermal_to_rgb(thermal, rgb)
    delta = estimate_deltaT_map(warped)
    elements = detect_elements_stub(rgb)

    findings=[]
    # draughts
    for mask, s in detect_draughts(delta):
        best, ol = None, 0
        for el in elements:
            o = overlap(mask, el.box)
            if o>ol: best, ol = el, o
        if best and ol>0.1:
            findings.append(Finding(
                element_id=best.id, issue="draught",
                delta_c=round(10*s,2), impact=impact_from_score(s),
                confidence=confidence_from_env((t_in - t_out), s),
                reason="Cold streak near opening"
            ))
    # bridges
    for mask, s in detect_bridges(delta):
        best, ol = None, 0
        for el in elements:
            o = overlap(mask, el.box)
            if o>ol: best, ol = el, o
        if best and ol>0.05:
            findings.append(Finding(
                element_id=best.id, issue="thermal_bridge",
                delta_c=round(10*s,2), impact=impact_from_score(s),
                confidence=confidence_from_env((t_in - t_out), s),
                reason="Long cold line at frame/lintel"
            ))
    # insulation gaps
    for mask, s in detect_insulation_gaps(delta):
        el = elements[0]
        findings.append(Finding(
            element_id=el.id, issue="insulation_gap",
            delta_c=round(10*s,2), impact=impact_from_score(s),
            confidence=confidence_from_env((t_in - t_out), s),
            reason="Large cold patch"
        ))

    cap = Capture(inside=inside, t_in=t_in, t_out=t_out, deltaT_env_c=(t_in - t_out))
    return elements, findings, cap
