"""
OMR স্ক্যানার — ব্যাকএন্ড (Flask)
মিঠাপুকুর মহাবিদ্যালয়, রংপুর
Resolution: 1241×1755 (Paint দিয়ে মাপা + HoughCircles দিয়ে calibrate করা)
"""

import cv2
import numpy as np
import math
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════
# CALIBRATED CONSTANTS (actual OMR image থেকে HoughCircles দিয়ে বের করা)
# ══════════════════════════════════════════════════════════════════════
W, H = 1241, 1755

# Registration mark centers (Paint দিয়ে মাপা, perspective correction এ ব্যবহার)
REG_MARKS = {
    'tl': (91,  148),
    'tr': (1150, 148),
    'br': (1195, 1614),
    'bl': (47,  1614),
}

# রোল নম্বর — প্রতিটি bubble এর center (x, y)
ROLL_COL_X  = [752, 784, 814, 844, 874]   # 5 digit columns
ROLL_ROW_Y  = [204, 234, 266, 296, 328, 358, 388, 418, 450, 478]  # digit 0-9

# প্রশ্নের উত্তর — প্রতিটি option এর x range (x1, x2)
# COL1: প্রশ্ন ১–২৫
COL1_X_RANGES = [(48,96), (110,162), (174,226), (238,290)]   # ক, খ, গ, ঘ
COL1_ROW_Y    = [200,260,319,374,432,489,545,601,658,714,
                 769,825,879,934,988,1042,1097,1149,1204,1256,
                 1308,1361,1413,1465,1518]

# COL2: প্রশ্ন ২৬–৫০
COL2_X_RANGES = [(368,416), (434,482), (496,544), (560,608)]  # ক, খ, গ, ঘ
COL2_ROW_Y    = [200,258,316,374,432,488,544,600,658,714,
                 768,822,878,934,988,1042,1094,1148,1204,1256,
                 1308,1362,1412,1464,1518]

BUBBLE_RADIUS = 14       # bubble এর approximate half-size
FILL_THRESHOLD = 0.30    # এর বেশি dark ratio হলে ভরাট ধরা হবে
OPTIONS = ['ক', 'খ', 'গ', 'ঘ']


# ══════════════════════════════════════════════════════════════════════
# ১. Perspective Correction
# ══════════════════════════════════════════════════════════════════════
def correct_perspective(img):
    """
    OMR শীটের ৪ কোণের registration mark খুঁজে perspective warp করে।
    পদ্ধতি ১: Registration mark (কালো বর্গ) detect
    পদ্ধতি ২: Largest quadrilateral contour
    পদ্ধতি ৩: শুধু resize (fallback)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape

    margin_x = int(iw * 0.13)
    margin_y = int(ih * 0.13)

    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    regions = [
        (thresh[0:margin_y,        0:margin_x],        0,        0),         # top-left
        (thresh[0:margin_y,        iw-margin_x:iw],    iw-margin_x, 0),      # top-right
        (thresh[ih-margin_y:ih,    iw-margin_x:iw],    iw-margin_x, ih-margin_y),  # bottom-right
        (thresh[ih-margin_y:ih,    0:margin_x],        0,        ih-margin_y),     # bottom-left
    ]

    mark_pts = []
    for (region, ox, oy) in regions:
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            mark_pts.append(None); continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 40:
            mark_pts.append(None); continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            mark_pts.append(None); continue
        cx = int(M['m10']/M['m00']) + ox
        cy = int(M['m01']/M['m00']) + oy
        mark_pts.append((cx, cy))

    # পদ্ধতি ১: সব ৪টি mark পাওয়া গেলে
    if all(p is not None for p in mark_pts):
        tl, tr, br, bl = mark_pts
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (W, H))

    # পদ্ধতি ২: Largest quadrilateral
    blur   = cv2.GaussianBlur(gray, (5,5), 0)
    edges  = cv2.Canny(blur, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in cnts[:8]:
        if cv2.contourArea(cnt) < ih * iw * 0.25:
            break
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            s = pts.sum(axis=1); d = np.diff(pts, axis=1).flatten()
            ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                                 pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)
            dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(ordered, dst)
            return cv2.warpPerspective(img, M, (W, H))

    # পদ্ধতি ৩: শুধু resize
    return cv2.resize(img, (W, H))
    def camscanner_effect(img):
    """
    CamScanner এর মতো:
    ১. Shadow remove
    ২. Contrast enhance  
    ৩. Document-like white background
    """
    # RGB তে convert
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ── Shadow Remove ──
    # প্রতিটি channel এ large kernel দিয়ে background estimate করি
    result_planes = []
    for plane in cv2.split(rgb):
        dilated = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        result_planes.append(norm)
    
    shadow_removed = cv2.merge(result_planes)
    
    # ── Adaptive Threshold — document look ──
    gray = cv2.cvtColor(shadow_removed, cv2.COLOR_RGB2GRAY)
    
    # CLAHE — local contrast enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # BGR তে ফেরত
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return result


# ══════════════════════════════════════════════════════════════════════
# ২. Bubble Fill Ratio
# ══════════════════════════════════════════════════════════════════════
def dark_ratio(gray, cy, x1, x2, r=BUBBLE_RADIUS):
    """একটি bubble zone কতটুকু dark (ভরাট) সেটা 0-1 এ return করে।"""
    bz = gray[cy-r:cy+r, x1:x2]
    if bz.size == 0:
        return 0.0
    return float(1 - np.mean(bz) / 255)


# ══════════════════════════════════════════════════════════════════════
# ৩. রোল নম্বর পড়া
# ══════════════════════════════════════════════════════════════════════
def read_roll_number(gray):
    """5-digit রোল নম্বর পড়ে। ভরাট না হলে '?' দেয়।"""
    roll = []
    for cx in ROLL_COL_X:
        best_ratio = 0.0
        best_digit = -1
        for digit, ry in enumerate(ROLL_ROW_Y):
            bz = gray[ry-BUBBLE_RADIUS:ry+BUBBLE_RADIUS,
                      cx-BUBBLE_RADIUS:cx+BUBBLE_RADIUS]
            if bz.size == 0:
                continue
            ratio = 1 - np.mean(bz) / 255
            if ratio > best_ratio:
                best_ratio = ratio
                best_digit = digit
        roll.append(str(best_digit) if best_ratio >= FILL_THRESHOLD else '?')
    return ''.join(roll)


# ══════════════════════════════════════════════════════════════════════
# ৪. প্রশ্নের উত্তর পড়া
# ══════════════════════════════════════════════════════════════════════
def read_answers(gray, total_questions):
    """
    প্রশ্ন সংখ্যা অনুযায়ী উত্তর পড়ে।
    ≤25 → শুধু COL1
    >25 → COL1 (১-২৫) + COL2 (২৬ থেকে বাকি)
    """
    answers = {}

    def scan_col(x_ranges, row_y_list, start_q, count):
        for i in range(count):
            ry = row_y_list[i]
            best_ratio = 0.0
            best_opt   = -1
            for oi, (x1, x2) in enumerate(x_ranges):
                ratio = dark_ratio(gray, ry, x1, x2)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_opt   = oi
            qnum = start_q + i
            answers[qnum] = OPTIONS[best_opt] if best_ratio >= FILL_THRESHOLD else ''

    if total_questions <= 25:
        scan_col(COL1_X_RANGES, COL1_ROW_Y, 1, total_questions)
    else:
        scan_col(COL1_X_RANGES, COL1_ROW_Y, 1,  25)
        scan_col(COL2_X_RANGES, COL2_ROW_Y, 26, total_questions - 25)

    return answers


# ══════════════════════════════════════════════════════════════════════
# ৫. Main Processing
# ══════════════════════════════════════════════════════════════════════
def process_omr(image_bytes, total_questions=50):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "ছবি পড়া যাচ্ছে না। JPG/PNG ফরম্যাট ব্যবহার করুন।"}

    img  = correct_perspective(img)
    img  = camscanner_effect(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mild blur — noise কমায়, bubble edge ঠিক রাখে
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    roll    = read_roll_number(gray)
    answers = read_answers(gray, total_questions)

    return {"roll": roll, "answers": answers}


# ══════════════════════════════════════════════════════════════════════
# ৬. API Endpoints
# ══════════════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "OMR সার্ভার চালু আছে ✓"})


@app.route('/upload-omr', methods=['POST'])
def upload_omr():
    if 'file' not in request.files:
        return jsonify({"error": "ছবি পাওয়া যায়নি"}), 400
    file = request.files['file']
    total_questions = int(request.form.get('total_questions', 50))
    total_questions = max(1, min(50, total_questions))
    result = process_omr(file.read(), total_questions)
    return jsonify(result)


@app.route('/upload-omr-batch', methods=['POST'])
def upload_omr_batch():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "কোনো ছবি পাওয়া যায়নি"}), 400
    total_questions = int(request.form.get('total_questions', 50))
    total_questions = max(1, min(50, total_questions))
    results = []
    for f in files:
        if f.filename == '':
            continue
        result = process_omr(f.read(), total_questions)
        result['filename'] = f.filename
        results.append(result)
    return jsonify({"total": len(results), "results": results})

@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    print("=" * 50)
    print("  OMR সার্ভার চালু হচ্ছে...")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
