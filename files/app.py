import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
import datetime
import pandas as pd
from PIL import Image
import pickle

st.set_page_config(page_title="Face Attendance System", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0a0f; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background: #111118; border-right: 1px solid #1e1e2e; }
    .main-title { font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #00f5a0, #00d9f5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-card { background: #111118; border: 1px solid #1e1e2e; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
    .stat-number { font-size: 2.5rem; font-weight: 700; color: #00f5a0; line-height: 1; }
    .stat-label { font-size: 0.8rem; color: #555580; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
    .section-header { font-size: 1.3rem; font-weight: 600; color: #00f5a0; border-left: 3px solid #00f5a0; padding-left: 0.8rem; margin-bottom: 1.5rem; }
    .stButton > button { background: linear-gradient(135deg, #00f5a0, #00d9f5); color: #000; font-weight: 700; border: none; border-radius: 8px; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ─── SETUP ─────────────────────────────────────────────────────────────────────
os.makedirs("faces", exist_ok=True)
CSV_FILE = "attendance.csv"
MODEL_FILE = "face_model.pkl"

conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, date TEXT, time TEXT, status TEXT DEFAULT 'Present'
)
""")
conn.commit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def sync_csv():
    rows = cursor.execute("SELECT name, date, time, status FROM attendance ORDER BY date DESC, time DESC").fetchall()
    pd.DataFrame(rows, columns=["Name","Date","Time","Status"]).to_csv(CSV_FILE, index=False)

def already_marked(name, date):
    return cursor.execute("SELECT id FROM attendance WHERE name=? AND date=?", (name, date)).fetchone() is not None

def get_stats():
    total = cursor.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_count = cursor.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    unique = cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance").fetchone()[0]
    registered = len([f for f in os.listdir("faces") if f.lower().endswith((".jpg",".jpeg",".png"))])
    return total, today_count, unique, registered

def decode_image(image):
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
    return frame, gray, faces

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_data, labels, label_map = [], [], {}
    lid = 0
    for file in os.listdir("faces"):
        if not file.lower().endswith((".jpg",".jpeg",".png")): continue
        name = os.path.splitext(file)[0]
        if name not in label_map:
            label_map[name] = lid; lid += 1
        img = cv2.imread(os.path.join("faces", file), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        det = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(50,50))
        if len(det) > 0:
            x,y,w,h = det[0]
            faces_data.append(cv2.resize(img[y:y+h, x:x+w], (200,200)))
        else:
            faces_data.append(cv2.resize(img, (200,200)))
        labels.append(label_map[name])
    if faces_data:
        recognizer.train(faces_data, np.array(labels))
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"rec": recognizer, "lmap": {v:k for k,v in label_map.items()}}, f)
        return True, len(label_map)
    return False, 0

def load_model():
    if not os.path.exists(MODEL_FILE): return None, None
    with open(MODEL_FILE, "rb") as f:
        d = pickle.load(f)
    return d["rec"], d["lmap"]

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 Face Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#555580;margin-bottom:1.5rem">OpenCV powered • No dlib required • Auto CSV export</div>', unsafe_allow_html=True)

total, today_count, unique, registered = get_stats()
for col, num, label in zip(
    st.columns(4),
    [registered, today_count, unique, total],
    ["Registered Faces","Present Today","Unique Students","Total Records"]
):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
menu = st.sidebar.selectbox("📂 Navigation", ["🧑 Register Face","📷 Mark Attendance","📊 View Attendance","🗑️ Manage Data"])

# ═══ REGISTER ══════════════════════════════════════════════════════════════════
if menu == "🧑 Register Face":
    st.markdown('<div class="section-header">Register New Face</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        name = st.text_input("👤 Full Name")
        image = st.camera_input("📸 Capture Photo")
        if image and name.strip():
            frame, gray, faces = decode_image(image)
            if len(faces) == 0:
                st.error("❌ No face detected. Try better lighting.")
            else:
                image.seek(0)
                with open(f"faces/{name.strip()}.jpg", "wb") as f:
                    f.write(image.getbuffer())
                x,y,w,h = faces[0]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,245,160),3)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Registered: {name.strip()}", width=300)
                ok, count = train_model()
                if ok: st.success(f"✅ {name.strip()} registered! Model trained with {count} person(s).")
                else: st.warning("Saved but training failed. Try again.")
    with col2:
        st.info("**Tips:**\n\n• Good lighting\n• Face camera directly\n• Single face in frame")
        people = [os.path.splitext(f)[0] for f in os.listdir("faces") if f.lower().endswith((".jpg",".jpeg",".png"))]
        if people:
            st.markdown("**Registered:**")
            for p in sorted(people): st.markdown(f"• {p}")

# ═══ MARK ATTENDANCE ═══════════════════════════════════════════════════════════
elif menu == "📷 Mark Attendance":
    st.markdown('<div class="section-header">Mark Attendance</div>', unsafe_allow_html=True)
    recognizer, label_map = load_model()
    if recognizer is None:
        st.warning("⚠️ No model found. Register faces first.")
    else:
        st.info(f"📋 {len(label_map)} person(s) loaded: {', '.join(label_map.values())}")
        threshold = st.slider("Recognition Sensitivity", 50, 150, 85, help="Increase if face not recognized")
        image = st.camera_input("📷 Scan Face")
        if image:
            frame, gray, faces = decode_image(image)
            if len(faces) == 0:
                st.error("❌ No face detected.")
            else:
                now = datetime.datetime.now()
                date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
                for (x,y,w,h) in faces:
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200,200))
                    lid, conf = recognizer.predict(face_roi)
                    name = label_map.get(lid, "Unknown") if conf < threshold else "Unknown"
                    color = (0,245,160) if name != "Unknown" else (255,85,85)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
                    cv2.rectangle(frame,(x,y+h-30),(x+w,y+h),color,-1)
                    cv2.putText(frame,f"{name}({int(conf)})",(x+4,y+h-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                    if name != "Unknown":
                        if already_marked(name, date_str):
                            st.warning(f"⚠️ {name} — Already marked today")
                        else:
                            cursor.execute("INSERT INTO attendance (name,date,time,status) VALUES (?,?,?,?)",(name,date_str,time_str,"Present"))
                            conn.commit(); sync_csv()
                            st.success(f"✅ {name} — Marked at {time_str} | attendance.csv updated")
                    else:
                        st.error(f"❌ Unknown face (conf:{int(conf)}) — Try increasing sensitivity")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

# ═══ VIEW ATTENDANCE ═══════════════════════════════════════════════════════════
elif menu == "📊 View Attendance":
    st.markdown('<div class="section-header">Attendance Records</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: filter_date = st.date_input("Date", value=None)
    with c2:
        names = [r[0] for r in cursor.execute("SELECT DISTINCT name FROM attendance").fetchall()]
        filter_name = st.selectbox("Name", ["All"]+sorted(names))
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        today_only = st.checkbox("Today Only")

    q, p, conds = "SELECT name,date,time,status FROM attendance", [], []
    if today_only: conds.append("date=?"); p.append(datetime.date.today().strftime("%Y-%m-%d"))
    elif filter_date: conds.append("date=?"); p.append(filter_date.strftime("%Y-%m-%d"))
    if filter_name != "All": conds.append("name=?"); p.append(filter_name)
    if conds: q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY date DESC, time DESC"

    df = pd.DataFrame(cursor.execute(q,p).fetchall(), columns=["Name","Date","Time","Status"])
    st.markdown(f"**{len(df)} records**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1,c2 = st.columns(2)
    with c1: st.download_button("⬇️ Filtered CSV", df.to_csv(index=False).encode(), f"attendance_{datetime.date.today()}.csv", "text/csv", use_container_width=True)
    with c2:
        all_df = pd.DataFrame(cursor.execute("SELECT name,date,time,status FROM attendance").fetchall(), columns=["Name","Date","Time","Status"])
        st.download_button("⬇️ Full CSV", all_df.to_csv(index=False).encode(), "attendance_full.csv", "text/csv", use_container_width=True)
    if len(df):
        st.markdown("---")
        st.dataframe(df.groupby("Name").size().reset_index(name="Days Present"), use_container_width=True, hide_index=True)

# ═══ MANAGE ════════════════════════════════════════════════════════════════════
elif menu == "🗑️ Manage Data":
    st.markdown('<div class="section-header">Manage Data</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Delete Face**")
        files = [f for f in os.listdir("faces") if f.lower().endswith((".jpg",".jpeg",".png"))]
        if files:
            to_del = st.selectbox("Person", [os.path.splitext(f)[0] for f in files])
            if st.button("🗑️ Delete & Retrain"):
                for ext in [".jpg",".jpeg",".png"]:
                    try: os.remove(f"faces/{to_del}{ext}")
                    except: pass
                train_model()
                st.success(f"Deleted {to_del}"); st.rerun()
        else: st.info("No faces registered.")
    with c2:
        st.markdown("**Clear Records**")
        clear_name = st.text_input("Name (blank = clear ALL)")
        if st.button("⚠️ Clear"):
            if clear_name.strip(): cursor.execute("DELETE FROM attendance WHERE name=?", (clear_name.strip(),))
            else: cursor.execute("DELETE FROM attendance")
            conn.commit(); sync_csv()
            st.success("Cleared!"); st.rerun()
    st.markdown("---")
    if st.button("🔄 Retrain Model"):
        ok, count = train_model()
        st.success(f"Retrained with {count} person(s)!") if ok else st.error("No faces to train on.")
