
import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import sqlite3
import datetime
import pandas as pd
from PIL import Image, ImageDraw
import io

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎯",
    layout="wide",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: #0a0a0f;
        color: #e0e0e0;
    }

    section[data-testid="stSidebar"] {
        background: #111118;
        border-right: 1px solid #1e1e2e;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #555580;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00f5a0;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.8rem;
        color: #555580;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    .success-badge {
        background: #0d2b1e;
        border: 1px solid #00f5a0;
        color: #00f5a0;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }

    .warn-badge {
        background: #2b2000;
        border: 1px solid #f5c400;
        color: #f5c400;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }

    .error-badge {
        background: #2b0000;
        border: 1px solid #f55;
        color: #f55;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00f5a0, #00d9f5);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.85;
        transform: translateY(-1px);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        overflow: hidden;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #00f5a0;
        border-left: 3px solid #00f5a0;
        padding-left: 0.8rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── SETUP ─────────────────────────────────────────────────────────────────────
os.makedirs("faces", exist_ok=True)
CSV_FILE = "attendance.csv"

conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT,
    status TEXT DEFAULT 'Present'
)
""")
conn.commit()


# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def sync_csv():
    """Sync SQLite → CSV after every change."""
    rows = cursor.execute("SELECT name, date, time, status FROM attendance ORDER BY date DESC, time DESC").fetchall()
    df = pd.DataFrame(rows, columns=["Name", "Date", "Time", "Status"])
    df.to_csv(CSV_FILE, index=False)


def load_known_faces():
    encodings, names = [], []
    if not os.path.exists("faces"):
        return encodings, names
    for file in os.listdir("faces"):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join("faces", file)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            encodings.append(enc[0])
            names.append(os.path.splitext(file)[0])
    return encodings, names


def already_marked(name, date):
    """Check if attendance already marked for today."""
    result = cursor.execute(
        "SELECT id FROM attendance WHERE name=? AND date=?", (name, date)
    ).fetchone()
    return result is not None


def draw_face_boxes(image_np, face_locations, names):
    """Draw bounding boxes and names on the image."""
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)
    for (top, right, bottom, left), name in zip(face_locations, names):
        color = "#00f5a0" if name != "Unknown" else "#ff5555"
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.rectangle([left, bottom - 25, right, bottom], fill=color)
        draw.text((left + 6, bottom - 22), name, fill="#000000")
    return np.array(pil_img)


def get_stats():
    total = cursor.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_count = cursor.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    unique_people = cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance").fetchone()[0]
    registered = len([f for f in os.listdir("faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.exists("faces") else 0
    return total, today_count, unique_people, registered


# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎯 Face Attendance System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered attendance tracking — automatic CSV export on every scan</div>', unsafe_allow_html=True)

# Stats bar
total, today_count, unique_people, registered = get_stats()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{registered}</div><div class="stat-label">Registered Faces</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{today_count}</div><div class="stat-label">Present Today</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{unique_people}</div><div class="stat-label">Unique Students</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{total}</div><div class="stat-label">Total Records</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
menu = st.sidebar.selectbox(
    "📂 Navigation",
    ["🧑 Register Face", "📷 Mark Attendance", "📊 View Attendance", "🗑️ Manage Data"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# REGISTER FACE
# ═══════════════════════════════════════════════════════════════════════════════
if menu == "🧑 Register Face":
    st.markdown('<div class="section-header">Register New Face</div>', unsafe_allow_html=True)

    col_form, col_info = st.columns([2, 1])

    with col_form:
        name = st.text_input("👤 Full Name", placeholder="e.g. Raja Chauhan")
        image = st.camera_input("📸 Capture Face Photo")

        if image is not None and name.strip() != "":
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb)
            if len(encodings) == 0:
                st.markdown('<div class="error-badge">❌ No face detected. Please try again with better lighting.</div>', unsafe_allow_html=True)
            else:
                file_path = f"faces/{name.strip()}.jpg"
                with open(file_path, "wb") as f:
                    image.seek(0)
                    f.write(image.getbuffer())
                st.markdown(f'<div class="success-badge">✅ {name.strip()} registered successfully!</div>', unsafe_allow_html=True)
                st.image(rgb, caption=f"Registered: {name.strip()}", width=300)

    with col_info:
        st.info("**Tips for best results:**\n\n• Good lighting on face\n• Look directly at camera\n• Single face in frame\n• Clear, unobstructed view")

        if os.path.exists("faces"):
            existing = [os.path.splitext(f)[0] for f in os.listdir("faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if existing:
                st.markdown("**Registered People:**")
                for person in sorted(existing):
                    st.markdown(f"• {person}")


# ═══════════════════════════════════════════════════════════════════════════════
# MARK ATTENDANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "📷 Mark Attendance":
    st.markdown('<div class="section-header">Mark Attendance</div>', unsafe_allow_html=True)

    known_encodings, known_names = load_known_faces()

    if len(known_encodings) == 0:
        st.warning("⚠️ No faces registered yet. Please register faces first.")
    else:
        st.info(f"📋 {len(known_names)} people registered: {', '.join(known_names)}")

        image = st.camera_input("📷 Scan Face for Attendance")

        if image is not None:
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb)
            face_encodings_list = face_recognition.face_encodings(rgb, face_locations)

            if len(face_locations) == 0:
                st.markdown('<div class="error-badge">❌ No face detected in frame. Please try again.</div>', unsafe_allow_html=True)
            else:
                detected_names = []
                now = datetime.datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                for face_encoding in face_encodings_list:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_idx = np.argmin(face_distances)
                        if matches[best_idx]:
                            name = known_names[best_idx]

                    detected_names.append(name)

                # Draw bounding boxes
                annotated = draw_face_boxes(rgb, face_locations, detected_names)
                st.image(annotated, caption="Face Detection Result", use_column_width=True)

                # Process results
                st.markdown("### Results:")
                marked_any = False
                for name in detected_names:
                    if name != "Unknown":
                        if already_marked(name, date_str):
                            st.markdown(f'<div class="warn-badge">⚠️ {name} — Already marked present today ({date_str})</div><br>', unsafe_allow_html=True)
                        else:
                            cursor.execute(
                                "INSERT INTO attendance (name, date, time, status) VALUES (?,?,?,?)",
                                (name, date_str, time_str, "Present")
                            )
                            conn.commit()
                            sync_csv()  # ← Auto-save to CSV
                            st.markdown(f'<div class="success-badge">✅ {name} — Attendance marked at {time_str}</div><br>', unsafe_allow_html=True)
                            marked_any = True
                    else:
                        st.markdown('<div class="error-badge">❌ Unknown face — Not registered in system</div><br>', unsafe_allow_html=True)

                if marked_any:
                    st.caption("📁 attendance.csv updated automatically")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW ATTENDANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "📊 View Attendance":
    st.markdown('<div class="section-header">Attendance Records</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_date = st.date_input("Filter by Date", value=None)
    with col_f2:
        all_names = [row[0] for row in cursor.execute("SELECT DISTINCT name FROM attendance").fetchall()]
        filter_name = st.selectbox("Filter by Name", ["All"] + sorted(all_names))
    with col_f3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_today = st.checkbox("Today Only", value=False)

    # Build query
    query = "SELECT name, date, time, status FROM attendance"
    params = []
    conditions = []

    if show_today:
        conditions.append("date = ?")
        params.append(datetime.date.today().strftime("%Y-%m-%d"))
    elif filter_date:
        conditions.append("date = ?")
        params.append(filter_date.strftime("%Y-%m-%d"))

    if filter_name != "All":
        conditions.append("name = ?")
        params.append(filter_name)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY date DESC, time DESC"

    rows = cursor.execute(query, params).fetchall()
    df = pd.DataFrame(rows, columns=["Name", "Date", "Time", "Status"])

    st.markdown(f"**{len(df)} records found**")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download buttons
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        csv_filtered = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered CSV",
            csv_filtered,
            f"attendance_filtered_{datetime.date.today()}.csv",
            "text/csv",
            use_container_width=True
        )
    with col_d2:
        # Full CSV download
        all_rows = cursor.execute("SELECT name, date, time, status FROM attendance ORDER BY date DESC").fetchall()
        df_all = pd.DataFrame(all_rows, columns=["Name", "Date", "Time", "Status"])
        csv_all = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full CSV",
            csv_all,
            "attendance_full.csv",
            "text/csv",
            use_container_width=True
        )

    # Summary table
    if len(df) > 0:
        st.markdown("---")
        st.markdown("**Attendance Summary**")
        summary = df.groupby("Name").size().reset_index(name="Days Present")
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGE DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif menu == "🗑️ Manage Data":
    st.markdown('<div class="section-header">Manage Data</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Delete a Person's Face**")
        if os.path.exists("faces"):
            face_files = [f for f in os.listdir("faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if face_files:
                to_delete = st.selectbox("Select Person", [os.path.splitext(f)[0] for f in face_files])
                if st.button("🗑️ Delete Face Registration"):
                    os.remove(f"faces/{to_delete}.jpg")
                    st.success(f"Deleted face for {to_delete}")
                    st.rerun()
            else:
                st.info("No faces registered.")

    with col_m2:
        st.markdown("**Clear Attendance Records**")
        clear_name = st.text_input("Name to clear (leave blank = clear ALL)")
        if st.button("⚠️ Clear Records"):
            if clear_name.strip():
                cursor.execute("DELETE FROM attendance WHERE name=?", (clear_name.strip(),))
                conn.commit()
                sync_csv()
                st.success(f"Cleared records for {clear_name.strip()}")
            else:
                cursor.execute("DELETE FROM attendance")
                conn.commit()
                sync_csv()
                st.success("All attendance records cleared")
            st.rerun()
