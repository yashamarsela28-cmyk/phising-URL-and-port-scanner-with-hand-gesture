"""
╔══════════════════════════════════════════════════════════════════╗
║         AI CYBERSECURITY TOOL WITH GESTURE CONTROL              ║
║         Compatible: Python 3.11 + MediaPipe 0.10.35             ║
║         Run: py -3.11 gui_tool.py                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import socket
import urllib.parse
import datetime
import time
import re
import os
import urllib.request

# ─────────────────────────────────────────────
# THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────
import cv2
from PIL import Image, ImageTk  # pip install pillow

# ─────────────────────────────────────────────
# MEDIAPIPE TASKS API (NEW — 0.10.x compatible)
# ─────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers as mp_containers


# ─────────────────────────────────────────────
# OPTIONAL: validators library
# ─────────────────────────────────────────────
try:
    import validators
    HAS_VALIDATORS = True
except ImportError:
    HAS_VALIDATORS = False


# ══════════════════════════════════════════════
# CONSTANTS & THEME
# ══════════════════════════════════════════════
BG_DARK     = "#0a0e14"
BG_PANEL    = "#0f1923"
BG_CARD     = "#141d2b"
ACCENT_GREEN = "#00ff88"
ACCENT_CYAN  = "#00cfff"
ACCENT_RED   = "#ff3c5a"
ACCENT_AMBER = "#ffb347"
TEXT_PRIMARY = "#e0e8f0"
TEXT_DIM     = "#4a6080"
BORDER_COLOR = "#1e3050"

FONT_MONO  = ("Consolas", 10)
FONT_TITLE = ("Consolas", 13, "bold")
FONT_LABEL = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)

# Common short-URL domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "shorte.st", "bc.vc"
}

PHISHING_KEYWORDS = [
    "login", "signin", "sign-in", "secure", "account",
    "update", "confirm", "verify", "banking", "paypal",
    "ebay", "amazon", "apple", "microsoft", "google",
    "password", "credential", "wallet", "recover", "suspended"
]

ATTACK_TYPES = {
    "credential_phishing": "Credential Phishing — fake login page stealing user credentials",
    "fake_banking":        "Fake Banking Portal — impersonates financial institution",
    "redirect_attack":     "Redirect Attack — redirects victim through malicious chain",
    "malware_delivery":    "Malware Delivery — URL likely drops executable payload",
    "social_engineering":  "Social Engineering — psychological manipulation vector",
    "session_hijacking":   "Session Hijacking — targets active authenticated sessions",
}

PORT_INFO = {
    21:   ("FTP",   "File transfer — often exploited for anonymous access"),
    22:   ("SSH",   "Remote shell — brute-force target if exposed"),
    80:   ("HTTP",  "Unencrypted web — MITM and eavesdropping risk"),
    443:  ("HTTPS", "Encrypted web — check cert validity"),
    8080: ("HTTP-Alt", "Dev/proxy port — often left open accidentally"),
}

# Path to download the MediaPipe hand landmarker model
MODEL_PATH = "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ══════════════════════════════════════════════
# UTILITY: Download model if missing
# ══════════════════════════════════════════════
def ensure_model():
    """Download the HandLandmarker .task file if not present."""
    if os.path.exists(MODEL_PATH):
        return True
    try:
        print("[INFO] Downloading hand landmarker model (~9 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded successfully.")
        return True
    except Exception as e:
        print(f"[WARN] Could not download model: {e}")
        return False


# ══════════════════════════════════════════════
# GESTURE CLASSIFIER
# Uses landmark geometry to classify hand poses.
# Works with the NEW Tasks API landmark output.
# ══════════════════════════════════════════════
class GestureClassifier:
    """
    Classify hand gestures from MediaPipe HandLandmarker results.
    Landmarks: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    """

    # Landmark indices
    WRIST        = 0
    THUMB_TIP    = 4
    INDEX_TIP    = 8
    MIDDLE_TIP   = 12
    RING_TIP     = 16
    PINKY_TIP    = 20
    INDEX_MCP    = 5
    MIDDLE_MCP   = 9
    RING_MCP     = 13
    PINKY_MCP    = 17
    THUMB_IP     = 3
    THUMB_MCP    = 2
    INDEX_PIP    = 6
    MIDDLE_PIP   = 10
    RING_PIP     = 14
    PINKY_PIP    = 18

    def classify(self, hand_landmarks) -> str:
        """
        Given a list of 21 NormalizedLandmark objects, return gesture name.
        Returns: 'OPEN_PALM', 'THUMBS_UP', 'TWO_FINGERS', or 'UNKNOWN'
        """
        lm = hand_landmarks  # shorthand

        def tip_above_pip(tip_idx, pip_idx):
            return lm[tip_idx].y < lm[pip_idx].y

        index_up  = tip_above_pip(self.INDEX_TIP,  self.INDEX_PIP)
        middle_up = tip_above_pip(self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_up   = tip_above_pip(self.RING_TIP,   self.RING_PIP)
        pinky_up  = tip_above_pip(self.PINKY_TIP,  self.PINKY_PIP)

        # Thumb direction (up = tip higher than mcp)
        thumb_up_gesture = lm[self.THUMB_TIP].y < lm[self.THUMB_MCP].y

        # ── OPEN PALM: all fingers extended ──
        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM"

        # ── THUMBS UP: thumb up, other fingers curled ──
        if (thumb_up_gesture
                and not index_up
                and not middle_up
                and not ring_up
                and not pinky_up):
            return "THUMBS_UP"

        # ── TWO FINGERS: index + middle up, rest curled ──
        if index_up and middle_up and not ring_up and not pinky_up:
            return "TWO_FINGERS"

        return "UNKNOWN"


# ══════════════════════════════════════════════
# PHISHING URL ANALYZER
# ══════════════════════════════════════════════
class URLAnalyzer:
    """Perform heuristic phishing analysis on a URL string."""

    def analyze(self, url: str) -> dict:
        """
        Returns a dict with:
          score (0-100), risk_level, flags, attack_types, detail_lines
        """
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        flags   = []
        attacks = []
        score   = 0

        try:
            parsed = urllib.parse.urlparse(url)
            hostname = parsed.hostname or ""
            path     = parsed.path or ""
            query    = parsed.query or ""
            full     = url.lower()
        except Exception:
            return self._error_result()

        # ── Check 1: No HTTPS ──
        if parsed.scheme != "https":
            flags.append("❌ No HTTPS — connection is unencrypted")
            score += 15

        # ── Check 2: @ symbol in URL ──
        if "@" in url:
            flags.append("❌ @ symbol detected — possible URL deception trick")
            score += 20
            attacks.append("redirect_attack")

        # ── Check 3: IP address as domain ──
        ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
        if ip_pattern.match(hostname):
            flags.append("❌ IP address used instead of domain name")
            score += 25
            attacks.append("malware_delivery")

        # ── Check 4: Too many hyphens ──
        hyphen_count = hostname.count("-")
        if hyphen_count >= 3:
            flags.append(f"⚠️  Excessive hyphens in domain ({hyphen_count})")
            score += 10

        # ── Check 5: Suspicious subdomains ──
        parts = hostname.split(".")
        if len(parts) > 3:
            flags.append(f"⚠️  Deep subdomain structure: {hostname}")
            score += 10
            attacks.append("social_engineering")

        # ── Check 6: Long URL ──
        if len(url) > 100:
            flags.append(f"⚠️  Unusually long URL ({len(url)} chars)")
            score += 8

        # ── Check 7: Phishing keywords ──
        matched_kw = [kw for kw in PHISHING_KEYWORDS if kw in full]
        if matched_kw:
            flags.append(f"❌ Phishing keywords found: {', '.join(matched_kw)}")
            score += min(25, len(matched_kw) * 6)
            attacks.append("credential_phishing")
            if any(k in matched_kw for k in ["banking", "paypal", "ebay"]):
                attacks.append("fake_banking")

        # ── Check 8: URL shortener ──
        if hostname in URL_SHORTENERS:
            flags.append(f"⚠️  URL shortener detected ({hostname}) — real destination hidden")
            score += 15
            attacks.append("redirect_attack")

        # ── Check 9: Fake login path patterns ──
        fake_login_re = re.compile(r"/(login|signin|auth|wp-admin|account)", re.I)
        if fake_login_re.search(path):
            flags.append("❌ Fake login/admin path detected in URL")
            score += 12
            attacks.append("session_hijacking")

        # ── Check 10: Double-dot tricks ──
        if ".." in path:
            flags.append("⚠️  Directory traversal pattern (..) in path")
            score += 10

        # ── Check 11: Encoded characters (obfuscation) ──
        if "%" in url and url.count("%") > 3:
            flags.append("⚠️  Heavy URL encoding — possible obfuscation")
            score += 8

        # Normalize score
        score = min(score, 100)

        # Determine risk level
        if score <= 10:
            risk_level = "SAFE"
        elif score <= 30:
            risk_level = "LOW RISK"
        elif score <= 60:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "HIGH RISK"

        if not flags:
            flags.append("✅ No obvious phishing indicators detected")

        # Deduplicate attacks
        unique_attacks = list(dict.fromkeys(attacks))

        return {
            "url":          url,
            "score":        score,
            "risk_level":   risk_level,
            "flags":        flags,
            "attack_types": unique_attacks,
        }

    def _error_result(self):
        return {
            "url":          "INVALID",
            "score":        0,
            "risk_level":   "ERROR",
            "flags":        ["Could not parse URL"],
            "attack_types": [],
        }


# ══════════════════════════════════════════════
# PORT SCANNER
# ══════════════════════════════════════════════
def scan_ports(hostname: str, timeout: float = 0.8) -> list:
    """
    Scan a fixed list of common ports on hostname.
    Returns list of dicts: {port, service, status, risk}
    """
    results = []
    for port, (service, risk) in PORT_INFO.items():
        try:
            with socket.create_connection((hostname, port), timeout=timeout):
                status = "OPEN"
        except (socket.timeout, ConnectionRefusedError, OSError):
            status = "CLOSED"
        results.append({
            "port":    port,
            "service": service,
            "status":  status,
            "risk":    risk,
        })
    return results


# ══════════════════════════════════════════════
# MAIN APPLICATION CLASS
# ══════════════════════════════════════════════
class CyberSecApp:
    """Main tkinter application with gesture-controlled cybersecurity tools."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Cybersecurity Tool — Gesture Control")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1200x750")
        self.root.resizable(True, True)

        # ── State variables ──
        self.tool_locked    = True      # Tool starts locked
        self.gesture_label  = tk.StringVar(value="— No hand detected —")
        self.lock_label     = tk.StringVar(value="🔒  LOCKED")
        self.last_gesture   = ""
        self.gesture_cooldown = 0       # timestamp for debounce

        # ── Webcam ──
        self.cap              = None
        self.cam_running      = False
        self.cam_thread       = None
        self.current_frame_tk = None   # PhotoImage kept alive

        # ── Gesture classifier ──
        self.classifier   = GestureClassifier()
        self.landmarker   = None        # MediaPipe HandLandmarker
        self.model_ready  = False

        # ── Analyzers ──
        self.url_analyzer = URLAnalyzer()

        # ── Build UI ──
        self._build_ui()

        # ── Load model & start camera ──
        self.root.after(200, self._init_mediapipe)

        # ── Cleanup on close ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────
    def _build_ui(self):
        """Construct the full single-window layout."""

        # Title bar
        title_frame = tk.Frame(self.root, bg=BG_DARK, pady=6)
        title_frame.pack(fill="x", padx=10)
        tk.Label(
            title_frame,
            text="◈  AI CYBERSECURITY TOOL  ◈  GESTURE CONTROL",
            font=("Consolas", 14, "bold"),
            fg=ACCENT_GREEN, bg=BG_DARK
        ).pack(side="left")
        tk.Label(
            title_frame,
            textvariable=self.lock_label,
            font=("Consolas", 12, "bold"),
            fg=ACCENT_AMBER, bg=BG_DARK
        ).pack(side="right", padx=10)

        # Separator
        tk.Frame(self.root, bg=BORDER_COLOR, height=1).pack(fill="x", padx=8)

        # Main container
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        # ── LEFT PANEL ──
        left = tk.Frame(main, bg=BG_PANEL, bd=0, relief="flat",
                        highlightthickness=1, highlightbackground=BORDER_COLOR)
        left.pack(side="left", fill="y", padx=(0, 6), pady=0)
        left.pack_propagate(False)
        left.configure(width=390)

        self._build_left_panel(left)

        # ── RIGHT PANEL ──
        right = tk.Frame(main, bg=BG_PANEL, bd=0, relief="flat",
                         highlightthickness=1, highlightbackground=BORDER_COLOR)
        right.pack(side="left", fill="both", expand=True)

        self._build_right_panel(right)

        # Status bar at bottom
        status_bar = tk.Frame(self.root, bg=BG_DARK, pady=3)
        status_bar.pack(fill="x", padx=10)
        self.status_var = tk.StringVar(
            value="Initializing MediaPipe HandLandmarker…"
        )
        tk.Label(
            status_bar, textvariable=self.status_var,
            font=FONT_SMALL, fg=TEXT_DIM, bg=BG_DARK
        ).pack(side="left")
        tk.Label(
            status_bar,
            text=f"Python 3.11 · MediaPipe 0.10.x · {datetime.date.today()}",
            font=FONT_SMALL, fg=TEXT_DIM, bg=BG_DARK
        ).pack(side="right")

    def _build_left_panel(self, parent):
        """Webcam feed + gesture/lock status."""

        tk.Label(
            parent, text="[ WEBCAM FEED ]",
            font=FONT_LABEL, fg=ACCENT_CYAN, bg=BG_PANEL
        ).pack(pady=(8, 4))

        # Camera canvas
        self.cam_canvas = tk.Canvas(
            parent, width=360, height=270,
            bg="#050a10", highlightthickness=1,
            highlightbackground=ACCENT_CYAN
        )
        self.cam_canvas.pack(padx=12)
        self.cam_canvas.create_text(
            180, 135, text="Camera initializing…",
            fill=TEXT_DIM, font=FONT_LABEL
        )

        # Gesture status
        gesture_frame = tk.Frame(parent, bg=BG_CARD,
                                 highlightthickness=1,
                                 highlightbackground=BORDER_COLOR)
        gesture_frame.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(
            gesture_frame, text="DETECTED GESTURE",
            font=FONT_SMALL, fg=TEXT_DIM, bg=BG_CARD
        ).pack(pady=(6, 2))
        tk.Label(
            gesture_frame, textvariable=self.gesture_label,
            font=("Consolas", 11, "bold"),
            fg=ACCENT_GREEN, bg=BG_CARD
        ).pack(pady=(0, 6))

        # Lock status
        lock_frame = tk.Frame(parent, bg=BG_CARD,
                              highlightthickness=1,
                              highlightbackground=BORDER_COLOR)
        lock_frame.pack(fill="x", padx=12, pady=(4, 6))

        tk.Label(
            lock_frame, text="TOOL STATUS",
            font=FONT_SMALL, fg=TEXT_DIM, bg=BG_CARD
        ).pack(pady=(6, 2))
        self.lock_display = tk.Label(
            lock_frame, textvariable=self.lock_label,
            font=("Consolas", 12, "bold"),
            fg=ACCENT_AMBER, bg=BG_CARD
        )
        self.lock_display.pack(pady=(0, 6))

        # Gesture legend
        legend_frame = tk.Frame(parent, bg=BG_PANEL)
        legend_frame.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(
            legend_frame, text="── GESTURE LEGEND ──",
            font=FONT_SMALL, fg=TEXT_DIM, bg=BG_PANEL
        ).pack()
        gestures = [
            ("✋  OPEN PALM",    "Unlock tool"),
            ("👍  THUMBS UP",   "Start URL scan"),
            ("✌️   TWO FINGERS", "Clear results"),
        ]
        for g, desc in gestures:
            row = tk.Frame(legend_frame, bg=BG_PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=g,   font=FONT_SMALL, fg=ACCENT_CYAN,  bg=BG_PANEL, width=18, anchor="w").pack(side="left")
            tk.Label(row, text=desc, font=FONT_SMALL, fg=TEXT_PRIMARY, bg=BG_PANEL).pack(side="left")

        # Manual unlock button (fallback when camera unavailable)
        btn_frame = tk.Frame(parent, bg=BG_PANEL)
        btn_frame.pack(fill="x", padx=12, pady=(12, 4))
        tk.Button(
            btn_frame, text="[ MANUAL UNLOCK ]",
            font=FONT_SMALL, fg=BG_DARK, bg=ACCENT_AMBER,
            activebackground="#ffca70", relief="flat", cursor="hand2",
            command=self._manual_unlock
        ).pack(fill="x")
        tk.Button(
            btn_frame, text="[ MANUAL LOCK ]",
            font=FONT_SMALL, fg=BG_DARK, bg=ACCENT_RED,
            activebackground="#ff6070", relief="flat", cursor="hand2",
            command=self._manual_lock
        ).pack(fill="x", pady=(4, 0))

    def _build_right_panel(self, parent):
        """URL input, analysis controls, results."""

        # ── URL Input Section ──
        input_frame = tk.Frame(parent, bg=BG_CARD,
                               highlightthickness=1,
                               highlightbackground=BORDER_COLOR)
        input_frame.pack(fill="x", padx=12, pady=(10, 6))

        tk.Label(
            input_frame, text="◈ URL PHISHING ANALYZER + PORT SCANNER",
            font=FONT_TITLE, fg=ACCENT_CYAN, bg=BG_CARD
        ).pack(anchor="w", padx=10, pady=(8, 4))

        url_row = tk.Frame(input_frame, bg=BG_CARD)
        url_row.pack(fill="x", padx=10, pady=(0, 8))

        tk.Label(url_row, text="Target URL:", font=FONT_LABEL,
                 fg=TEXT_DIM, bg=BG_CARD).pack(side="left")

        self.url_entry = tk.Entry(
            url_row, font=FONT_MONO,
            bg="#0a1520", fg=ACCENT_GREEN,
            insertbackground=ACCENT_GREEN,
            relief="flat", bd=4,
            highlightthickness=1, highlightbackground=ACCENT_CYAN
        )
        self.url_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.url_entry.insert(0, "https://example.com")

        # Buttons row
        btn_row = tk.Frame(input_frame, bg=BG_CARD)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        self.analyze_btn = tk.Button(
            btn_row, text="▶  ANALYZE",
            font=("Consolas", 10, "bold"),
            fg=BG_DARK, bg=ACCENT_GREEN,
            activebackground="#00cc66",
            relief="flat", cursor="hand2", pady=4,
            command=self._run_analysis
        )
        self.analyze_btn.pack(side="left", padx=(0, 6))

        self.clear_btn = tk.Button(
            btn_row, text="✕  CLEAR",
            font=FONT_LABEL,
            fg=TEXT_PRIMARY, bg=BG_PANEL,
            activebackground=BORDER_COLOR,
            relief="flat", cursor="hand2", pady=4,
            command=self._clear_results
        )
        self.clear_btn.pack(side="left")

        self.scan_ports_btn = tk.Button(
            btn_row, text="◈  SCAN PORTS",
            font=FONT_LABEL,
            fg=BG_DARK, bg=ACCENT_CYAN,
            activebackground="#00aadd",
            relief="flat", cursor="hand2", pady=4,
            command=self._run_port_scan
        )
        self.scan_ports_btn.pack(side="left", padx=(6, 0))

        # ── Risk Level Display ──
        risk_outer = tk.Frame(parent, bg=BG_PANEL)
        risk_outer.pack(fill="x", padx=12, pady=(0, 6))

        self.risk_var   = tk.StringVar(value="—")
        self.score_var  = tk.StringVar(value="Score: —")

        risk_row = tk.Frame(risk_outer, bg=BG_PANEL)
        risk_row.pack(fill="x")

        tk.Label(risk_row, text="RISK LEVEL:", font=FONT_LABEL,
                 fg=TEXT_DIM, bg=BG_PANEL).pack(side="left")
        self.risk_label = tk.Label(
            risk_row, textvariable=self.risk_var,
            font=("Consolas", 12, "bold"),
            fg=ACCENT_AMBER, bg=BG_PANEL
        )
        self.risk_label.pack(side="left", padx=8)
        tk.Label(
            risk_row, textvariable=self.score_var,
            font=FONT_LABEL, fg=TEXT_DIM, bg=BG_PANEL
        ).pack(side="left")

        # Risk bar (canvas)
        self.risk_canvas = tk.Canvas(
            risk_outer, height=10, bg="#0a0e14",
            highlightthickness=0
        )
        self.risk_canvas.pack(fill="x", pady=(4, 0))

        # ── Scrollable Result Box ──
        result_label = tk.Label(
            parent, text="◈ ANALYSIS RESULTS",
            font=FONT_LABEL, fg=ACCENT_CYAN, bg=BG_PANEL
        )
        result_label.pack(anchor="w", padx=14, pady=(4, 2))

        result_outer = tk.Frame(parent, bg=BG_PANEL,
                                highlightthickness=1,
                                highlightbackground=BORDER_COLOR)
        result_outer.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.result_box = scrolledtext.ScrolledText(
            result_outer,
            font=FONT_MONO,
            bg="#070d14", fg=TEXT_PRIMARY,
            insertbackground=ACCENT_GREEN,
            relief="flat", bd=6,
            state="normal",
            wrap="word"
        )
        self.result_box.pack(fill="both", expand=True)

        # Configure text tags for coloring output
        self.result_box.tag_configure("header",  foreground=ACCENT_CYAN,  font=("Consolas", 10, "bold"))
        self.result_box.tag_configure("safe",     foreground=ACCENT_GREEN)
        self.result_box.tag_configure("low",      foreground="#88ff44")
        self.result_box.tag_configure("medium",   foreground=ACCENT_AMBER)
        self.result_box.tag_configure("high",     foreground=ACCENT_RED)
        self.result_box.tag_configure("flag",     foreground="#ff8800")
        self.result_box.tag_configure("attack",   foreground="#ff4488")
        self.result_box.tag_configure("open_port",foreground=ACCENT_RED)
        self.result_box.tag_configure("closed",   foreground=TEXT_DIM)
        self.result_box.tag_configure("dim",      foreground=TEXT_DIM)
        self.result_box.tag_configure("normal",   foreground=TEXT_PRIMARY)

        self._print_welcome()

    # ─────────────────────────────────────────
    # WELCOME MESSAGE
    # ─────────────────────────────────────────
    def _print_welcome(self):
        self._result_clear()
        lines = [
            ("╔══════════════════════════════════════╗\n", "header"),
            ("║  AI CYBERSECURITY TOOL — READY       ║\n", "header"),
            ("╚══════════════════════════════════════╝\n", "header"),
            ("\n", "normal"),
            ("  HOW TO USE:\n", "header"),
            ("  1. Show OPEN PALM to unlock the tool\n", "normal"),
            ("  2. Enter a URL in the input field\n", "normal"),
            ("  3. Show THUMBS UP or click ANALYZE\n", "normal"),
            ("  4. Show TWO FINGERS or click CLEAR\n", "normal"),
            ("\n", "normal"),
            ("  TOOL IS LOCKED — show open palm to begin\n", "medium"),
        ]
        for text, tag in lines:
            self.result_box.insert("end", text, tag)
        self.result_box.see("end")

    # ─────────────────────────────────────────
    # MEDIAPIPE INITIALIZATION
    # ─────────────────────────────────────────
    def _init_mediapipe(self):
        """Download model + initialize HandLandmarker, then start webcam."""
        def _worker():
            model_ok = ensure_model()
            if model_ok:
                try:
                    base_options = mp_python.BaseOptions(
                        model_asset_path=MODEL_PATH
                    )
                    options = mp_vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_hands=1,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
                    self.model_ready = True
                    self.root.after(0, lambda: self.status_var.set(
                        "HandLandmarker loaded ✓ — Camera starting…"
                    ))
                except Exception as e:
                    self.root.after(0, lambda: self.status_var.set(
                        f"MediaPipe error: {e}"
                    ))
            else:
                self.root.after(0, lambda: self.status_var.set(
                    "Model download failed — gesture control unavailable"
                ))

            # Start camera regardless
            self.root.after(0, self._start_camera)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ─────────────────────────────────────────
    # CAMERA LOOP
    # ─────────────────────────────────────────
    def _start_camera(self):
        """Open webcam and schedule periodic frame reads."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("No webcam detected — use manual unlock buttons")
            return
        self.cam_running = True
        self.status_var.set("Camera active ✓")
        self._read_frame()

    def _read_frame(self):
        """
        Read one frame from webcam, run gesture detection,
        update canvas, then schedule next frame via after().
        Using after() instead of a separate thread prevents tkinter crashes.
        """
        if not self.cam_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(100, self._read_frame)
            return

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run gesture detection
        gesture = "UNKNOWN"
        if self.model_ready and self.landmarker is not None:
            gesture = self._detect_gesture(rgb, frame)

        # Display frame on canvas
        self._update_canvas(frame)

        # Act on gesture (with debounce)
        now = time.time()
        if gesture != self.last_gesture or (now - self.gesture_cooldown) > 1.5:
            if gesture not in ("UNKNOWN", self.last_gesture):
                self.gesture_cooldown = now
                self._handle_gesture(gesture)
            self.last_gesture = gesture

        # Schedule next frame (~30 fps = 33 ms)
        self.root.after(33, self._read_frame)

    def _detect_gesture(self, rgb_frame, bgr_frame) -> str:
        """
        Run MediaPipe HandLandmarker on a single frame.
        Uses RunningMode.IMAGE (synchronous) — safe for tkinter after() calls.
        Returns gesture string.
        """
        try:
            h, w = rgb_frame.shape[:2]
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            result = self.landmarker.detect(mp_image)

            if not result.hand_landmarks:
                self.root.after(0, lambda: self.gesture_label.set("— No hand —"))
                return "UNKNOWN"

            landmarks = result.hand_landmarks[0]  # first hand

            # Draw landmarks on bgr_frame
            self._draw_landmarks(bgr_frame, landmarks, w, h)

            gesture = self.classifier.classify(landmarks)
            label_map = {
                "OPEN_PALM":   "✋  OPEN PALM",
                "THUMBS_UP":   "👍  THUMBS UP",
                "TWO_FINGERS": "✌️   TWO FINGERS",
                "UNKNOWN":     "— Gesture unknown —",
            }
            display = label_map.get(gesture, gesture)
            self.root.after(0, lambda d=display: self.gesture_label.set(d))
            return gesture

        except Exception as e:
            self.root.after(0, lambda: self.gesture_label.set("— Detection error —"))
            return "UNKNOWN"

    def _draw_landmarks(self, frame, landmarks, w, h):
        """Draw hand skeleton on the frame."""
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 210, 100), 1)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (0, 255, 180), -1)

    def _update_canvas(self, bgr_frame):
        """Resize frame and display on tkinter canvas."""
        try:
            rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            img  = img.resize((360, 270), Image.LANCZOS)
            self.current_frame_tk = ImageTk.PhotoImage(img)
            self.cam_canvas.create_image(
                0, 0, anchor="nw", image=self.current_frame_tk
            )
        except Exception:
            pass

    # ─────────────────────────────────────────
    # GESTURE HANDLER
    # ─────────────────────────────────────────
    def _handle_gesture(self, gesture: str):
        """React to a detected gesture."""
        if gesture == "OPEN_PALM":
            self._unlock_tool()
        elif gesture == "THUMBS_UP":
            if not self.tool_locked:
                self._run_analysis()
        elif gesture == "TWO_FINGERS":
            if not self.tool_locked:
                self._clear_results()

    def _unlock_tool(self):
        self.tool_locked = False
        self.lock_label.set("🔓  UNLOCKED")
        self.lock_display.configure(fg=ACCENT_GREEN)
        self.status_var.set("Tool UNLOCKED via gesture")

    def _lock_tool(self):
        self.tool_locked = True
        self.lock_label.set("🔒  LOCKED")
        self.lock_display.configure(fg=ACCENT_AMBER)
        self.status_var.set("Tool LOCKED")

    def _manual_unlock(self):
        self._unlock_tool()

    def _manual_lock(self):
        self._lock_tool()

    # ─────────────────────────────────────────
    # ANALYSIS RUNNERS
    # ─────────────────────────────────────────
    def _run_analysis(self):
        """Triggered by button or THUMBS UP gesture."""
        if self.tool_locked:
            self.status_var.set("Tool is LOCKED — show open palm to unlock")
            return
        url = self.url_entry.get().strip()
        if not url:
            self.status_var.set("Please enter a URL first")
            return
        self.status_var.set(f"Analyzing: {url} …")
        threading.Thread(target=self._analysis_worker, args=(url,), daemon=True).start()

    def _analysis_worker(self, url: str):
        """Run analysis in background thread, then update UI from main thread."""
        result = self.url_analyzer.analyze(url)
        self.root.after(0, lambda: self._display_analysis(result))

    def _run_port_scan(self):
        """Port scan triggered by button."""
        if self.tool_locked:
            self.status_var.set("Tool is LOCKED — show open palm to unlock")
            return
        url = self.url_entry.get().strip()
        if not url:
            self.status_var.set("Please enter a URL first")
            return
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        try:
            hostname = urllib.parse.urlparse(url).hostname or ""
        except Exception:
            hostname = ""
        if not hostname:
            self.status_var.set("Could not parse hostname from URL")
            return
        self.status_var.set(f"Scanning ports on {hostname} …")
        threading.Thread(
            target=self._port_scan_worker, args=(hostname,), daemon=True
        ).start()

    def _port_scan_worker(self, hostname: str):
        results = scan_ports(hostname)
        self.root.after(0, lambda: self._display_port_results(hostname, results))

    # ─────────────────────────────────────────
    # DISPLAY RESULTS
    # ─────────────────────────────────────────
    def _display_analysis(self, r: dict):
        """Write phishing analysis to the result box."""
        self._result_clear()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        score     = r["score"]
        risk      = r["risk_level"]

        # Choose risk color tag
        risk_tag = {
            "SAFE": "safe", "LOW RISK": "low",
            "MEDIUM RISK": "medium", "HIGH RISK": "high"
        }.get(risk, "normal")

        # Update risk widgets
        self.risk_var.set(risk)
        self.score_var.set(f"Score: {score}/100")
        self.risk_label.configure(fg={
            "safe": ACCENT_GREEN, "low": "#88ff44",
            "medium": ACCENT_AMBER, "high": ACCENT_RED
        }.get(risk_tag, ACCENT_AMBER))
        self._draw_risk_bar(score)

        # Write header
        self._rprint(f"╔{'═'*44}╗\n", "header")
        self._rprint(f"║  PHISHING ANALYSIS REPORT{'':>17}║\n", "header")
        self._rprint(f"║  {timestamp}{'':>14}║\n", "header")
        self._rprint(f"╚{'═'*44}╝\n\n", "header")

        self._rprint(f"  URL       : {r['url']}\n", "normal")
        self._rprint(f"  RISK LEVEL: ", "normal")
        self._rprint(f"{risk}\n", risk_tag)
        self._rprint(f"  RISK SCORE: {score} / 100\n\n", "normal")

        # Flags
        self._rprint("  ── DETECTION FLAGS ──\n", "header")
        for flag in r["flags"]:
            tag = "flag" if "❌" in flag or "⚠️" in flag else "safe"
            self._rprint(f"  {flag}\n", tag)

        # Attack types
        if r["attack_types"]:
            self._rprint("\n  ── POSSIBLE ATTACK TYPES ──\n", "header")
            for at in r["attack_types"]:
                desc = ATTACK_TYPES.get(at, at)
                self._rprint(f"  ⚠  {desc}\n", "attack")

        self._rprint("\n  ── RECOMMENDATIONS ──\n", "header")
        if score > 60:
            recs = [
                "  🛡  Do NOT enter credentials on this URL",
                "  🛡  Report this URL to your security team",
                "  🛡  Use a sandbox browser to investigate",
                "  🛡  Enable browser phishing protection",
            ]
        elif score > 30:
            recs = [
                "  ℹ  Proceed with caution",
                "  ℹ  Verify the sender/source before clicking",
                "  ℹ  Check SSL certificate manually",
            ]
        else:
            recs = [
                "  ✅  URL appears relatively safe",
                "  ℹ  Always stay vigilant online",
            ]
        for rec in recs:
            self._rprint(rec + "\n", "normal")

        self._rprint("\n  [Use SCAN PORTS button for port analysis]\n", "dim")
        self.result_box.see("end")
        self.status_var.set(f"Analysis complete: {risk} ({score}/100)")

    def _display_port_results(self, hostname: str, results: list):
        """Append port scan results to result box."""
        self._rprint(f"\n╔{'═'*44}╗\n", "header")
        self._rprint(f"║  PORT SCAN: {hostname[:30]:<30}  ║\n", "header")
        self._rprint(f"╚{'═'*44}╝\n\n", "header")

        open_count = 0
        for r in results:
            status_tag = "open_port" if r["status"] == "OPEN" else "closed"
            icon       = "◉" if r["status"] == "OPEN" else "○"
            self._rprint(
                f"  {icon} PORT {r['port']:5}  [{r['service']:<10}]  "
                f"{r['status']:<6}  — {r['risk']}\n",
                status_tag
            )
            if r["status"] == "OPEN":
                open_count += 1

        self._rprint(f"\n  Open ports: {open_count}/{len(results)}\n", "normal")
        if open_count > 2:
            self._rprint("  ⚠  Multiple open ports — reduce attack surface\n", "flag")
        self.result_box.see("end")
        self.status_var.set(f"Port scan complete: {open_count} open ports found")

    def _draw_risk_bar(self, score: int):
        """Draw a colored progress bar in the risk canvas."""
        self.risk_canvas.update_idletasks()
        w = self.risk_canvas.winfo_width()
        if w < 10:
            w = 600
        filled = int(w * score / 100)
        color  = ACCENT_GREEN if score <= 30 else (ACCENT_AMBER if score <= 60 else ACCENT_RED)
        self.risk_canvas.delete("all")
        self.risk_canvas.create_rectangle(0, 0, w, 10, fill="#0a1520", outline="")
        self.risk_canvas.create_rectangle(0, 0, filled, 10, fill=color, outline="")

    # ─────────────────────────────────────────
    # RESULT BOX HELPERS
    # ─────────────────────────────────────────
    def _rprint(self, text: str, tag: str = "normal"):
        self.result_box.insert("end", text, tag)

    def _result_clear(self):
        self.result_box.delete("1.0", "end")

    def _clear_results(self):
        self._result_clear()
        self.risk_var.set("—")
        self.score_var.set("Score: —")
        self.risk_canvas.delete("all")
        self.status_var.set("Results cleared")
        self._print_welcome()

    # ─────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────
    def _on_close(self):
        """Cleanly shut down camera and landmark detector."""
        self.cam_running = False
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()
        self.root.destroy()


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = CyberSecApp(root)
    root.mainloop()