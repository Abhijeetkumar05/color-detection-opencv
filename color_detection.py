import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import csv
import os

# Webcam
cap = cv2.VideoCapture(0)

# Kernel for noise removal
kernel = np.ones((5, 5), np.uint8)

# HSV Color Ranges
colors = {
    "Red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255))
    ],
    "Green": [((35, 50, 50), (85, 255, 255))],
    "Blue": [((100, 150, 50), (130, 255, 255))],
    "Yellow": [((20, 100, 100), (35, 255, 255))],
    "Orange": [((10, 100, 100), (20, 255, 255))],
    "Pink": [((145, 50, 150), (170, 255, 255))],
    "Purple": [((130, 50, 50), (145, 255, 255))],
    "Brown": [((5, 100, 20), (20, 255, 150))],
    "White": [((0, 0, 120), (180, 60, 255))],
    "Black": [((0, 0, 0), (180, 255, 50))]
}

# Drawing Colors (BGR)
draw_colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Orange": (0, 165, 255),
    "Pink": (203, 192, 255),
    "Purple": (255, 0, 255),
    "Brown": (19, 69, 139),
    "White": (255, 255, 255),
    "Black": (50, 50, 50)
}

# Matplotlib Colors (for graphs)
graph_colors = {
    "Red": "#FF4444",
    "Green": "#44BB44",
    "Blue": "#4444FF",
    "Yellow": "#FFD700",
    "Orange": "#FFA500",
    "Pink": "#FF69B4",
    "Purple": "#9400D3",
    "Brown": "#8B4513",
    "White": "#AAAAAA",
    "Black": "#333333"
}

# Color Detection Counter (frame count)
color_count = {c: 0 for c in colors}

# Unique detection tracking (to avoid re-counting same object per frame)
last_detected = {c: False for c in colors}
unique_count = {c: 0 for c in colors}

# Timeline log: (timestamp, color)
timeline_log = []

frame_number = 0
start_time = datetime.now()

print("=" * 50)
print("   Real-Time Multi Color Detection")
print("   Press Q to stop and generate graphs")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_number += 1

    detected_this_frame = {c: False for c in colors}

    for color_name, ranges in colors.items():
        mask = None
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            current_mask = cv2.inRange(hsv, lower, upper)
            if mask is None:
                mask = current_mask
            else:
                mask = cv2.bitwise_or(mask, current_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000:
                continue

            # Count every frame this color is visible
            color_count[color_name] += 1
            detected_this_frame[color_name] = True

            x, y, w, h = cv2.boundingRect(contour)
            box_color = draw_colors.get(color_name, (0, 255, 0))

            # Black border + colored rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 5)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

            # Label with area info
            label = f"{color_name} ({int(area)}px)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            break  # Count only largest contour per color per frame

    # Track unique detections (new appearance after absence)
    for color_name in colors:
        if detected_this_frame[color_name] and not last_detected[color_name]:
            unique_count[color_name] += 1
            elapsed = (datetime.now() - start_time).seconds
            timeline_log.append((elapsed, color_name))

    last_detected = detected_this_frame.copy()

    # Overlay: Title
    cv2.putText(frame, "Real-Time Multi Color Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Overlay: Date & Time
    current_datetime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(frame, current_datetime, (350, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Overlay: Live color count panel (bottom left)
    panel_y = frame.shape[0] - 200
    cv2.rectangle(frame, (0, panel_y), (220, frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, "Live Counts:", (5, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    row = 0
    for cname, cnt in color_count.items():
        if cnt > 0:
            bcolor = draw_colors[cname]
            cv2.putText(frame, f"{cname}: {cnt}", (5, panel_y + 40 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bcolor, 1)
            row += 1
            if row > 8:
                break

    cv2.imshow("Real-Time Multi Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ============================================================
# GRAPH GENERATION
# ============================================================

# Filter only detected colors
detected_colors = {c: v for c, v in color_count.items() if v > 0}
unique_detected = {c: v for c, v in unique_count.items() if v > 0}

if not detected_colors:
    print("No colors were detected. Please try again.")
    exit()

labels = list(detected_colors.keys())
counts = list(detected_colors.values())
bar_colors = [graph_colors[c] for c in labels]

u_labels = list(unique_detected.keys())
u_counts = list(unique_detected.values())
u_bar_colors = [graph_colors[c] for c in u_labels]

# ── FIGURE 1: Bar Graph (Frame Count) ──────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(labels, counts, color=bar_colors, edgecolor="black", linewidth=0.8)

for bar, val in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
             str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_title("Color Detection — Frame Count per Color", fontsize=16, fontweight="bold", pad=15)
ax1.set_xlabel("Detected Colors", fontsize=13)
ax1.set_ylabel("Number of Frames Detected", fontsize=13)
ax1.set_facecolor("#F5F5F5")
fig1.patch.set_facecolor("#FFFFFF")
ax1.grid(axis="y", linestyle="--", alpha=0.6)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("color_detection_bar_graph.png", dpi=150, bbox_inches="tight")
print("Saved: color_detection_bar_graph.png")

# ── FIGURE 2: Pie Chart ────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 9))
wedge_props = {"edgecolor": "white", "linewidth": 2}
wedges, texts, autotexts = ax2.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    colors=bar_colors,
    startangle=140,
    wedgeprops=wedge_props,
    pctdistance=0.82
)
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight("bold")

ax2.set_title("Color Distribution — Pie Chart", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("color_detection_pie_chart.png", dpi=150, bbox_inches="tight")
print("Saved: color_detection_pie_chart.png")

# ── FIGURE 3: Unique Detections Bar Chart ─────────────────
if u_labels:
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    bars3 = ax3.bar(u_labels, u_counts, color=u_bar_colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars3, u_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_title("Unique Color Appearances (New Detections)", fontsize=16, fontweight="bold", pad=15)
    ax3.set_xlabel("Color", fontsize=13)
    ax3.set_ylabel("Times Newly Appeared", fontsize=13)
    ax3.set_facecolor("#F5F5F5")
    fig3.patch.set_facecolor("#FFFFFF")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("color_detection_unique_bar.png", dpi=150, bbox_inches="tight")
    print("Saved: color_detection_unique_bar.png")

# ── FIGURE 4: Timeline / Horizontal Bar (Gantt-style) ─────
if timeline_log:
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    color_names_seen = list(dict.fromkeys([e[1] for e in timeline_log]))
    y_pos = {c: i for i, c in enumerate(color_names_seen)}

    for t, cname in timeline_log:
        ax4.scatter(t, y_pos[cname], color=graph_colors[cname],
                    s=120, edgecolors="black", linewidths=0.5, zorder=3)

    ax4.set_yticks(range(len(color_names_seen)))
    ax4.set_yticklabels(color_names_seen, fontsize=11)
    ax4.set_xlabel("Time (seconds from start)", fontsize=12)
    ax4.set_title("Color Detection Timeline", fontsize=16, fontweight="bold", pad=15)
    ax4.set_facecolor("#F0F0F0")
    fig4.patch.set_facecolor("#FFFFFF")
    ax4.grid(axis="x", linestyle="--", alpha=0.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("color_detection_timeline.png", dpi=150, bbox_inches="tight")
    print("Saved: color_detection_timeline.png")

# ── FIGURE 5: Summary Dashboard (2x2 combined) ────────────
fig5, axes = plt.subplots(2, 2, figsize=(16, 12))
fig5.suptitle("Color Detection — Full Summary Dashboard", fontsize=18, fontweight="bold", y=0.98)

# Top-left: Bar graph
ax = axes[0][0]
b = ax.bar(labels, counts, color=bar_colors, edgecolor="black", linewidth=0.7)
for bar, val in zip(b, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            str(val), ha="center", fontsize=8, fontweight="bold")
ax.set_title("Frame Count per Color", fontweight="bold")
ax.set_xlabel("Color")
ax.set_ylabel("Frames")
ax.set_facecolor("#F5F5F5")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis='x', rotation=30)

# Top-right: Pie chart
ax = axes[0][1]
ax.pie(counts, labels=labels, autopct="%1.1f%%", colors=bar_colors,
       startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Color Distribution", fontweight="bold")

# Bottom-left: Unique detections
ax = axes[1][0]
if u_labels:
    b2 = ax.bar(u_labels, u_counts, color=u_bar_colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(b2, u_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Unique Appearances", fontweight="bold")
    ax.set_xlabel("Color")
    ax.set_ylabel("Count")
    ax.set_facecolor("#F5F5F5")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', rotation=30)
else:
    ax.text(0.5, 0.5, "No unique data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Unique Appearances", fontweight="bold")

# Bottom-right: Stats table
ax = axes[1][1]
ax.axis("off")
table_data = [["Color", "Frames", "Unique", "% Share"]]
total = sum(counts)
for c in labels:
    pct = f"{color_count[c] / total * 100:.1f}%"
    table_data.append([c, color_count[c], unique_count[c], pct])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)

# Color header row
for j in range(4):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i, c in enumerate(labels):
    table[i + 1, 0].set_facecolor(graph_colors[c])
    table[i + 1, 0].set_text_props(
        color="white" if c not in ("White", "Yellow") else "black",
        fontweight="bold"
    )

ax.set_title("Detection Summary Table", fontweight="bold", pad=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("color_detection_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved: color_detection_dashboard.png")

# ── CSV Export ─────────────────────────────────────────────
with open("color_detection_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Color", "Frame Count", "Unique Detections", "Percentage"])
    total = sum(color_count.values())
    for c in colors:
        if color_count[c] > 0:
            pct = round(color_count[c] / total * 100, 2)
            writer.writerow([c, color_count[c], unique_count[c], f"{pct}%"])
print("Saved: color_detection_results.csv")

# ── Final Summary in Terminal ──────────────────────────────
print("\n" + "=" * 50)
print("         DETECTION SUMMARY")
print("=" * 50)
print(f"{'Color':<12} {'Frames':>10} {'Unique':>10} {'%':>8}")
print("-" * 50)
for c in colors:
    if color_count[c] > 0:
        pct = color_count[c] / total * 100
        print(f"{c:<12} {color_count[c]:>10} {unique_count[c]:>10} {pct:>7.1f}%")
print("=" * 50)
print("\nAll graphs and CSV saved successfully!")

plt.show()