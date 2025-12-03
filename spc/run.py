import subprocess
import sys
import os
import datetime

# --- Create log folder ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# --- Timestamp file name ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"log_{timestamp}.txt")

print(f"[RUNNER] Logging to {log_path}")

# Open logfile
with open(log_path, "w", encoding="utf-8") as log_file:
    # Start subprocess running main.py
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # line-buffered
    )

    # Read output line-by-line and tee into file + console
    for line in process.stdout:
        print(line, end="")     # show in console
        log_file.write(line)    # write to file
        log_file.flush()

    process.wait()

print("[RUNNER] Finished.")
