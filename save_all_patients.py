# save_all_patients.py  — place this in your E:\Agentic\ folder
import subprocess, sys

for i in range(100):
    print(f"\n--- Patient {i+1}/100 ---")
    subprocess.run([sys.executable, "step5_predict.py", "--patient", str(i)])