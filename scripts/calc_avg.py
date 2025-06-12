import os

TASK = "rte"
MODE = "10"

# Determine repo root from script location
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log_dir = os.path.join(ROOT, "logs", "run", TASK, MODE)
accuracy_list, f1_list = [], []
for filename in os.listdir(log_dir):
    if filename.endswith(".log"):
        with open(os.path.join(log_dir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 3:
                try:
                    accuracy_list.append(float(lines[-3].split("Acc:")[1]))
                    f1_list.append(float(lines[-1].split("F1:")[1]))
                except Exception:
                    continue

if accuracy_list:
    print("Acc:", accuracy_list)
    print("Acc:", sum(accuracy_list)/len(accuracy_list))
if f1_list:
    print("F1:", f1_list)
    print("F1:", sum(f1_list)/len(f1_list))

out_path = os.path.join(log_dir, "avg.txt")
with open(out_path, "w", encoding="utf-8") as f:
    if accuracy_list:
        f.write("Acc:" + str(accuracy_list) + "\n")
        f.write("Acc:" + str(sum(accuracy_list)/len(accuracy_list)) + "\n")
    if f1_list:
        f.write("F1:" + str(f1_list) + "\n")
        f.write("F1:" + str(sum(f1_list)/len(f1_list)) + "\n")
