import re
import matplotlib.pyplot as plt

log_file = "training_log.err"

steps = []
losses = []

pattern = re.compile(r"(\d+)/10000\s+\[.*?loss=([0-9.]+)")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            if loss < 1.0: 
                steps.append(step)
                losses.append(loss)

def moving_average(data, window_size=50):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        smoothed.append(sum(data[start:end]) / len(data[start:end]))
    return smoothed

smoothed_losses = moving_average(losses, window_size=100)

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(steps, losses, color='lightgray', alpha=0.6, label='Raw Loss')
plt.plot(steps, smoothed_losses, color='blue', linewidth=2, label='Smoothed Loss')

plt.title('Training Loss of ControlNet (Low-Light Domain)', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

plt.savefig('loss_curve.png', bbox_inches='tight')
print("Loss curve has been saved to 'loss_curve.png'!")