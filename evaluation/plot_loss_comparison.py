import re
import matplotlib.pyplot as plt

def extract_loss(log_file):
    """从训练日志中提取 step 和 loss"""
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
    return steps, losses

def moving_average(data, window_size=100):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2)
        smoothed.append(sum(data[start:end]) / len(data[start:end]))
    return smoothed

m1_steps, m1_losses = extract_loss("training_log.err")
m2_steps, m2_losses = extract_loss("ablation_training_log.err")

m1_smoothed = moving_average(m1_losses, window_size=100)
m2_smoothed = moving_average(m2_losses, window_size=100)

print(f"M1: {len(m1_steps)} 个数据点, 最终 smoothed loss: {m1_smoothed[-1]:.4f}")
print(f"M2: {len(m2_steps)} 个数据点, 最终 smoothed loss: {m2_smoothed[-1]:.4f}")

plt.figure(figsize=(12, 7), dpi=300)

plt.plot(m1_steps, m1_losses, color='lightblue', alpha=0.3)
plt.plot(m1_steps, m1_smoothed, color='blue', linewidth=2,
         label='M1 - Enhanced (Ours)')

plt.plot(m2_steps, m2_losses, color='lightsalmon', alpha=0.3)
plt.plot(m2_steps, m2_smoothed, color='red', linewidth=2,
         label='M2 - Raw Low-light (Ablation)')

plt.title('Training Loss Comparison: M1 vs M2', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=13, loc='upper right')

plt.savefig('loss_curve_comparison.png', bbox_inches='tight')
print("Loss curve saved to 'loss_curve_comparison.png'")