import matplotlib.pyplot as plt

# Read data from the text file
timestamps = []
correl_values = []
with open('correl.txt', 'r') as f:
    for line in f:
        line = line.strip().split()  # Split the line into timestamp and correl value
        timestamp = float(line[0])
        correl_value = float(line[1])
        timestamps.append(timestamp)
        correl_values.append(correl_value)

# Plot the data
plt.plot(timestamps, correl_values)
plt.xlabel('Time (seconds)')
plt.ylabel('Correlation Value')
plt.title('Correlation Value vs Time')
plt.grid(True)
plt.show()
