import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read the network data from a structured text file
def read_network_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    networks = []
    current_network = {}
    is_connections = False

    for line in lines:
        line = line.strip()
        if line.startswith("Network"):
            if current_network:
                networks.append(current_network)
            current_network = {"neurons": [], "connections": []}
            continue
        if line.startswith("Neuron Type"):
            continue  # Skip header line
        if line.startswith("Connections"):
            is_connections = True
            continue
        if line.startswith("From Neuron"):
            continue
        if is_connections:
            # Process connection data
            connection_data = line.split(',')
            if len(connection_data) == 5:
                current_network["connections"].append({
                    "From Neuron Type": connection_data[0],
                    "From Neuron Bias": float(connection_data[1]),
                    "To Neuron Type": connection_data[2],
                    "To Neuron Bias": float(connection_data[3]),
                    "Weight": float(connection_data[4])
                })
        else:
            # Process neuron data
            neuron_data = line.split(',')
            if len(neuron_data) == 4:
                current_network["neurons"].append({
                    "Neuron Type": neuron_data[0],
                    "Bias": float(neuron_data[1]),
                    "Activation": float(neuron_data[2]),
                    "State": float(neuron_data[3])
                })

    if current_network:
        networks.append(current_network)

    return networks

# Read the network data
network_data = read_network_data('analyze.txt')

# Prepare data for visualization
neurons = []
connections = []

for network in network_data:
    for neuron in network["neurons"]:
        neurons.append(neuron)
    for connection in network["connections"]:
        connections.append(connection)

# Convert to DataFrames
neuron_df = pd.DataFrame(neurons)
connection_df = pd.DataFrame(connections)
# Box plot for Activation by Neuron Type
plt.figure(figsize=(10, 5))
sns.boxplot(x='Neuron Type', y='Activation', data=neuron_df)
plt.title('Activation Distribution by Neuron Type')
plt.ylabel('Activation')
plt.xlabel('Neuron Type')
plt.grid()
plt.show()

# Visualize Neuron Activations
plt.figure(figsize=(10, 5))
sns.barplot(x='Neuron Type', y='Activation', data=neuron_df)
plt.title('Neuron Activations')
plt.ylabel('Activation Value')
plt.xlabel('Neuron Type')
plt.show()

# Visualize Weights
plt.figure(figsize=(10, 5))
sns.scatterplot(x='From Neuron Bias', y='Weight', hue='To Neuron Bias', data=connection_df)
plt.title('Weights from Input Neurons')
plt.ylabel('Weight')
plt.xlabel('From Neuron Bias')
plt.legend(title='To Neuron Bias')
plt.show()