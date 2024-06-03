import numpy as np
import matplotlib.pyplot as plt
import time
import random


BW = 20 #MHz; can setup as 60 MHz in the presence of mmwave
NF = 7 #noise figure (dB)
BPS = random(0,64) #bits per symbol
SNR_mod = 10 * (log * (2^BPS) - 1)

def LinkBudget()
    #each modulation calculation
    RSL_mod = -174 + 10 * (log*BW) + NF + #imf + SNR_mod


class FronthaulOptim:
    def __init__(self, num_UEs, num_RBs, num_antennas, num_nodes):
        self.num_UEs = num_UEs
        self.num_RBs = num_RBs
        self.num_antennas = num_antennas
        self.num_nodes = num_nodes
        self.UEs = [i for i in range(num_UEs)]
        self.RBs = [i for i in range(num_RBs)]
        self.antennas = [i for i in range(num_antennas)]
        self.nodes = [i for i in range(num_nodes)]
        self.current_block = None

    throughput_tps = ###
    throughput_mbps = ###

    return throughput_tps, throughput_mbps

def plot_results(results):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot([result['num_nodes'] for result in results], [result['throughput_tps'] for result in results], label='PBFT')
    plt.xlabel('Number of nodes')
    plt.ylabel('Throughput (TPS)')
    plt.title('PBFT Throughput vs. Number of Nodes')

    plt.subplot(2, 2, 2)
    plt.plot([result['num_nodes'] for result in results], [result['throughput_mbps'] for result in results], label='PBFT')
    plt.xlabel('Number of nodes')
    plt.ylabel('Throughput (Mbps)')
    plt.title('PBFT Throughput (Mbps) vs. Number of Nodes')

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_transactions = 10000

    results = []
    for num_nodes in [10, 50, 100, 1000]:
        throughput_tps, throughput_mbps = simulate(num_nodes, num_transactions)

        results.append({
            'num_nodes': num_nodes,
            'throughput_tps': throughput_tps,
            'throughput_mbps': throughput_mbps
        })

    plot_results(results)
