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