import numpy as np
import matplotlib.pyplot as plt
import time
import random

def iterative_bargaining(vendors, initial_allocation, max_iterations=100):
    allocation = initial_allocation
    for _ in range(max_iterations):
        for vendor in vendors:
            # Calculate current utility
            current_utility = vendor.utility(allocation)
            # Propose new allocation
            proposed_allocation = vendor.propose_allocation(allocation)
            # Calculate new utility
            new_utility = vendor.utility(proposed_allocation)
            # Update allocation if new utility is better
            if new_utility > current_utility:
                allocation = proposed_allocation
    return allocation

vendors = [Vendor1(), Vendor2(), Vendor3()]
initial_allocation = [300, 300, 300]
optimal_allocation = iterative_bargaining(vendors, initial_allocation)
# Calculate Shannon Capacity
def shannon_capacity(bandwidth, signal_power, noise_power):
    capacity = bandwidth * np.log2(1 + (signal_power / noise_power))
    return capacity

# Set parameters
bandwidth = 20e6  # 20 MHz bandwidth
signal_power = 10**(114/10)
noise_power = 10**(-174/10)


start_time = time.time()
while time.time() - start_time < 24*60*60:
    
    throughput = []
    for vendor in vendors:
        vendor_signal_power = vendor.calculate_signal_power()
        vendor_capacity = shannon_capacity(bandwidth, vendor_signal_power, noise_power)
        throughput.append(vendor_capacity)
    
    # Update allocation based on network throughput
    max_throughput_vendor = np.argmax(throughput)
    allocation[max_throughput_vendor] += 1

print("Optimal Allocation:", optimal_allocation)