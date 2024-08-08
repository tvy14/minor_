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
initial_allocation = [10, 10, 10]
optimal_allocation = iterative_bargaining(vendors, initial_allocation)
print("Optimal Allocation:", optimal_allocation)