import numpy as np
import matplotlib.pyplot as plt

class PBFT:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.validators = [i for i in range(num_nodes)]
        self.current_block = None

    def validate_transaction(self, transaction):
        for validator in self.validators:
            if not validator.validate(transaction):
                return False
        return True

    def produce_block(self):
        transactions = []
        for validator in self.validators:
            transactions.extend(validator.get_pending_transactions())

        block = Block(transactions)

        for validator in self.validators:
            validator.add_block(block)

        self.current_block = block

    def get_throughput(self):
        return len(self.current_block.transactions)

class Block:
    def __init__(self, transactions):
        self.transactions = transactions

    def get_size(self):
        return sum([len(transaction) for transaction in self.transactions])

def simulate(num_nodes, num_transactions):
    consensus = PBFT(num_nodes)

    for i in range(num_transactions):
        transaction = Transaction(i)
        consensus.validate_transaction(transaction)

    consensus.produce_block()

    throughput_tps = consensus.get_throughput()
    throughput_mbps = throughput_tps * Block.get_size()

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
