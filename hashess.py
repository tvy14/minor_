import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data = f'{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}'.encode()
        return hashlib.sha256(data).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "0", int(time.time()), "Genesis Block")

    def add_block(self, new_block):
        new_block.previous_hash = self.chain[-1].hash
        new_block.timestamp = int(time.time())
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True
#Put the FL , and add Consensus Mechanism here on

class WeightAggr:
    def __init__(self):
        self.weights = []

    def add_weight(self, weight):
        self.weights.append(weight)

    def aggregate(self):
        total_weight = sum(self.weights)
        aggregated_weight = sum(weight * block.index for weight, block in zip(self.weights, blockchain.chain))
        return aggregated_weight / total_weight

    