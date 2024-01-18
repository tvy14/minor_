import numpy as np

def create_dummy_models(num_models, model_size):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return [np.random.rand(model_size) for _ in range(num_models)]

def replace_global_model(local_models, global_model, malicious_model, n, kappa):
    """
    Replaces the global model with a malicious model using the strategy outlined in the equations.
    
    Args:
    local_models (list): List of local models.
    global_model (np.array): The current global model.
    malicious_model (np.array): The malicious model to introduce.
    n (int): Number of participants in federated learning.
    kappa (float): Scaling factor used by the attacker.
    
    Returns:
    np.array: The model to be submitted by the attacker.
    """
    # Sum of differences between local models and global model
    sum_differences = sum([model - global_model for model in local_models])

    # Calculate the new malicious model (Equation 2)
    new_malicious_model = global_model + (kappa / n) * sum_differences

    # Solve for the model to submit (Equation 3)
    attacker_model = ((n * kappa) / (n * kappa - 1)) * global_model - (1 / len(local_models)) * sum_differences
    
    return attacker_model

# Simulation Parameters
num_local_models = 5
model_size = 10  # Number of parameters in each model
global_model = np.random.rand(model_size)  # Global model
malicious_model = np.random.rand(model_size)  # Malicious model
n = 100  # Number of participants
kappa = 1.5  # Scaling factor

# Create dummy local models
local_models = create_dummy_models(num_local_models, model_size)

# Perform model replacement
attacker_model = replace_global_model(local_models, global_model, malicious_model, n, kappa)

print("Attacker Model:", attacker_model)
