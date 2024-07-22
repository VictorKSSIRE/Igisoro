import h5py
from tensorflow.keras.models import load_model

# Load the policy network model
policy_model = load_model('models/policy_network_epoch_10.h5')
value_model = load_model('models/value_network_epoch_10.h5')

# Print the model architecture
print(policy_model.summary())
print(value_model.summary())

# Access weights directly using h5py
with h5py.File('models/policy_network_epoch_10.h5', 'r') as f:
    print("Keys in policy network file:", list(f.keys()))
    print("Model weights:", list(f['model_weights'].keys()))

with h5py.File('models/value_network_epoch_10.h5', 'r') as f:
    print("Keys in value network file:", list(f.keys()))
    print("Model weights:", list(f['model_weights'].keys()))
