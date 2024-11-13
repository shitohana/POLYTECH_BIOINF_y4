import numpy as np

# Example HMM parameters
states = ['CpG', 'non-CpG']
observations = ['A', 'C', 'G', 'T']
state_to_idx = {'CpG': 0, 'non-CpG': 1}
obs_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Transition probabilities
trans_probs = np.array([[0.999, 0.001],  # From CpG
                        [0.005, 0.995]])  # From non-CpG

# Emission probabilities
emis_probs = np.array([[0.1, 0.4, 0.4, 0.1],  # CpG
                       [0.25, 0.25, 0.25, 0.25]])  # non-CpG

# Observed sequence (e.g., 'ACGT')
observed_sequence = 'ACGT'
obs_idx_sequence = [obs_to_idx[nuc] for nuc in observed_sequence]

# Number of states and sequence length
n_states = len(states)
seq_len = len(observed_sequence)

# Viterbi matrix to store max probabilities
V = np.zeros((n_states, seq_len))

# Backpointer matrix to store the best path
backpointer = np.zeros((n_states, seq_len), dtype=int)

# Initialization
for s in range(n_states):
    V[s, 0] = emis_probs[s, obs_idx_sequence[0]]
    backpointer[s, 0] = 0

# Iteration
for t in range(1, seq_len):
    for s in range(n_states):
        transition_probs = [V[prev_state, t - 1] * trans_probs[prev_state, s] for prev_state in range(n_states)]
        max_transition_prob = max(transition_probs)
        V[s, t] = max_transition_prob * emis_probs[s, obs_idx_sequence[t]]
        backpointer[s, t] = np.argmax(transition_probs)

# Termination: Finding the best last state
last_state = np.argmax(V[:, -1])
best_path = [last_state]

# Tracking back to find the best path
for t in range(seq_len - 1, 0, -1):
    last_state = backpointer[last_state, t]
    best_path.append(last_state)

# Reverse the path to get the correct order
best_path = best_path[::-1]

# Convert state indices back to state names
best_path_states = [states[state] for state in best_path]

# Output the result
print("Most probable sequence of states:")
print(best_path_states)
