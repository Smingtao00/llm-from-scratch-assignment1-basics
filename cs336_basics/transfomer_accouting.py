vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

parameters = 2 * vocab_size * d_model + num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
print("parameters = ", parameters)
save = parameters * 4 / (1024 ** 3)
print(f"requires about {save} GB of memory")

total_FLOPs = num_layers * (8 * context_length * d_model * d_model + 4 * context_length * context_length * d_model + 6 * context_length * d_ff * d_model) + 2 * d_model * vocab_size
print(f"total_FLOPs is {total_FLOPs}")