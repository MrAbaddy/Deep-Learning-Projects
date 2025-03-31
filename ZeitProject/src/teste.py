import torch

# Verifica se o PyTorch pode usar GPU (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Criando tensores simples
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# Operações básicas
soma = x + y
produto = x * y

print(f"Soma: {soma}")
print(f"Produto: {produto}")

# Movendo tensor para GPU (se disponível)
if torch.cuda.is_available():
    x = x.to("cuda")
    print("Tensor movido para GPU!")
