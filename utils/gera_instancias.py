with open("teste.txt", "r") as f:
    capacidade_total = 0

    for linha in f:
        if linha.startswith("#") or not linha.strip():
            continue

        capacidades = linha.strip().split(", ")[5]
        capacidade_total += sum(map(int, capacidades.split(";")))

print(capacidade_total)