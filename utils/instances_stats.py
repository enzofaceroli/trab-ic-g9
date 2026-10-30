import numpy as np
import pandas as pd
from pathlib import Path
from utils.read_instances import load_instance

def get_instance_stats(num_candidates, num_schools, candidates_df, schools_df, distances_df):
    capacities = [
        capacity
        for rooms in schools_df['capacidades_salas']
        for capacity in rooms
    ]

    demand = candidates_df['tipo_prova'].value_counts()

    nearest_distances = distances_df.to_numpy(dtype=float).min(axis=1)

    total_capacity = sum(capacities)

    return {
        'Candidatos': num_candidates,
        'Escolas': num_schools,
        'Salas': len(capacities),
        'Capacidade total': total_capacity,
        'Excedente (%)': 100 * (total_capacity - num_candidates) / num_candidates,
    }


def main():
    instances_path = Path(__file__).resolve().parent / "instances"
    rows = []

    for file in sorted(instances_path.glob("*.txt")):
        if "stats" in file.stem.lower():
            continue

        instance = load_instance(file)
        stats = get_instance_stats(*instance)

        rows.append({
            'Instância': file.stem,
            **stats
        })

    df = pd.DataFrame(rows)
    # print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    df.to_csv("tabela.csv", index=False, float_format="%.2f")


if __name__ == '__main__':
    main()