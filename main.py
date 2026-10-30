from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from utils.read_instances import load_instance


INSTANCES_PATH = Path(__file__).resolve().parent / "instances"

def create_rooms_matrix(schools_df):
    max_rooms = max(len(capacities) for capacities in schools_df["capacidades_salas"])

    rooms_matrix = np.zeros((len(schools_df), max_rooms), dtype=int)

    for i, capacities in enumerate(schools_df["capacidades_salas"]):
        rooms_matrix[i, :len(capacities)] = capacities

    return rooms_matrix

def get_closest_candidates(exam_type, school_id, room_capacity, candidates_df, distances_df, allocated_candidates):
    candidates = candidates_df[
        (candidates_df["tipo_prova"] == exam_type)
        & (~candidates_df.index.isin(allocated_candidates))
    ].index

    column = str(school_id)

    return (
        distances_df.loc[candidates, column]
        .sort_values()
        .head(room_capacity)
        .index
        .tolist()
    )

def initial_solution(candidates_df, schools_df, distances_df, instance_name):
    allocated_candidates = set()
    allocation = []
    total_distance = 0
    
    rooms_matrix = create_rooms_matrix(schools_df)
    
    # print(rooms_matrix)
    
    ordered_distances = sorted(
        (
            (candidate_id, school_id, distance)
            for candidate_id, row in distances_df.iterrows()
            for school_id, distance in row.items()
        ),
        key=lambda x: x[2]
    )
    
    for candidate_id, school_id, distance in ordered_distances:
        if candidate_id in allocated_candidates:
            continue 
        
        school_idx = int(school_id) - 1
        exam_type = candidates_df.loc[candidate_id, "tipo_prova"]
        room_id = next(
            (i for i, capacity in enumerate(rooms_matrix[school_idx]) if capacity > 0),
            None
        )
        
        if room_id is None:
            continue
        
        allocated_candidates.add(candidate_id)
        allocation.append({
            "candidate_id": candidate_id,
            "school_id": school_id,
            "room_id": room_id
        })
        total_distance += distance
        room_capacity = rooms_matrix[school_idx][room_id] - 1
 
        closest_candidates = get_closest_candidates(
            exam_type,
            school_id,
            room_capacity,
            candidates_df,
            distances_df,
            allocated_candidates
        )
        
        for candidate in closest_candidates:
            allocated_candidates.add(candidate)
            allocation.append({
                "candidate_id": candidate,
                "school_id": school_id,
                "room_id": room_id
            })
            total_distance += distances_df.loc[candidate][school_id]
            
        rooms_matrix[school_idx][room_id] = 0
        
    if len(allocated_candidates) != len(candidates_df):
        missing_candidates = candidates_df.loc[
            ~candidates_df.index.isin(allocated_candidates)
        ]

        allocated_count = len(allocated_candidates)
        missing_count = len(missing_candidates)
        total_count = len(candidates_df)

        summary = pd.DataFrame({
            "Demanda": candidates_df["tipo_prova"].value_counts(),
            "Alocados": candidates_df.loc[
                candidates_df.index.isin(allocated_candidates), "tipo_prova"
            ].value_counts(),
            "Não alocados": missing_candidates["tipo_prova"].value_counts()
        }).fillna(0).astype(int)

        summary["% não alocados"] = (
            100 * summary["Não alocados"] / summary["Demanda"]
        )

        with open("failed_instances.txt", "a", encoding="utf-8") as f:
            f.write(f"INSTÂNCIA: {instance_name}\n")

            f.write(f"Candidatos: {total_count}\n")
            f.write(f"Alocados: {allocated_count}\n")
            f.write(f"Não alocados: {missing_count} ({missing_count / total_count:.2%})\n")
            f.write(f"Distância total: {total_distance:.4f}\n\n")

            f.write("Distribuição por tipo de prova:\n")
            f.write(summary.to_string(float_format=lambda x: f"{x:.2f}"))
            f.write("\n\nCandidatos não alocados:\n")

            for exam_type, group in missing_candidates.groupby("tipo_prova"):
                f.write(f"{exam_type}: {list(group.index)}\n")

            f.write("\n")

        return None
    
    allocation_df = pd.DataFrame(allocation)
    print(f"Total distance: {total_distance}")
    return allocation_df

@njit
def find_best_move(
    candidate_rows,
    allocation_types,
    school_indices,
    room_ids,
    distance_cols,
    distances,
    capacities,
    school_col_by_index,
    counts,
    room_types,
):
    n = len(candidate_rows)
    num_schools, max_rooms = capacities.shape

    best_delta = 0.0
    move_type = -1
    move_i = -1
    move_arg = -1
    move_room = -1

    # Movimento 1: transferências
    for i in range(n):
        source_school = school_indices[i]
        source_col = distance_cols[i]
        exam_type = allocation_types[i]
        current_distance = distances[candidate_rows[i], source_col]

        for s in range(num_schools):
            if s == source_school:
                continue

            target_col = school_col_by_index[s]

            delta = (
                distances[candidate_rows[i], target_col]
                - current_distance
            )

            if delta >= best_delta - 0.0001:
                continue

            for r in range(max_rooms):
                count = counts[s, r]

                if count == 0:
                    continue

                if count >= capacities[s, r]:
                    continue

                if room_types[s, r] != exam_type:
                    continue

                best_delta = delta
                move_type = 0
                move_i = i
                move_arg = s
                move_room = r

    # Movimento 2: trocas
    for i in range(n - 1):
        school_i = school_indices[i]
        room_i = room_ids[i]
        type_i = allocation_types[i]
        col_i = distance_cols[i]
        count_i = counts[school_i, room_i]

        for j in range(i + 1, n):
            school_j = school_indices[j]

            if school_i == school_j:
                continue

            room_j = room_ids[j]
            type_j = allocation_types[j]
            col_j = distance_cols[j]
            count_j = counts[school_j, room_j]

            # O candidato j precisa caber no cômodo i.
            if count_i > 1 and room_types[school_i, room_i] != type_j:
                continue

            # O candidato i precisa caber no cômodo j.
            if count_j > 1 and room_types[school_j, room_j] != type_i:
                continue

            delta = (
                distances[candidate_rows[i], col_j]
                + distances[candidate_rows[j], col_i]
                - distances[candidate_rows[i], col_i]
                - distances[candidate_rows[j], col_j]
            )

            if delta < best_delta - 0.0001:
                best_delta = delta
                move_type = 1
                move_i = i
                move_arg = j
                move_room = -1

    return best_delta, move_type, move_i, move_arg, move_room


def local_search(allocation_df, candidates_df, schools_df, distances_df):
    allocation_df = allocation_df.copy().reset_index(drop=True)

    candidate_ids = allocation_df["candidate_id"].to_numpy()
    school_ids = allocation_df["school_id"].to_numpy(copy=True)
    room_ids = allocation_df["room_id"].to_numpy(dtype=np.int64, copy=True)

    candidate_pos = {cid: i for i, cid in enumerate(candidates_df.index)}
    school_pos = {str(sid): i for i, sid in enumerate(distances_df.columns)}

    candidate_rows = np.array(
        [candidate_pos[cid] for cid in candidate_ids],
        dtype=np.int64,
    )

    type_codes, _ = __import__("pandas").factorize(
        candidates_df["tipo_prova"], sort=True
    )
    allocation_types = type_codes[candidate_rows].astype(np.int64)

    distances = distances_df.to_numpy(dtype=np.float64)

    distance_cols = np.array(
        [school_pos[str(sid)] for sid in school_ids],
        dtype=np.int64,
    )

    school_indices = np.array(
        [int(sid) - 1 for sid in school_ids],
        dtype=np.int64,
    )

    capacity_lists = [
        np.asarray(school["capacidades_salas"], dtype=np.int64)
        for _, school in schools_df.iterrows()
    ]

    max_rooms = max(len(c) for c in capacity_lists)
    capacities = np.zeros(
        (len(capacity_lists), max_rooms),
        dtype=np.int64,
    )

    for s, cap in enumerate(capacity_lists):
        capacities[s, :len(cap)] = cap

    # Mapeia cada escola (índice zero-based) para a coluna de distâncias.
    school_col_by_index = np.array(
        [school_pos[str(s + 1)] for s in range(len(capacity_lists))],
        dtype=np.int64,
    )

    n = len(candidate_ids)

    def build_room_state():
        counts = np.zeros_like(capacities)
        room_types = np.full(capacities.shape, -1, dtype=np.int64)

        for i in range(n):
            s = school_indices[i]
            r = room_ids[i]

            counts[s, r] += 1
            room_types[s, r] = allocation_types[i]

        return counts, room_types

    best_distance = distances[candidate_rows, distance_cols].sum()
    iteration = 0

    while True:
        iteration += 1

        print(
            f"Iteração {iteration} | "
            f"Distância atual: {best_distance:.4f}"
        )

        counts, room_types = build_room_state()

        (
            best_delta,
            move_type,
            move_i,
            move_arg,
            move_room,
        ) = find_best_move(
            candidate_rows,
            allocation_types,
            school_indices,
            room_ids,
            distance_cols,
            distances,
            capacities,
            school_col_by_index,
            counts,
            room_types,
        )

        if move_type == -1:
            print("Nenhuma melhoria encontrada.")
            break

        if move_type == 0:
            # Transferência para um cômodo ocupado de outra escola.
            i = move_i
            target_school = move_arg
            target_room = move_room

            school_indices[i] = target_school
            school_ids[i] = target_school + 1
            room_ids[i] = target_room
            distance_cols[i] = school_col_by_index[target_school]

        else:
            # Troca das escolas e dos cômodos entre dois candidatos.
            i = move_i
            j = move_arg

            school_indices[i], school_indices[j] = (
                school_indices[j],
                school_indices[i],
            )
            school_ids[i], school_ids[j] = (
                school_ids[j],
                school_ids[i],
            )
            room_ids[i], room_ids[j] = room_ids[j], room_ids[i]
            distance_cols[i], distance_cols[j] = (
                distance_cols[j],
                distance_cols[i],
            )

        best_distance += best_delta
        print(f"Melhoria: {-best_delta:.4f}")

    allocation_df["school_id"] = school_ids
    allocation_df["room_id"] = room_ids

    actual_distance = distances[candidate_rows, distance_cols].sum()
    print(f"Distância final: {actual_distance:.4f}")

    return allocation_df
            
def total_distance(allocation_df, distances_df):
    return sum(
        distances_df.loc[row.candidate_id, str(row.school_id)]
        for row in allocation_df.itertuples()
    )         
    
import time

def main():
    with open("results.txt", "w", encoding="utf-8") as log:
        log.write(
            "Instância | Distância inicial | Tempo heurística (s) | "
            "Distância local search | Tempo local search (s)\n"
        )

        for file in sorted(INSTANCES_PATH.glob("*.txt")):
            (
                num_candidates,
                num_schools,
                candidates_df,
                schools_df,
                distances_df,
            ) = load_instance(file)

            start = time.perf_counter()
            allocation_df = initial_solution(
                candidates_df, schools_df, distances_df, file.name
            )
            initial_time = time.perf_counter() - start

            # if allocation_df is not None:
            #     initial_distance = total_distance(
            #         allocation_df, distances_df
            #     )

            #     start = time.perf_counter()
            #     local_search_df = local_search(
            #         allocation_df, candidates_df, schools_df, distances_df
            #     )
            #     local_search_time = time.perf_counter() - start

            #     local_search_distance = total_distance(
            #         local_search_df, distances_df
            #     )

            #     log.write(
            #         f"{file.name} | {initial_distance:.4f} | "
            #         f"{initial_time:.4f} | {local_search_distance:.4f} | "
            #         f"{local_search_time:.4f}\n"
            #     )
            #     log.flush()

            #     print(
            #         f"{file.name}: inicial = {initial_distance:.4f} "
            #         f"({initial_time:.4f}s) | "
            #         f"local search = {local_search_distance:.4f} "
            #         f"({local_search_time:.4f}s)"
            #     )
                
                    
if __name__ == "__main__":
    main()