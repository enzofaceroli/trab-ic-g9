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

def initial_solution(candidates_df, schools_df, distances_df):
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
        print(f"No valid solution found. {len(allocated_candidates)}/{len(candidates_df)} candidates allocated. Total Distance: {total_distance}\n")
        # unallocated = candidates_df.loc[
        #     ~candidates_df.index.isin(allocated_candidates)
        # ]

        # print(f"Não alocados: {len(unallocated)}")
        # print("Diagnóstico:")

        # print("\nCandidatos não alocados por tipo de prova:")
        # print(unallocated["tipo_prova"].value_counts().to_string())

        # print("\nCandidatos não alocados:")
        # print(f"{unallocated.to_string()}\n")
        
        # allocation_df = pd.DataFrame(
        #     allocation,
        #     columns=["candidate_id", "school_id", "room_id"]
        # ).merge(
        #     candidates_df[["tipo_prova"]],
        #     left_on="candidate_id",
        #     right_index=True
        # )

        # print("\nOcupação das salas:")
        # for (school_id, room_id, exam_type), count in allocation_df.groupby(
        #     ["school_id", "room_id", "tipo_prova"]
        # ).size().items():
        #     capacity = schools_df.iloc[int(school_id) - 1]["capacidades_salas"][room_id]
        #     print(
        #         f"Escola {school_id}, sala {room_id}: "
        #         f"{exam_type}, {count}/{capacity} candidatos"
        #     )

        # print("\nResumo por escola, sala e tipo de prova:")
        # print(
        #     allocation_df.groupby(
        #         ["school_id", "room_id", "tipo_prova"]
        #     ).size().to_string()
        # )

        return None
    
    allocation_df = pd.DataFrame(allocation)
    print(f"Total distance: {total_distance}")
    return allocation_df


import numpy as np
from numba import njit


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

            if delta >= best_delta:
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

            if delta < best_delta:
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


# def local_search(allocation_df, candidates_df, schools_df, distances_df):
#     allocation_df = allocation_df.copy().reset_index(drop=True)

#     candidate_ids = allocation_df["candidate_id"].to_numpy()
#     school_ids = allocation_df["school_id"].to_numpy(copy=True)
#     room_ids = allocation_df["room_id"].to_numpy(dtype=int, copy=True)

#     candidate_pos = {cid: i for i, cid in enumerate(candidates_df.index)}
#     school_pos = {str(sid): i for i, sid in enumerate(distances_df.columns)}

#     candidate_rows = np.array(
#         [candidate_pos[cid] for cid in candidate_ids], dtype=int
#     )
#     allocation_types = candidates_df["tipo_prova"].to_numpy()[
#         candidate_rows
#     ]

#     distances = distances_df.to_numpy(dtype=float)
#     distance_cols = np.array(
#         [school_pos[str(sid)] for sid in school_ids], dtype=int
#     )
#     school_indices = np.array(
#         [int(sid) - 1 for sid in school_ids], dtype=int
#     )

#     capacities = [
#         np.asarray(school["capacidades_salas"], dtype=int)
#         for _, school in schools_df.iterrows()
#     ]

#     n = len(candidate_ids)

#     counts = {}
#     types = {}

#     for i in range(n):
#         key = (school_indices[i], room_ids[i])
#         counts[key] = counts.get(key, 0) + 1
#         types[key] = allocation_types[i]

#     def objective():
#         return distances[candidate_rows, distance_cols].sum()

#     def can_enter(school, room, exam_type):
#         key = (school, room)

#         if room < 0 or room >= len(capacities[school]):
#             return False

#         if counts.get(key, 0) >= capacities[school][room]:
#             return False

#         return counts.get(key, 0) == 0 or types[key] == exam_type

#     def remove_candidate(i):
#         key = (school_indices[i], room_ids[i])
#         counts[key] -= 1

#         if counts[key] == 0:
#             del counts[key]
#             del types[key]

#     def add_candidate(i, school, room):
#         key = (school, room)

#         if counts.get(key, 0) == 0:
#             types[key] = allocation_types[i]

#         counts[key] = counts.get(key, 0) + 1

#     best_distance = objective()
#     iteration = 0

#     while True:
#         iteration += 1
#         move_found = False

#         print(f"Iteração {iteration} | Distância atual: {best_distance:.4f}")

#         occupied_rooms = list(counts.keys())

#         # Movimento 1: transferências
#         for i in range(n):
#             source_school = school_indices[i]
#             exam_type = allocation_types[i]
#             current_col = distance_cols[i]

#             for target_school, target_room in occupied_rooms:
#                 if target_school == source_school:
#                     continue

#                 if not can_enter(target_school, target_room, exam_type):
#                     continue

#                 target_col = school_pos[str(target_school + 1)]
#                 delta = (
#                     distances[candidate_rows[i], target_col]
#                     - distances[candidate_rows[i], current_col]
#                 )

#                 if delta < 0:
#                     remove_candidate(i)

#                     school_indices[i] = target_school
#                     school_ids[i] = target_school + 1
#                     room_ids[i] = target_room
#                     distance_cols[i] = target_col

#                     add_candidate(i, target_school, target_room)

#                     best_distance += delta
#                     move_found = True
#                     print(f"Melhoria: {-delta:.4f}")
#                     break

#             if move_found:
#                 break

#         if move_found:
#             continue

#         # Movimento 2: trocas vetorizadas
#         for i in range(n):
#             school_i = school_indices[i]
#             room_i = room_ids[i]
#             type_i = allocation_types[i]
#             col_i = distance_cols[i]

#             candidates_j = np.flatnonzero(
#                 (school_indices != school_i)
#                 & (np.arange(n) > i)
#             )

#             if len(candidates_j) == 0:
#                 continue

#             schools_j = school_indices[candidates_j]
#             rooms_j = room_ids[candidates_j]
#             types_j = allocation_types[candidates_j]
#             cols_j = distance_cols[candidates_j]

#             key_i = (school_i, room_i)
#             count_i = counts[key_i]

#             valid = np.ones(len(candidates_j), dtype=bool)

#             # Compatibilidade de tipo após a troca
#             if count_i > 1:
#                 valid &= types_j == types[key_i]

#             # A sala de j precisa aceitar o candidato i
#             for k, j in enumerate(candidates_j):
#                 key_j = (schools_j[k], rooms_j[k])

#                 if counts[key_j] > 1 and type_i != types[key_j]:
#                     valid[k] = False

#             if not valid.any():
#                 continue

#             # Delta de todas as trocas candidatas em uma operação NumPy
#             deltas = (
#                 distances[candidate_rows[i], cols_j]
#                 + distances[candidate_rows[candidates_j], col_i]
#                 - distances[candidate_rows[i], col_i]
#                 - distances[candidate_rows[candidates_j], cols_j]
#             )

#             deltas[~valid] = np.inf
#             improving = np.flatnonzero(deltas < 0)

#             if len(improving) == 0:
#                 continue

#             # First improvement: primeiro candidato válido que melhora
#             k = improving[0]
#             j = candidates_j[k]
#             delta = deltas[k]

#             key_i = (school_i, room_i)
#             key_j = (school_indices[j], room_ids[j])

#             # Atualiza as contagens das duas salas
#             counts[key_i] -= 1
#             counts[key_j] -= 1

#             if counts[key_i] == 0:
#                 del counts[key_i]
#                 del types[key_i]

#             if counts.get(key_j, 0) == 0:
#                 counts.pop(key_j, None)
#                 types.pop(key_j, None)

#             school_indices[i], school_indices[j] = (
#                 school_indices[j], school_indices[i]
#             )
#             school_ids[i], school_ids[j] = (
#                 school_ids[j], school_ids[i]
#             )
#             room_ids[i], room_ids[j] = room_ids[j], room_ids[i]
#             distance_cols[i], distance_cols[j] = (
#                 distance_cols[j], distance_cols[i]
#             )

#             # Registra os candidatos nas novas salas
#             add_candidate(i, school_indices[i], room_ids[i])
#             add_candidate(j, school_indices[j], room_ids[j])

#             best_distance += delta
#             move_found = True
#             print(f"Melhoria: {-delta:.4f}")
#             break

#         if not move_found:
#             print("Nenhuma melhoria encontrada.")
#             break

#     allocation_df["school_id"] = school_ids
#     allocation_df["room_id"] = room_ids

#     print(f"Distância final: {best_distance:.4f}")

#     return allocation_df
            
def total_distance(allocation_df, distances_df):
    return sum(
        distances_df.loc[row.candidate_id, str(row.school_id)]
        for row in allocation_df.itertuples()
    )         
    
def main():
    with open("results.txt", "w", encoding="utf-8") as log:
        log.write("Instância | Distância inicial | Distância local search\n")

        for file in sorted(INSTANCES_PATH.glob("*.txt")):
            (
                num_candidates,
                num_schools,
                candidates_df,
                schools_df,
                distances_df,
            ) = load_instance(file)

            allocation_df = initial_solution(
                candidates_df, schools_df, distances_df
            )

            if allocation_df is not None:
                initial_distance = total_distance(
                    allocation_df, distances_df
                )

                local_search_df = local_search(
                    allocation_df, candidates_df, schools_df, distances_df
                )

                local_search_distance = total_distance(
                    local_search_df, distances_df
                )

                log.write(
                    f"{file.name} | {initial_distance:.4f} | "
                    f"{local_search_distance:.4f}\n"
                )
                log.flush()

                print(
                    f"{file.name}: inicial = {initial_distance:.4f} | "
                    f"local search = {local_search_distance:.4f}"
                )
                
                    
if __name__ == "__main__":
    main()