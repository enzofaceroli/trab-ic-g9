from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from utils.read_instances import load_instance


INSTANCES_PATH = Path(__file__).resolve().parent / "instances"

def initial_solution(candidates_df, schools_df, distances_df):
    if candidates_df.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "exam_type",
                "school_id",
                "classroom_id",
                "distance",
            ]
        )

    if schools_df.empty:
        raise ValueError("Não existem escolas na instância.")

    # Use row and column positions for distances.
    # Candidate and school IDs are kept only for the solution.
    distance_matrix = distances_df.to_numpy(dtype=float)

    if distance_matrix.shape != (len(candidates_df), len(schools_df)):
        raise ValueError(
            "Dimensões incompatíveis: "
            f"distâncias={distance_matrix.shape}, "
            f"candidatos={len(candidates_df)}, "
            f"escolas={len(schools_df)}."
        )

    if not np.isfinite(distance_matrix).all():
        raise ValueError("A matriz de distâncias contém valores inválidos.")

    candidate_ids = candidates_df.index.tolist()
    school_ids = schools_df["id"].tolist()

    if not candidates_df.index.is_unique:
        raise ValueError("Os índices dos candidatos devem ser únicos.")

    if not schools_df["id"].is_unique:
        raise ValueError("Os IDs das escolas devem ser únicos.")

    # School capacities, preserving the original school order.
    school_rooms = {}
    school_capacity = {}

    for _, row in schools_df.iterrows():
        school_id = row["id"]
        capacities = [int(c) for c in row["capacidades_salas"]]

        if any(c < 0 for c in capacities):
            raise ValueError(
                f"Capacidade negativa encontrada na escola {school_id}."
            )

        school_rooms[school_id] = capacities
        school_capacity[school_id] = sum(capacities)

    # Candidate positions grouped by exam type.
    candidates_by_type = {}

    for pos, exam_type in enumerate(candidates_df["tipo_prova"].tolist()):
        candidates_by_type.setdefault(exam_type, []).append(pos)

    demand = {
        exam_type: len(positions)
        for exam_type, positions in candidates_by_type.items()
    }

    # Process the largest demands first.
    exam_types = tuple(
        sorted(demand, key=lambda t: demand[t], reverse=True)
    )

    school_positions = {
        school_id: pos
        for pos, school_id in enumerate(school_ids)
    }

    # Mean distance from each exam type to each school.
    mean_distance = {}

    for exam_type, positions in candidates_by_type.items():
        for school_id in school_ids:
            school_pos = school_positions[school_id]

            mean_distance[exam_type, school_id] = float(
                distance_matrix[positions, school_pos].mean()
            )

    # Search for a feasible disjoint assignment of schools to exam types.
    # This version dedicates each school to at most one exam type.
    @lru_cache(maxsize=None)
    def assign_schools(i, available):
        if i == len(exam_types):
            return ()

        exam_type = exam_types[i]
        required = demand[exam_type]
        options = []

        for size in range(1, len(available) + 1):
            for subset in combinations(available, size):
                capacity = sum(school_capacity[s] for s in subset)

                if capacity < required:
                    continue

                proximity = sum(
                    mean_distance[exam_type, s]
                    for s in subset
                )

                options.append(
                    (
                        capacity - required,
                        proximity,
                        size,
                        subset,
                    )
                )

        options.sort()

        for _, _, _, subset in options:
            selected = set(subset)
            remaining = tuple(
                school_id
                for school_id in available
                if school_id not in selected
            )

            result = assign_schools(i + 1, remaining)

            if result is not None:
                return ((exam_type, subset),) + result

        return None

    school_assignment = assign_schools(0, tuple(school_ids))

    if school_assignment is None:
        return None

    schools_by_type = {
        exam_type: list(selected_schools)
        for exam_type, selected_schools in school_assignment
    }

    # Assign each candidate to a school of the corresponding exam type.
    candidate_school = {}
    remaining_capacity = school_capacity.copy()

    for exam_type in exam_types:
        positions = candidates_by_type[exam_type]
        eligible_schools = schools_by_type[exam_type]

        pairs = sorted(
            (
                (
                    distance_matrix[
                        candidate_pos,
                        school_positions[school_id],
                    ],
                    candidate_pos,
                    school_id,
                )
                for candidate_pos in positions
                for school_id in eligible_schools
            ),
            key=lambda item: (item[0], item[1]),
        )

        for distance, candidate_pos, school_id in pairs:
            if candidate_pos in candidate_school:
                continue

            if remaining_capacity[school_id] <= 0:
                continue

            candidate_school[candidate_pos] = (
                school_id,
                float(distance),
            )
            remaining_capacity[school_id] -= 1

        unallocated = [
            pos for pos in positions
            if pos not in candidate_school
        ]

        if unallocated:
            raise RuntimeError(
                f"Não foi possível alocar {len(unallocated)} candidatos "
                f"da prova {exam_type}."
            )

    # Assign candidates to individual rooms.
    allocations = []

    for school_id in school_ids:
        school_candidates = [
            (pos, *candidate_school[pos])
            for pos in range(len(candidates_df))
            if candidate_school[pos][0] == school_id
        ]

        if not school_candidates:
            continue

        school_candidates.sort(key=lambda item: item[2])
        position = 0

        for room_id, capacity in enumerate(
            school_rooms[school_id], start=1
        ):
            room_candidates = school_candidates[
                position:position + capacity
            ]

            for candidate_pos, _, distance in room_candidates:
                allocations.append({
                    "candidate_id": candidate_ids[candidate_pos],
                    "exam_type": candidates_df.iloc[
                        candidate_pos
                    ]["tipo_prova"],
                    "school_id": school_id,
                    "classroom_id": room_id,
                    "distance": distance,
                })

            position += capacity

        if position < len(school_candidates):
            raise RuntimeError(
                f"A capacidade das salas da escola {school_id} "
                "não comporta os candidatos atribuídos."
            )

    allocation_df = pd.DataFrame(allocations)

    # Final consistency checks.
    if len(allocation_df) != len(candidates_df):
        raise RuntimeError(
            f"Foram alocados {len(allocation_df)} de "
            f"{len(candidates_df)} candidatos."
        )

    if allocation_df["candidate_id"].duplicated().any():
        raise RuntimeError("Há candidatos alocados mais de uma vez.")

    room_occupancy = allocation_df.groupby(
        ["school_id", "classroom_id"]
    ).size()

    for (school_id, room_id), occupancy in room_occupancy.items():
        capacity = school_rooms[school_id][room_id - 1]

        if occupancy > capacity:
            raise RuntimeError(
                f"A sala {room_id} da escola {school_id} "
                "ultrapassou sua capacidade."
            )

    return allocation_df

def local_search(initial_solution, distances_df, schools_df):
    import numpy as np

    allocation = initial_solution.copy().reset_index(drop=True)

    school_ids = schools_df["id"].tolist()
    n = len(allocation)

    room_capacities = {
        row["id"]: np.asarray(row["capacidades_salas"], dtype=int)
        for _, row in schools_df.iterrows()
    }
    school_capacity = {
        school: capacities.sum()
        for school, capacities in room_capacities.items()
    }

    school_columns = {}
    for school in school_ids:
        column = school if school in distances_df.columns else str(school)[1:]
        if column not in distances_df.columns:
            column = int(column)

        if column not in distances_df.columns:
            raise ValueError(f"Coluna de distância não encontrada: {school}")

        school_columns[school] = distances_df.columns.get_loc(column)

    candidate_positions = {
        candidate_id: pos
        for pos, candidate_id in enumerate(distances_df.index)
    }

    candidate_ids = allocation["candidate_id"].to_numpy()
    exam_types = allocation["exam_type"].to_numpy()
    assigned_schools = allocation["school_id"].to_numpy(dtype=object)
    classrooms = allocation["classroom_id"].to_numpy(copy=True)

    candidate_rows = np.array([
        candidate_positions.get(
            candidate_id,
            int(candidate_id) if isinstance(candidate_id, (int, np.integer)) else -1
        )
        for candidate_id in candidate_ids
    ])

    if np.any(candidate_rows < 0) or np.any(candidate_rows >= len(distances_df)):
        raise ValueError("Não foi possível mapear todos os candidatos à matriz de distâncias.")

    distance_matrix = distances_df.to_numpy(dtype=float)
    school_col = {s: school_columns[s] for s in school_ids}
    school_col_array = np.array([school_col[s] for s in school_ids])
    school_index = {s: i for i, s in enumerate(school_ids)}

    school_counts = {
        s: int(np.count_nonzero(assigned_schools == s))
        for s in school_ids
    }

    school_types = {
        s: (exam_types[np.flatnonzero(assigned_schools == s)[0]]
            if school_counts[s] else None)
        for s in school_ids
    }

    distances = np.array([
        distance_matrix[candidate_rows[i], school_col[assigned_schools[i]]]
        for i in range(n)
    ])

    def update_rooms(school):
        positions = np.flatnonzero(assigned_schools == school)
        capacities = room_capacities[school]

        if len(positions) > capacities.sum():
            raise RuntimeError(f"Capacidade excedida na escola {school}.")

        offset = 0
        for room_id, capacity in enumerate(capacities, start=1):
            end = min(offset + capacity, len(positions))
            classrooms[positions[offset:end]] = room_id
            offset = end

    initial_distance = distances.sum()
    iteration = 0

    while True:
        iteration += 1
        improved = False

        # Relocate: apply the first improving move.
        for i in range(n):
            current_school = assigned_schools[i]
            exam_type = exam_types[i]
            current_distance = distances[i]
            candidate_row = candidate_rows[i]

            for new_school in school_ids:
                if new_school == current_school:
                    continue

                if school_types[new_school] is not None and school_types[new_school] != exam_type:
                    continue

                if school_counts[new_school] >= school_capacity[new_school]:
                    continue

                new_distance = distance_matrix[
                    candidate_row, school_col[new_school]
                ]

                if new_distance < current_distance:
                    assigned_schools[i] = new_school
                    distances[i] = new_distance

                    school_counts[current_school] -= 1
                    school_counts[new_school] += 1

                    if school_counts[current_school] == 0:
                        school_types[current_school] = None
                    school_types[new_school] = exam_type

                    update_rooms(current_school)
                    update_rooms(new_school)

                    improved = True
                    break

            if improved:
                break

        if improved:
            if iteration % 10 == 0:
                print(f"Iteração {iteration} | Distância: {distances.sum():.2f}")
            continue

        # Swap: vectorize comparisons for each first candidate.
        for i in range(n):
            eligible = (
                (np.arange(n) > i)
                & (exam_types == exam_types[i])
                & (assigned_schools != assigned_schools[i])
            )
            js = np.flatnonzero(eligible)

            if len(js) == 0:
                continue

            school_a = assigned_schools[i]
            rows_j = candidate_rows[js]
            cols_j = np.array([school_col[s] for s in assigned_schools[js]])

            new_distance_i = distance_matrix[candidate_rows[i], cols_j]
            new_distance_j = distance_matrix[rows_j, school_col[school_a]]

            deltas = new_distance_i + new_distance_j - distances[i] - distances[js]
            improving = np.flatnonzero(deltas < 0)

            if len(improving) == 0:
                continue

            k = improving[0]
            j = js[k]
            school_b = assigned_schools[j]

            assigned_schools[i], assigned_schools[j] = school_b, school_a
            distances[i] = new_distance_i[k]
            distances[j] = new_distance_j[k]

            # Counts and school types do not change in a same-type swap.
            improved = True
            break

        if not improved:
            break

        if iteration % 10 == 0:
            print(f"Iteração {iteration} | Distância: {distances.sum():.2f}")

    allocation["school_id"] = assigned_schools
    allocation["distance"] = distances
    allocation["classroom_id"] = classrooms

    final_distance = distances.sum()

    print(f"Distância inicial: {initial_distance:.2f}")
    print(f"Distância final: {final_distance:.2f}")
    print(f"Redução: {initial_distance - final_distance:.2f}")

    return allocation

def main():
    for file in sorted(INSTANCES_PATH.glob("*.txt")):
        (
            num_candidates,
            num_schools,
            candidates_df,
            schools_df,
            distances_df,
        ) = load_instance(file)

        allocation_df = initial_solution(
            candidates_df,
            schools_df,
            distances_df,
        )
        
        if allocation_df is None:
            print("Não foi encontrada solução viável para K=1")
        
        else:
            total_distance = allocation_df["distance"].sum()

            print(
                f"{file.name}: \n"
                f"{len(allocation_df)}/{num_candidates} candidatos alocados\n"
                f"Distância total: {total_distance}\n"
            )
        
        # allocation_df = local_search(allocation_df, distances_df, schools_df)
        
        # total_distance = allocation_df["distance"].sum()

        # print(
        #     f"APÓS LOCAL SEARCH\n"
        #     f"{file.name}: \n"
        #     f"{len(allocation_df)}/{num_candidates} candidatos alocados\n"
        #     f"Distância total: {total_distance}\n"
        # )
        
if __name__ == "__main__":
    main()