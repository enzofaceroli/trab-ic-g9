import pandas as pd

# função pra ajudar a ler o arquivo. basicamente pula para a linha de interesse
def jump_lines(file, str):
    for line in file:
        if line.strip().startswith(str):
            break

def load_instance(file_path):
    print(f"Loading instance from: {file_path.name}")
    with open(file_path, 'r', encoding='utf-8') as file:
        # lendo: CANDIDATOS (id, CEP, tipo_prova, latitude, longitude)
        num_candidates = int(file.readline().split(':')[1].strip())
        num_schools = int(file.readline().split(':')[1].strip())

        # aqui pula para a linha que começa a matriz dos candidatos
        jump_lines(file, '# CANDIDATOS') 
        
        # le matriz de candidatos
        candidates_cols = ['id', 'CEP', 'tipo_prova', 'latitude', 'longitude']
        candidates_df = pd.read_csv(file, nrows=num_candidates, header=None, names=candidates_cols, skipinitialspace=True)
        
        file.close()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # lendo: ESCOLAS (id, CEP, latitude, longitude, num_salas, capacidades_salas)  
        # aqui pula para a linha que começa a matriz das escolas
        jump_lines(file, '# ESCOLAS') 
        
        # le matriz das escolas
        schools_cols = ['id', 'CEP', 'latitude', 'longitude', 'num_salas', 'capacidades_salas']
        schools_df = pd.read_csv(file, nrows=num_schools, header=None, names=schools_cols, skipinitialspace=True)
        schools_df['capacidades_salas'] = schools_df['capacidades_salas'].apply(
            lambda x: [int(c) for c in str(x).split(';')]
        )

        file.close()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # lendo: MATRIZ_DISTANCIAS (linhas=candidatos, colunas=escolas, em km)
        # aqui pula para a linha que começa a matriz de distancias
        jump_lines(file, '# MATRIZ_DISTANCIAS')
        
        # le matriz de distancias
        distances_cols = [f'{i+1}' for i in range(num_schools)]
        distances_df = pd.read_csv(file, nrows=num_candidates, header=None, names=distances_cols, skipinitialspace=True)
        
        file.close()
    
    return num_candidates, num_schools, candidates_df, schools_df, distances_df

    
