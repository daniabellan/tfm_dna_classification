import h5py

def load_dict_h5(file_path):
    dictionary = {}
    
    with h5py.File(file_path, 'r') as f:
        for read_id in f.keys():
            # Acceder a la secuencia
            sequence = f[f"{read_id}/sequence"]
            # Si la secuencia está almacenada como string, decodificar
            if isinstance(sequence, h5py._hl.dataset.Dataset):
                sequence = sequence[()].decode()  # Convertir bytes a string
            
            # Acceder a la señal eléctrica (se debe almacenar como array de NumPy)
            signal_pa = f[f"{read_id}/signal_pa"][()]  # Acceder a los datos (de tipo np.array)
            
            # Agregar los datos al diccionario
            dictionary[read_id] = {
                "sequence": sequence,
                "signal_pa": signal_pa.tolist()  # Convertir a lista de Python
            }
    
    print(f"Diccionario cargado desde {file_path}")
    return dictionary


load_dict_h5("data/ecoli_k12/pod5_1.h5")