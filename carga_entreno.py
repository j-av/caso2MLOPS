import pandas as pd

# Cargar los datos del archivo MarathonData.csv
marathon_data_extra = pd.read_csv('MarathonData.csv')

# Función para calcular el score de esfuerzo
def calcular_score_entreno(row):
    # Usamos 'km4week' y 'sp4week' para calcular el score de esfuerzo
    if row['sp4week'] > 0 and row['km4week'] > 0:
        score_esfuerzo = row['km4week'] * row['sp4week']  # Ajusta la fórmula según sea necesario
    else:
        score_esfuerzo = 0
    return score_esfuerzo

# Aplicar la función a cada fila del dataset
marathon_data_extra['EffortScore'] = marathon_data_extra.apply(calcular_score_entreno, axis=1)

# Guardar el resultado en un nuevo archivo CSV
marathon_data_extra[['id', 'Name', 'EffortScore']].to_csv('EffortScore_Output.csv', index=False)

print("Calculo de EffortScore completado. Resultados guardados en EffortScore_Output.csv.")
