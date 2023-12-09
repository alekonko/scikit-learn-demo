import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def extract_feature_with_decisiontree_old(df, removeNotImportantFeature=True):
    # Estrai le colonne delle features e della variabile target
    features_columns = df.columns[:-1]
    target_column = df.columns[-1]
    
    # Estrai le colonne delle feature (X) e della variabile target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Crea un modello di albero decisionale
    clf = DecisionTreeClassifier()
    
    # Addestra il modello sull'intero dataset
    clf.fit(X, y)
    
    # Estrai le feature più importanti
    feature_importance = clf.feature_importances_
    
    # Crea un DataFrame con le feature e le relative importanze
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    
    if removeNotImportantFeature:
        # Rimuovi le features con importanza 0
        non_zero_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
        return non_zero_importance_df
    else:
        return feature_importance_df

def extract_feature_with_decisiontree(df, target_column="target", removeNotImportantFeature=True):
    # Estrai le colonne delle features e della variabile target
    features_columns = df.columns[:-1]
    
    # Estrai le colonne delle feature (X) e della variabile target (y)
    X = df.iloc[:, :-1]
    y = df[target_column]
    
    # Crea un modello di albero decisionale
    clf = DecisionTreeClassifier()
    
    # Addestra il modello sull'intero dataset
    clf.fit(X, y)
    
    # Estrai le feature più importanti
    feature_importance = clf.feature_importances_
    
    # Crea un DataFrame con le feature e le relative importanze
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    
    if removeNotImportantFeature:
        # Rimuovi le features con importanza 0
        non_zero_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
        return non_zero_importance_df
    else:
        return feature_importance_df


def sort_df_features(df, colonna_target='target'):
    # Verifica se la colonna del target esiste nel DataFrame
    if colonna_target in df.columns:
        # Rimuovi la colonna del target
        features_df = df.drop(columns=[colonna_target])

        # Ordina le colonne alfabeticamente
        features_df_sorted = features_df.sort_index(axis=1)

        # Unisci le colonne delle features ordinate con la colonna del target
        df_sorted = pd.concat([features_df_sorted, df[colonna_target]], axis=1)

        return df_sorted
    else:
        print(f"La colonna del target '{colonna_target}' non è presente nel DataFrame.")
        return df

def extract_X(df):
    # Estrai le colonne delle feature (X) e della variabile target (y)
    return df.iloc[:, :-1]

def extract_y(df):
    # estrae target - ultima colonna
    return df.iloc[:, -1]


csv_file_path = 'XMAS_data2023.csv'
df = pd.read_csv(csv_file_path, index_col=0)
# Set pandas options to display all columns and rows
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.expand_frame_repr', False)

feature_extracted_features_df=extract_feature_with_decisiontree(df,removeNotImportantFeature=True)
# feature_extracted_features_sorted=extract_feature_with_decisiontree(sort_df_features(df,'target'),removeNotImportantFeature=True)

print(feature_extracted_features_df)

non_zero_columns = feature_extracted_features_df['Feature'].tolist()
# Seleziona solo le colonne desiderate nel DataFrame originale
ascii_df = df[non_zero_columns + ['target']]
# Crea una nuova colonna con la fusione delle prime sette feature
ascii_df['binary'] = df[non_zero_columns].astype(str).agg(''.join, axis=1)

# Muovi la colonna 'binary' all'inizio del DataFrame
column_order = ['binary'] + non_zero_columns + ['target']
ascii_df = ascii_df[column_order]

# Stampa il nuovo DataFrame
print(ascii_df)

print(extract_feature_with_decisiontree(ascii_df))


# # Seleziona solo le colonne non nulle nel DataFrame originale
# X_processed = X[non_zero_columns].copy()

# # Aggiungi la variabile target al DataFrame X_processed
# X_processed['target'] = y

# # Stampa il DataFrame risultante
# print(X_processed)

# # Crea una nuova colonna con la fusione delle prime sette feature
# X_processed['code'] = X_processed.iloc[:, :7].astype(str).agg(''.join, axis=1)

# # Seleziona solo le colonne necessarie, inclusa la variabile target
# asciicode_df = X_processed[['code', 'target']]

# # Stampa il nuovo DataFrame
# print(asciicode_df)

# for index, row in asciicode_df.iterrows():
#     binary_code = row['code']
#     character = chr(int(binary_code, 2))
#     target = row['target']
#     if target > 0:
#         print(f"Index {index}: Binary Code: {binary_code}, Character: {character}, Target: {target}")



