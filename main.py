from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier,plot_tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image  
from sklearn import tree
import pydotplus
import datetime

#  ha la parte di verifica
def extract_feature_with_decisiontree(df, target_column="target", removeNotImportantFeature=True, graphvizPdf=False, plotTree=False):
    
    print("## extract_feature_with_decisiontree target_column=", target_column, " removeNotImportantFeature=",removeNotImportantFeature)

    # Estrai le colonne delle features e della variabile target
    features_columns = df.columns[:-1]
    
    # Estrai le colonne delle feature (X) e della variabile target (y)
    X = df[features_columns]
    y = df[target_column]

    # Dividi il dataset in set di addestramento e set di verifica
    # Il parametro test_size specifica la percentuale di dati da utilizzare per la verifica
    # random_state è impostato per garantire la riproducibilità dei risultati
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea un modello di albero decisionale
    clf = DecisionTreeClassifier(random_state=42)

    # Addestra il modello sul set di addestramento
    clf.fit(X_train, y_train)

    # Valuta le prestazioni del modello sul set di verifica
    accuracy = clf.score(X_val, y_val)
    print(f'## extract_feature_with_decisiontree - Accuracy on validation set: {accuracy}')

    # qui uso plt per visualizzare tree
    if plotTree:
        plt.figure(figsize=(12, 6))
        plt.title("Decision Tree")
        plot_tree(clf, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True, fontsize=8)
        plt.show()

    # qui mostro in console l'albero
    dt_tree_rules = export_text(clf, feature_names=X.columns.tolist())
    print("### extract_feature_with_decisiontree Decision Tree Rules:")
    print(dt_tree_rules)

    # qui uso graphviz per la visualizzazione pdf
    if graphvizPdf:
        dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns,class_names=['0', '1'],filled=True, rounded=True, special_characters=True)  
        graph = graphviz.Source(dot_data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Ottieni una stringa temporale unica
        try:
            # Salva l'immagine con timestamp nel nome del file
            graph.render(f"pdf/decision_tree_{timestamp}")
        except graphviz.backend.ExecutableNotFound:
            # Se Graphviz non è installato, fornisci un messaggio di avviso
            print("Graphviz non è installato. Installalo per generare la visualizzazione dell'albero.")
        except graphviz.backend.CalledProcessError as e:
            # Se c'è un errore nel processo chiamato, gestisci l'eccezione
            if "Permission denied" in e.stderr.decode():
                print(f"Errore: Impossibile sovrascrivere il file 'decision_tree_{timestamp}'. "
                    "Chiudi il file esistente prima di generare nuovamente l'albero.")
            else:
                raise  # Se si tratta di un altro tipo di errore, sollevalo nuovamente
        graph.view(f"pdf/decision_tree_{timestamp}")  # Apri l'immagine con il visualizzatore predefinito
        
    # Estrai le feature più importanti
    feature_importance = clf.feature_importances_
    # Crea un DataFrame con le feature e le relative importanze
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

    if removeNotImportantFeature:
        # Rimuovi le features con importanza 0
        non_zero_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
        print("### extract_feature_with_decisiontree Features with Importance > 0")
        print(non_zero_importance_df['Feature'].tolist())
        return non_zero_importance_df
    else:
        return feature_importance_df

def extract_feature_with_extratreeclassifier(df, target_column="target", removeNotImportantFeature=True, graphvizPdf=False, plotTree=False):
    
    print("## extract_feature_with_extratreeclassifier target_column=", target_column, " removeNotImportantFeature=",removeNotImportantFeature)

    # Estrai le colonne delle features e della variabile target
    features_columns = df.columns[:-1]
    
    # Estrai le colonne delle feature (X) e della variabile target (y)
    X = df[features_columns]
    y = df[target_column]

    # Dividi il dataset in set di addestramento e set di verifica
    # Il parametro test_size specifica la percentuale di dati da utilizzare per la verifica
    # random_state è impostato per garantire la riproducibilità dei risultati
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crea un modello di albero decisionale
    clf = ExtraTreeClassifier(random_state=42)

    # Addestra il modello sul set di addestramento
    clf.fit(X_train, y_train)

    # Valuta le prestazioni del modello sul set di verifica
    accuracy = clf.score(X_val, y_val)
    print(f'## extract_feature_with_extratreeclassifier - Accuracy on validation set: {accuracy}')

    # qui uso plt per visualizzare tree
    if plotTree:
        plt.figure(figsize=(12, 6))
        plt.title("Decision Tree")
        plot_tree(clf, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True, fontsize=8)
        plt.show()

    # qui mostro in console l'albero
    dt_tree_rules = export_text(clf, feature_names=X.columns.tolist())
    print("### extract_feature_with_extratreeclassifier Decision Tree Rules:")
    print(dt_tree_rules)

    # qui uso graphviz per la visualizzazione pdf
    if graphvizPdf:
        dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns,class_names=['0', '1'],filled=True, rounded=True, special_characters=True)  
        graph = graphviz.Source(dot_data)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Ottieni una stringa temporale unica
        try:
            # Salva l'immagine con timestamp nel nome del file
            graph.render(f"pdf/decision_tree_{timestamp}")
        except graphviz.backend.ExecutableNotFound:
            # Se Graphviz non è installato, fornisci un messaggio di avviso
            print("Graphviz non è installato. Installalo per generare la visualizzazione dell'albero.")
        except graphviz.backend.CalledProcessError as e:
            # Se c'è un errore nel processo chiamato, gestisci l'eccezione
            if "Permission denied" in e.stderr.decode():
                print(f"Errore: Impossibile sovrascrivere il file 'decision_tree_{timestamp}'. "
                    "Chiudi il file esistente prima di generare nuovamente l'albero.")
            else:
                raise  # Se si tratta di un altro tipo di errore, sollevalo nuovamente
        graph.view(f"pdf/decision_tree_{timestamp}")  # Apri l'immagine con il visualizzatore predefinito
        
    # Estrai le feature più importanti
    feature_importance = clf.feature_importances_
    # Crea un DataFrame con le feature e le relative importanze
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

    if removeNotImportantFeature:
        # Rimuovi le features con importanza 0
        non_zero_importance_df = feature_importance_df[feature_importance_df['Importance'] > 0]
        print("### extract_feature_with_decisiontree Features with Importance > 0")
        print(non_zero_importance_df['Feature'].tolist())
        return non_zero_importance_df
    else:
        return feature_importance_df

# Funzione per ottenere il testo dell'albero solo per i nodi <= 0.5 e ordinato per Gini
def get_text_for_threshold(tree, feature_names, threshold):
    tree_rules = export_text(tree, feature_names=feature_names)
    lines = tree_rules.split('\n')

    # Filtra solo le linee che contengono la soglia specificata
    filtered_lines = [line for line in lines if f'<= {threshold:.6f}' in line]

    # Ordina le linee in base al Gini
    sorted_lines = sorted(filtered_lines, key=lambda x: float(x.split(" ")[-1]))

    return '\n'.join(sorted_lines)

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

def binary_to_text(df,):
    for index, row in df.iterrows():
        binary_code = row['binary']
        character = chr(int(binary_code, 2))
        target = row['target']
        # print(f"Index {index}: Binary Code: {binary_code}, Character: {character}, Target: {target}")
        print(f"Binary Code: {binary_code}, Character: {character}, Target: {target}")


csv_file_path = 'XMAS_data2023.csv'
df = pd.read_csv(csv_file_path, index_col=0)
# Set pandas options to display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

feature_extracted_features_df=extract_feature_with_decisiontree(df,removeNotImportantFeature=True,graphvizPdf=False)
print(feature_extracted_features_df.sort_values('Importance', ascending=False))

# # prova con extratreeclassifier
# feature_extracted_features_df=extract_feature_with_extratreeclassifier(df,removeNotImportantFeature=True)
# print(feature_extracted_features_df.sort_values('Importance', ascending=False))

# exit(0)

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
ascii_df.to_csv("ascii_df.csv", index=False)

ascii_df_no_duplicates = ascii_df.drop_duplicates()

binary_to_text(ascii_df_no_duplicates)




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


