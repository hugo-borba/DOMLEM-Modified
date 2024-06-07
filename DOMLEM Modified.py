import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from tabulate import tabulate
from collections import OrderedDict

def open_file():
    filename = filedialog.askopenfilename(initialdir="/", title="Select CSV file",
                                           filetypes=(("CSV files", "*.csv"),))
    if filename:
        try:
            df = pd.read_csv(filename)
            return df
        except Exception as e:
            print(f"Error reading the file: {str(e)}")
    return None

def formatting_file(df):
    try:
        index = df.index[df.iloc[:, 0] == "EXEMPLES"][0]
        df_atributes = df.iloc[:index]
        df_exemples = df.iloc[index:]

        EXEMPLES_header = df_atributes.iloc[:, 0].tolist()

        num_columns = len(df_exemples.columns)
        expected_num_headers = num_columns - 1
        
        if len(EXEMPLES_header) < expected_num_headers:
            print(f"Error: Not enough headers provided. Expected {expected_num_headers}, but got {len(EXEMPLES_header)}.")
            return None, None
        elif len(EXEMPLES_header) > expected_num_headers:
            print(f"Warning: Too many headers provided. Expected {expected_num_headers}, but got {len(EXEMPLES_header)}. Truncating headers.")
            EXEMPLES_header = EXEMPLES_header[:expected_num_headers]

        df_exemples.columns = ["EXEMPLES"] + EXEMPLES_header
        df_exemples = df_exemples.iloc[1:]
        df_exemples["EXEMPLES"] = df_exemples["EXEMPLES"].str.replace("_", " ")

        if "PREFERENCE" in df_atributes.columns:
            df_atributes = df_atributes.loc[:, :"PREFERENCE"]
        df_atributes = df_atributes.rename(columns={"ATRIBUTES": "CRITERIA"})

        if "CRITERIA" not in df_atributes.columns:
            raise KeyError("The column 'CRITERIA' was not found after renaming.")

        df_atributes.insert(0, "ATTRIBUTES", [f"Attribute {i+1}" for i in range(len(df_atributes))])

        print("DataFrame 'ATRIBUTES':")
        print(tabulate(df_atributes, headers='keys', tablefmt='fancy_grid', showindex=False))
        
        print("\nDataframe 'EXEMPLES' - Printing the first 20 rows of the dataframe")
        print(tabulate(df_exemples.head(20), headers='keys', tablefmt='fancy_grid', showindex=False))
        print('\n')

    except IndexError:
        print("The 'EXEMPLES' cell was not found in the file.")
    except KeyError as ke:
        print(f"KeyError: {ke}")
    except ValueError as ve:
        print(f"ValueError: {ve}")

    return df_atributes, df_exemples

def creating_vectors(df_atributes, df_exemples):
    criteria = df_atributes["CRITERIA"].tolist()
    data_type = df_atributes["DATA TYPE"].tolist()
    preferences = df_atributes["PREFERENCE"].tolist()
    decision = df_exemples["Dec"].tolist()
    
    return criteria, data_type, preferences, decision

def union_classes(df_exemples):
    if 'Dec' not in df_exemples.columns:
        raise ValueError("A coluna 'Dec' não foi encontrada no DataFrame.")
    
    # Verificar e remover linhas onde 'Dec' é NaN
    if df_exemples['Dec'].isnull().any():
        print("Warning: Found NaN values in 'Dec' column. Removing these rows.")
        df_exemples = df_exemples.dropna(subset=['Dec'])

    decision_classes = df_exemples['Dec'].unique()
    class_dict = {cls: [] for cls in decision_classes}
    
    for index, row in df_exemples.iterrows():
        class_dict[row['Dec']].append(row)
    
    # Cria um OrderedDict para garantir que as classes estejam ordenadas
    class_dfs = OrderedDict(sorted((cls, pd.DataFrame(rows)) for cls, rows in class_dict.items()))
    
    return class_dfs

def downward_union_classes(class_dfs):
    decision_classes = sorted(class_dfs.keys())
    downward_unions = {}
    d_agg_df = pd.DataFrame()
    
    for cls in decision_classes:
        d_agg_df = pd.concat([d_agg_df, class_dfs[cls]])
        downward_unions[cls] = d_agg_df.copy()
    
    return downward_unions, d_agg_df

def upward_union_classes(class_dfs):
    decision_classes = sorted(class_dfs.keys(), reverse=True)
    upward_unions = {}
    u_agg_df = pd.DataFrame()
    
    for cls in decision_classes:
        u_agg_df = pd.concat([class_dfs[cls], u_agg_df])
        upward_unions[cls] = u_agg_df.copy()
    
    return upward_unions, u_agg_df

def dominating_exemples(class_dfs, criteria, preferences):
    # Ordena as classes de decisão
    decision_classes = sorted(class_dfs.keys())
    
    def is_better(r1, r2, preferences):
        # Verifica se r1 é melhor ou igual a r2 com base nas preferências
        for x, y, p in zip(r1[:-1], r2[:-1], preferences[:-1]):  # Exclui a última coluna e preferência
            if (p == "gain" and x < y) or (p == "cost" and x > y):
                return False
        return True

    # Dicionário para armazenar as informações de dominância
    dominating_info = {}
    
    # Itera sobre todas as classes de decisão
    for i, cls in enumerate(decision_classes):
        df = class_dfs[cls]
        # Inicializa a estrutura para armazenar informações sobre dominância
        class_dominating_info = []
        
        # Itera sobre cada linha na classe atual
        for index, row in df.iterrows():
            example_info = {
                "Example": row["EXEMPLES"],
                "Class": cls,
                "Performance": row[criteria].values,
                "Dominates": []
            }
            # Verifica a dominância apenas com exemplos das classes superiores
            for other_cls in decision_classes[i+1:]:
                other_df = class_dfs[other_cls]
                for other_index, other_row in other_df.iterrows():
                    if is_better(row[criteria].values, other_row[criteria].values, preferences):
                        example_info["Dominates"].append({
                            "Example": other_row["EXEMPLES"],
                            "Performance": other_row[criteria].values,
                            "Class": other_cls
                        })
            
            # Apenas adiciona ao dicionário se dominar algum outro exemplo
            if example_info["Dominates"]:
                class_dominating_info.append(example_info)
        
        dominating_info[cls] = class_dominating_info
    
    return dominating_info

def dominated_exemples(class_dfs, criteria, preferences):
    # Ordena as classes de decisão
    decision_classes = sorted(class_dfs.keys(), reverse=True)
    
    def is_better(r1, r2, preferences):
        # Verifica se r1 é melhor ou igual a r2 com base nas preferências
        for x, y, p in zip(r1[:-1], r2[:-1], preferences[:-1]):  # Exclui a última coluna e preferência
            if (p == "gain" and x < y) or (p == "cost" and x > y):
                return False
        return True

    # Dicionário para armazenar as informações de dominância
    dominated_info = {}
    
    # Itera sobre todas as classes de decisão
    for i, cls in enumerate(decision_classes):
        df = class_dfs[cls]
        # Inicializa a estrutura para armazenar informações sobre dominância
        class_dominated_info = []
        
        # Itera sobre cada linha na classe atual
        for index, row in df.iterrows():
            example_info = {
                "Example": row["EXEMPLES"],
                "Class": cls,
                "Performance": row[criteria].values,
                "Dominated By": []
            }
            # Verifica a dominância apenas com exemplos das classes inferiores
            for other_cls in decision_classes[i+1:]:
                other_df = class_dfs[other_cls]
                for other_index, other_row in other_df.iterrows():
                    if is_better(other_row[criteria].values, row[criteria].values, preferences):
                        example_info["Dominated By"].append({
                            "Example": other_row["EXEMPLES"],
                            "Performance": other_row[criteria].values,
                            "Class": other_cls
                        })
            
            # Apenas adiciona ao dicionário se for dominado por algum outro exemplo
            if example_info["Dominated By"]:
                class_dominated_info.append(example_info)
        
        dominated_info[cls] = class_dominated_info
    
    return dominated_info

def main():
    df = open_file()
    if df is not None:
        df_atributes, df_exemples = formatting_file(df)
        if df_atributes is None or df_exemples is None:
            print("Formatting file failed. Exiting...")
            return
        criteria, data_type, preferences, decisions = creating_vectors(df_atributes, df_exemples)
        class_dfs = union_classes(df_exemples)            

        for cls, df in class_dfs.items():
            print(f"Decision Class {cls} - {len(df)} Exemples")
            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
        print('\n')

        downward_unions, d_agg_df = downward_union_classes(class_dfs)
        for cls, union in downward_unions.items():
            print(f"Downwards Union Classes {cls} - {len(d_agg_df)} Exemples")
            print(tabulate(union, headers='keys', tablefmt='fancy_grid', showindex=False))
        print('\n')

        upward_unions, u_agg_df = upward_union_classes(class_dfs)
        for cls, union in upward_unions.items():
            print(f"Upward Union Classes {cls} - {len(u_agg_df)} Exemples")
            print(tabulate(union, headers='keys', tablefmt='fancy_grid', showindex=False))
        print('\n')

        # Utilizando a nova função dominating_exemples
        dominating_info = dominating_exemples(class_dfs, criteria, preferences)
        for cls, examples in dominating_info.items():
            for example in examples:
                print(f"Example: {example['Example']}, Performance: {example['Performance']}")
                print(f"Dominates:")
                for dominated in example['Dominates']:
                    print(f"  - Example: {dominated['Example']}, Performance: {dominated['Performance']}, Class: {dominated['Class']}")
                print('\n')

        # Utilizando a nova função dominated_exemples
        dominated_info = dominated_exemples(class_dfs, criteria, preferences)
        for cls, examples in dominated_info.items():
            for example in examples:
                print(f"Example: {example['Example']}, Performance: {example['Performance']}")
                print(f"Dominated By:")
                for dominated_by in example['Dominated By']:
                    print(f"  - Example: {dominated_by['Example']}, Performance: {dominated_by['Performance']}, Class: {dominated_by['Class']}")
                print('\n')

if __name__ == "__main__":
    main()
