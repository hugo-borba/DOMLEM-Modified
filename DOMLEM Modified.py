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

        # Aqui está a nova adição para converter as colunas de performance para tipo float
        for col in df_exemples.columns[1:]:
            df_exemples[col] = pd.to_numeric(df_exemples[col], errors='coerce')

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

def upward_union_classes(class_dfs):
    decision_classes = sorted(class_dfs.keys(), reverse=True)
    upward_unions = {}
    u_agg_df = pd.DataFrame()
    
    for cls in decision_classes:
        u_agg_df = pd.concat([class_dfs[cls], u_agg_df])
        upward_unions[cls] = u_agg_df.copy()
    
    return upward_unions, u_agg_df

def downward_union_classes(class_dfs):
    decision_classes = sorted(class_dfs.keys())
    downward_unions = {}
    d_agg_df = pd.DataFrame()
    
    for cls in decision_classes:
        d_agg_df = pd.concat([d_agg_df, class_dfs[cls]])
        downward_unions[cls] = d_agg_df.copy()
    
    return downward_unions, d_agg_df

def dominating_exemples(class_dfs, criteria, preferences):  
    # Ordena as classes de decisão
    decision_classes = sorted(class_dfs.keys())

    def is_better(r1, r2, preferences):
        # Verifica se r1 é melhor ou igual a r2 com base nas preferências
        for x, y, p in zip(r1[:-1], r2[:-1], preferences):
            if (p == "gain" and x < y) or (p == "cost" and x > y):
                return False
        return True

    # Dicionário para armazenar as informações de dominância
    dominating_info = {}

    # Itera sobre todas as classes de decisão
    for cls in decision_classes:
        df = class_dfs[cls]

        # Itera sobre cada linha na classe atual
        for index, row in df.iterrows():
            exemple = row["EXEMPLES"]
            dominates_list = []

            # Verifica a dominância apenas com exemplos das classes superiores
            for other_cls in decision_classes:
                other_df = class_dfs[other_cls]
                for other_index, other_row in other_df.iterrows():
                    if is_better(row[criteria].values, other_row[criteria].values, preferences):
                        dominates_list.append({
                            f"Dominating - {exemple}": other_row["EXEMPLES"],
                            "Performance": other_row[criteria].values.tolist(),  # Convertendo para lista
                            "Class": other_cls
                        })

            # Adiciona ao dicionário se dominar algum outro exemplo
            if dominates_list:
                dominating_info[exemple] = pd.DataFrame(dominates_list)

    # Ordena as chaves do dicionário alfabeticamente
    dominating_info = {k: dominating_info[k] for k in sorted(dominating_info)}

    return dominating_info

def dominated_exemples(class_dfs, criteria, preferences):
    # Ordena as classes de decisão
    decision_classes = sorted(class_dfs.keys(), reverse=True)

    def is_worse(r1, r2, preferences):
        # Verifica se r1 é pior ou igual a r2 com base nas preferências
        for x, y, p in zip(r1[:-1], r2[:-1], preferences):
            if (p == "gain" and x > y) or (p == "cost" and x < y):
                return False
        return True

    # Dicionário para armazenar as informações de dominância
    dominated_info = {}

    # Itera sobre todas as classes de decisão
    for cls in decision_classes:
        df = class_dfs[cls]

        # Itera sobre cada linha na classe atual
        for index, row in df.iterrows():
            exemple = row["EXEMPLES"]
            dominated_by_list = []

            # Verifica a dominância apenas com exemplos das classes superiores
            for other_cls in decision_classes:
                other_df = class_dfs[other_cls]
                for other_index, other_row in other_df.iterrows():
                    if is_worse(row[criteria].values, other_row[criteria].values, preferences):
                        dominated_by_list.append({
                            f"Dominated - {exemple}": other_row["EXEMPLES"],
                            "Performance": other_row[criteria].values.tolist(),  # Convertendo para lista
                            "Class": other_cls
                        })

            # Adiciona ao dicionário se for dominado por algum outro exemplo
            if dominated_by_list:
                dominated_info[exemple] = pd.DataFrame(dominated_by_list)

    # Ordena as chaves do dicionário alfabeticamente
    dominated_info = {k: dominated_info[k] for k in sorted(dominated_info)}

    return dominated_info

def lower_approximation(downward_unions, dominating_info, df_exemples):
    low_app = OrderedDict()  # Dicionário ordenado para armazenar as aproximações inferiores

    # Ordena as downward_unions por chave (classe) em ordem crescente
    sorted_classes = sorted(downward_unions.keys())

    for i, cls in enumerate(sorted_classes):
        union = downward_unions[cls].copy()  # Cria uma cópia do DataFrame para modificar
        tmp = []  # Lista temporária para armazenar objetos que pertencem à aproximação inferior
        UClass = set(union['EXEMPLES'])  # Cria um conjunto dos exemplos na união de classes atual

        for exemple in UClass:
            is_conflicting = False

            # Verifica se o exemplo está em dominating_info
            if exemple in dominating_info:
                obj = dominating_info[exemple]
                domination_col = next((col for col in obj.columns if "Dominating" in col), None)

                if domination_col:
                    domination_set = set(obj[domination_col])
                    # Verifica se o exemplo domina outro exemplo de classe maior
                    for dominating_exemple in domination_set:
                        dominating_class = df_exemples[df_exemples["EXEMPLES"] == dominating_exemple]["Dec"].values
                        if dominating_class.size > 0 and dominating_class[0] > cls:
                            is_conflicting = True
                            break

            if not is_conflicting:
                performance = df_exemples[df_exemples["EXEMPLES"] == exemple].iloc[0, 1:-1].tolist()
                original_class = df_exemples[df_exemples["EXEMPLES"] == exemple]["Dec"].values[0]
                tmp.append({
                    "Exemples": exemple,
                    "Performance": performance,
                    "Class": original_class
                })

        # Ordena a lista temporária de exemplos alfabeticamente pela coluna "Exemples"
        tmp = sorted(tmp, key=lambda x: x["Exemples"])

        low_app[cls] = tmp  # Adiciona a classe e os objetos não conflitantes ao dicionário de aproximações inferiores

    return low_app

def upward_approximation(upward_unions, dominating_info, df_exemples):
    up_app = OrderedDict()  # Dicionário ordenado para armazenar as aproximações superiores

    # Ordena as upward_unions por chave (classe) em ordem crescente
    sorted_classes = sorted(upward_unions.keys())

    for i, cls in enumerate(sorted_classes):
        union = upward_unions[cls].copy()  # Cria uma cópia do DataFrame para modificar
        tmp = []  # Lista temporária para armazenar objetos que pertencem à aproximação superior
        UClass = set(union['EXEMPLES'])  # Cria um conjunto dos exemplos na união de classes atual

        for exemple in UClass:
            performance = df_exemples[df_exemples["EXEMPLES"] == exemple].iloc[0, 1:-1].tolist()
            original_class = df_exemples[df_exemples["EXEMPLES"] == exemple]["Dec"].values[0]
            tmp.append({
                "Exemples": exemple,
                "Performance": performance,
                "Class": original_class
            })

            # Verifica se o exemplo está em dominating_info
            if exemple in dominating_info:
                obj = dominating_info[exemple]
                domination_col = next((col for col in obj.columns if "Dominating" in col), None)

                if domination_col:
                    domination_set = set(obj[domination_col])
                    # Adiciona os exemplos de classe maior dominados pelo exemplo atual
                    for dominating_exemple in domination_set:
                        dominating_class = df_exemples[df_exemples["EXEMPLES"] == dominating_exemple]["Dec"].values
                        if dominating_class.size > 0 and dominating_class[0] > cls:
                            performance = df_exemples[df_exemples["EXEMPLES"] == dominating_exemple].iloc[0, 1:-1].tolist()
                            tmp.append({
                                "Exemples": dominating_exemple,
                                "Performance": performance,
                                "Class": dominating_class[0]
                            })

        # Remove duplicatas
        seen = set()
        unique_tmp = []
        for d in tmp:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                unique_tmp.append(d)

        # Ordena a lista temporária de exemplos alfabeticamente pela coluna "Exemples"
        unique_tmp = sorted(unique_tmp, key=lambda x: x["Exemples"])

        up_app[cls] = unique_tmp  # Adiciona a classe e os objetos não conflitantes ao dicionário de aproximações superiores

    return up_app

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

        downward_unions, d_agg_df = downward_union_classes(class_dfs)
        for cls, union in downward_unions.items():
            print(f"Downwards Union Classes {cls} - {len(d_agg_df)} Exemples")
            print(tabulate(union, headers='keys', tablefmt='fancy_grid', showindex=False))

        upward_unions, u_agg_df = upward_union_classes(class_dfs)
        for cls, union in upward_unions.items():
            print(f"Upward Union Classes {cls} - {len(u_agg_df)} Exemples")
            print(tabulate(union, headers='keys', tablefmt='fancy_grid', showindex=False))
        
        dominating_info = dominating_exemples(class_dfs, criteria, preferences)
        dominated_info = dominated_exemples(class_dfs, criteria, preferences)

        # Loop for duplo para imprimir dominating_info e dominated_info
        for example, dom_df in dominating_info.items():
            print(tabulate(dom_df, headers='keys', tablefmt='fancy_grid', showindex=False))
            if example in dominated_info:
                domd_df = dominated_info[example]
                print(tabulate(domd_df, headers='keys', tablefmt='fancy_grid', showindex=False))

        lower_approx = lower_approximation(downward_unions, dominating_info, df_exemples)
        print("\nLower Approximation:")
        for cls, info in lower_approx.items():
            print(f"Class: {cls}")
            print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

        upward_approx = upward_approximation(upward_unions, dominating_info, df_exemples)
        print("\nUpward Approximation:")
        for cls, info in upward_approx.items():
            print(f"Class: {cls}")
            print(tabulate(info, headers='keys', tablefmt='fancy_grid'))

if __name__ == "__main__":
    main()
