import numpy as np  # Importa a biblioteca NumPy, que é útil para operações numéricas e manipulação de arrays
import pandas as pd  # Importa a biblioteca pandas, que é útil para manipulação e análise de dados
from functools import reduce  # Importa a função reduce do módulo functools, que aplica uma função acumulada a elementos de uma sequência
from tabulate import tabulate  # Importa a função tabulate, que formata tabelas de dados
from tkinter import Tk  # Importa o módulo Tk do Tkinter, usado para criar interfaces gráficas
from tkinter.filedialog import asksaveasfilename  # Importa a função asksaveasfilename, que abre uma janela de diálogo para salvar arquivos

def build_info_system(df, preferences):
    df = df.reset_index(drop=True)  # Redefine o índice do DataFrame sem manter o índice antigo
    df.index = df.index + 1  # Faz o índice começar do 1
    df = df.reset_index().rename(columns={"index": "ID"})  # Adiciona uma coluna ID ao DataFrame
    attributes = list(df.columns[1:-1])  # Extrai os nomes dos atributos (todas as colunas, exceto a primeira e a última)
    decision = df.columns[-1]  # Define a coluna de decisão como a última coluna
    infosystem = {
        "attributes": [{"name": attr, "preference": pref} for attr, pref in zip(attributes, preferences)],  # Cria uma lista de dicionários para atributos e suas preferências
        "examples": df.values.tolist()  # Converte os valores do DataFrame para uma lista de listas
    }
    return infosystem  # Retorna o sistema de informações

def is_better(r1, r2, preference):
    return all(
        ((x >= y and p == "gain") or (x <= y and p == "cost"))
        for x, y, p in zip(r1, r2, preference)  # Compara os valores de acordo com a preferência (ganho ou custo)
    )

def is_worst(r1, r2, preference):
    return all(
        ((x <= y and p == "gain") or (x >= y and p == "cost"))
        for x, y, p in zip(r1, r2, preference)  # Compara os valores de acordo com a preferência (ganho ou custo)
    )

def downward_unions_of_classes(infosystem):
    matrix = infosystem["examples"]  # Obtém os exemplos do sistema de informações
    decision_classes = sorted(list(set(int(r[-1]) for r in matrix)))  # Extrai e ordena as classes de decisão
    return [[int(r[0]) for r in matrix if int(r[-1]) <= c] for c in decision_classes]  # Cria uniões descendentes das classes

def upward_unions_of_classes(infosystem):
    matrix = infosystem["examples"]  # Obtém os exemplos do sistema de informações
    decision_classes = sorted(list(set(int(r[-1]) for r in matrix)))  # Extrai e ordena as classes de decisão
    return [[int(r[0]) for r in matrix if int(r[-1]) >= c] for c in decision_classes]  # Cria uniões ascendentes das classes

def dominating_set(infosystem):
    matrix = infosystem["examples"]  # Obtém os exemplos do sistema de informações
    preference = [s["preference"] for s in infosystem["attributes"]]  # Obtém as preferências dos atributos
    return [
        {
            "object": int(row[0]),  # Define o objeto atual
            "dominance": [int(i[0]) for i in matrix if i != row and is_better(row[1:], i[1:], preference) and not is_worst(row[1:], i[1:], preference)],  # Lista de objetos dominados
            "examples": [i for i in matrix if i != row and is_better(row[1:], i[1:], preference) and not is_worst(row[1:], i[1:], preference)]  # Lista de exemplos dominados
        }
        for row in matrix  # Itera sobre cada linha no matrix
    ]

def dominated_set(infosystem):
    matrix = infosystem["examples"]  # Obtém os exemplos do sistema de informações
    preference = [s["preference"] for s in infosystem["attributes"]]  # Obtém as preferências dos atributos
    return [
        {
            "object": int(row[0]),  # Define o objeto atual
            "dominance": [int(i[0]) for i in matrix if i != row and is_worst(row[1:], i[1:], preference)],  # Lista de objetos que dominam o atual
            "examples": [i for i in matrix if i != row and is_worst(row[1:], i[1:], preference)]  # Lista de exemplos que dominam o atual
        }
        for row in matrix  # Itera sobre cada linha no matrix
    ]

def lower_approximation(union_classes, dom_set, union_type='downward'):
    # Inicializa a lista que armazenará as aproximações inferiores de cada classe
    lower_approx = []

    # Itera sobre cada classe de união com seu índice
    for c, union in enumerate(union_classes):
        # Inicializa a lista para armazenar objetos que pertencem à aproximação inferior da classe atual
        lower_class_objects = []
        # Itera sobre cada objeto na união da classe atual
        for obj in union:
            # Itera sobre cada objeto no conjunto de dominância
            for dom_obj in dom_set:
                # Verifica se o objeto atual do conjunto de dominância é o mesmo que o objeto da união
                if dom_obj["object"] == obj:
                    if union_type == 'downward':
                        # Inicializa a variável para verificar se o objeto domina algum objeto de classe superior
                        dominates_superior = False
                        # Itera sobre os objetos dominados pelo objeto atual do conjunto de dominância
                        for dominated in dom_obj["examples"]:
                            # Supondo que a última entrada na lista 'dominated' representa a classe do objeto
                            dominated_class = dominated[-1]
                            if dominated_class > c + 1:
                                dominates_superior = True
                                break
                    if union_type == 'upward':
                        # Inicializa a variável para verificar se o objeto domina algum objeto de classe superior
                        dominates_superior = False
                        # Itera sobre os objetos dominados pelo objeto atual do conjunto de dominância
                        for dominated in dom_obj["examples"]:
                            # Supondo que a última entrada na lista 'dominated' representa a classe do objeto
                            dominated_class = dominated[-1]
                            if dominated_class < c + 1:
                                dominates_superior = True
                                break
                    # Se o objeto não domina nenhum objeto de classe superior, adiciona-o à lista de objetos da classe atual
                    if not dominates_superior:
                        lower_class_objects.append(obj)
                    break
        # Adiciona a aproximação inferior da classe atual à lista principal
        lower_approx.append({"class": c + 1, "objects": lower_class_objects})
    
    # Retorna a lista com as aproximações inferiores de todas as classes
    return lower_approx

def upper_approximation(union_classes, dom_set, union_type='downward'):
    # Inicializa a lista que armazenará as aproximações inferiores de cada classe
    upper_approx = []
    dominated_inferior_approx = []

    if union_type == 'downward':
        # Itera sobre cada classe de união com seu índice
        for c, union in enumerate(union_classes):
            # Inicializa a lista para armazenar objetos que pertencem à aproximação inferior da classe atual
            lower_class_objects = []
            dominated_class_objects = []

            # Itera sobre cada objeto na união da classe atual
            for obj in union:
                # Inicializa a variável 'dominated_inferior' para verificar se o objeto deve ser adicionado a 'dominated_class_objects'
                dominated_inferior = True
                # Itera sobre cada objeto no conjunto de dominância
                for dom_obj in dom_set:
                    # Verifica se o objeto atual do conjunto de dominância é o mesmo que o objeto da união
                    if dom_obj["object"] == obj:
                        # Inicializa a variável para verificar se o objeto domina algum objeto de classe superior
                        dominates_superior = True
                        # Itera sobre os objetos dominados pelo objeto atual do conjunto de dominância
                        for dominated in dom_obj["examples"]:
                            # Supondo que a última entrada na lista 'dominated' representa a classe do objeto
                            dominated_class = dominated[-1]
                            if dominated_class >= c + 1:
                                dominates_superior = False
                            if dominated_class < c + 1:
                                dominated_inferior = False
                                break
                        # Se o objeto não domina nenhum objeto de classe superior, adiciona-o à lista de objetos da classe atual
                        if not dominates_superior:
                            lower_class_objects.append(obj)
                        if not dominated_inferior:
                            dominated_class_objects.append(obj)
                        break
                # Se o objeto não domina ninguém, adiciona-o à lista de objetos da classe atual
                if dominates_superior:
                    lower_class_objects.append(obj)
            # Adiciona a aproximação inferior da classe atual à lista principal
            upper_approx.append({"class": c + 1, "objects": lower_class_objects})
            dominated_inferior_approx.append({"class": c, "objects": dominated_class_objects})
            if c == 0:
                dominated_inferior_approx.pop()

        # Comparação entre upper_approx e dominated_inferior_approx, adicionando objetos faltantes
        for i in range(len(upper_approx) - 1):
            upper_objects_set = set(upper_approx[i]["objects"])
            for obj in dominated_inferior_approx[i]["objects"]:
                if obj not in upper_objects_set:
                    upper_approx[i]["objects"].append(obj)
            # Ordena a lista de objetos do menor valor ao maior
            upper_approx[i]["objects"].sort()

    if union_type == 'upward':
        # Itera sobre cada classe de união com seu índice
        for c, union in enumerate(union_classes):
            # Inicializa a lista para armazenar objetos que pertencem à aproximação inferior da classe atual
            lower_class_objects = []
            dominated_class_objects = []

            # Itera sobre cada objeto na união da classe atual
            for obj in union:
                # Inicializa a variável 'dominated_inferior' para verificar se o objeto deve ser adicionado a 'dominated_class_objects'
                dominated_inferior = True
                # Itera sobre cada objeto no conjunto de dominância
                for dom_obj in dom_set:
                    # Verifica se o objeto atual do conjunto de dominância é o mesmo que o objeto da união
                    if dom_obj["object"] == obj:
                        # Inicializa a variável para verificar se o objeto domina algum objeto de classe superior
                        dominates_superior = True
                        # Itera sobre os objetos dominados pelo objeto atual do conjunto de dominância
                        for dominated in dom_obj["examples"]:
                            # Supondo que a última entrada na lista 'dominated' representa a classe do objeto
                            dominated_class = dominated[-1]
                            if dominated_class <= c + 1:
                                dominates_superior = False
                            if dominated_class > c + 1:
                                dominated_inferior = False
                                break
                        # Se o objeto não domina nenhum objeto de classe superior, adiciona-o à lista de objetos da classe atual
                        if not dominates_superior:
                            lower_class_objects.append(obj)
                        if not dominated_inferior:
                            dominated_class_objects.append(obj)
                        break
                # Se o objeto não domina ninguém, adiciona-o à lista de objetos da classe atual
                if dominates_superior:
                    lower_class_objects.append(obj)
            # Adiciona a aproximação inferior da classe atual à lista principal
            upper_approx.append({"class": c + 1, "objects": lower_class_objects})
            dominated_inferior_approx.append({"class": c + 2, "objects": dominated_class_objects})
            if c == len(union_classes) - 1:
                dominated_inferior_approx.pop()

        # Comparação entre upper_approx e dominated_inferior_approx, ignorando a primeira linha de upper_approx
        for i in range(1, len(upper_approx)):
            upper_objects_set = set(upper_approx[i]["objects"])
            for obj in dominated_inferior_approx[i - 1]["objects"]:
                if obj not in upper_objects_set:
                    upper_approx[i]["objects"].append(obj)
            # Ordena a lista de objetos do menor valor ao maior
            upper_approx[i]["objects"].sort()

    # Retorna a lista com as aproximações inferiores de todas as classes
    return upper_approx

    # Retorna a lista com as aproximações inferiores de todas as classes
    return upper_approx

def boundaries(upper_approx, lower_approx):
    return [
        {"class": i + 1, "objects": list(set(upper["objects"]) - set(lower["objects"]))}
        for i, (upper, lower) in enumerate(zip(upper_approx, lower_approx))  # Calcula as fronteiras das aproximações
    ]

def domlem(lower_upward, lower_downward, infosystem):
    rules = []

    for ld in lower_downward[:-1]:
        print(f"Processing lower_downward class {ld['class']}")
        rules_found = find_rules(ld["objects"], infosystem, "three")
        print(f"Rules found: {rules_found}")
        rules.extend(rules_found)

    for lu in lower_upward[1:]:
        print(f"Processing lower_upward class {lu['class']}")
        rules_found = find_rules(lu["objects"], infosystem, "one")
        print(f"Rules found: {rules_found}")
        rules.extend(rules_found)

    return rules

def find_rules(objects, infosystem, rule_type):
    print("Início de find_rules")
    rules = []
    matrix = infosystem["examples"]
    preferences = [attr["preference"] for attr in infosystem["attributes"]]

    max_iterations = 1000
    iterations = 0

    while objects and iterations < max_iterations:
        print(f"Iteração {iterations + 1}")
        print(f"Objetos restantes: {objects}")
        rule = []
        covered_objects = find_covered_objects(rule, matrix)
        print(f"Objetos inicialmente cobertos: {covered_objects}")

        last_covered_objects = set()

        while not all(obj in objects for obj in covered_objects):
            best_condition = None
            best_score = 0
            for c, pref in enumerate(preferences):
                for e in {row[c + 1] for row in matrix if row[0] in objects}:
                    candidate = create_rule(c + 1, e, pref, matrix, rule_type)
                    score = evaluate_rule(candidate, rule, objects, matrix)
                    print(f"Condição candidata: {candidate}, Pontuação: {score}")
                    if score > best_score:
                        best_score = score
                        best_condition = candidate
            if best_condition and best_score > 0:
                print(f"Melhor condição encontrada: {best_condition}")
                rule.append(best_condition)
                covered_objects = find_covered_objects(rule, matrix)
                print(f"Objetos cobertos atualizados: {covered_objects}")
                if set(covered_objects) == last_covered_objects:
                    print("Nenhum progresso, interrompendo iteração interna para evitar loop infinito.")
                    break
                last_covered_objects = set(covered_objects)
            else:
                break
        print(f"Regra encontrada: {rule}")
        if rule:
            rules.append(rule)
        if not covered_objects:
            break
        objects = list(set(objects) - set(covered_objects))
        iterations += 1

    if iterations >= max_iterations:
        print("Número máximo de iterações atingido, interrompendo para evitar loop infinito.")

    return rules

def create_rule(attribute, condition, preference, matrix, rule_type):
    if rule_type == "one":
        return {
            "attribute": attribute,
            "condition": condition,
            "preference": preference,
            "objects_covered": [row[0] for row in matrix if (row[attribute] >= condition if preference == "gain" else row[attribute] <= condition)]
        }
    elif rule_type == "three":
        return {
            "attribute": attribute,
            "condition": condition,
            "preference": preference,
            "objects_covered": [row[0] for row in matrix if (row[attribute] <= condition if preference == "gain" else row[attribute] >= condition)]
        }

def evaluate_rule(candidate, rule, objects, matrix):
    if not candidate:
        return 0
    covered_objects = find_covered_objects(rule + [candidate], matrix)
    return len(set(covered_objects) & set(objects)) / len(covered_objects) if covered_objects else 0

def find_covered_objects(rule, matrix):
    if not rule:
        return [row[0] for row in matrix]
    return list(reduce(set.intersection, [set(cond["objects_covered"]) for cond in rule]))

# Função para salvar tabelas no Excel
def save_to_excel(file_name, infosystem, downward, upward, dominating, dominated, lower_upward, lower_downward, upper_upward, upper_downward, boundaries_downward, rules):
    with pd.ExcelWriter(file_name) as writer:
        pd.DataFrame(infosystem["examples"], columns=['ID', 'Criterion1', 'Criterion2', 'Criterion3', 'Decision']).to_excel(writer, sheet_name='Examples', index=False)  # Salva os exemplos
        pd.DataFrame({"Downward Unions": downward}).to_excel(writer, sheet_name='Downward Unions')  # Salva as uniões descendentes
        pd.DataFrame({"Upward Unions": upward}).to_excel(writer, sheet_name='Upward Unions')  # Salva as uniões ascendentes
        pd.DataFrame(dominating).to_excel(writer, sheet_name='Dominating Set')  # Salva o conjunto de dominância
        pd.DataFrame(dominated).to_excel(writer, sheet_name='Dominated Set')  # Salva o conjunto de dominados
        pd.DataFrame(lower_upward).to_excel(writer, sheet_name='Lower Approx (Upward)')  # Salva a aproximação inferior (ascendente)
        pd.DataFrame(lower_downward).to_excel(writer, sheet_name='Lower Approx (Downward)')  # Salva a aproximação inferior (descendente)
        pd.DataFrame(upper_upward).to_excel(writer, sheet_name='Upper Approx (Upward)')  # Salva a aproximação superior (ascendente)
        pd.DataFrame(upper_downward).to_excel(writer, sheet_name='Upper Approx (Downward)')  # Salva a aproximação superior (descendente)
        pd.DataFrame(boundaries_downward).to_excel(writer, sheet_name='Boundaries (Downward)')  # Salva as fronteiras (descendentes)
        pd.DataFrame(rules).to_excel(writer, sheet_name='Decision Rules', index=False)  # Salva as regras de decisão

# Função para abrir a janela de salvar arquivo e salvar o Excel
def save_file_dialog():
    root = Tk()  # Cria uma instância da janela principal do Tkinter
    root.withdraw()  # Oculta a janela principal do Tkinter
    file_path = asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])  # Abre uma janela de diálogo para salvar arquivo
    if file_path:
        return file_path  # Retorna o caminho do arquivo
    else:
        return None  # Retorna None se o salvamento for cancelado

# Exemplo aleatório de dados
data = {
    'Criterion1': [1.5, 1.7, 0.5, 0.7, 3, 1, 1, 2.3, 1, 1.7, 2.5, 0.5, 1.2, 2, 1.9, 2.3, 2.7],
    'Criterion2': [3, 5, 2, 0.5, 4.3, 2, 1.2, 3.3, 3, 2.8, 4, 3, 1, 2.4, 4.3, 4, 5.5],
    'Criterion3': [12, 9.5, 2.5, 1.5, 9, 4.5, 8, 9, 5, 3.5, 11, 6, 7, 6, 14, 13, 15],
    'Decision':   [2, 2, 1, 1, 3, 2, 1, 3, 1, 2, 2, 2, 2, 1, 2, 3, 3]
}
preferences = ['gain', 'gain', 'gain']  # Define as preferências dos critérios como ganho

df = pd.DataFrame(data)  # Cria um DataFrame com os dados
infosystem = build_info_system(df, preferences)  # Constrói o sistema de informações
print("infosystem:\n", infosystem, "\n")

# Imprime apenas os exemplos do infosystem
print("Examples from infosystem:\n", tabulate(infosystem["examples"], headers=['ID', 'Criterion1', 'Criterion2', 'Criterion3', 'Decision'], tablefmt='fancy_grid'), "\n")

# Calcula as uniões descendentes e ascendentes das classes
downward = downward_unions_of_classes(infosystem)
upward = upward_unions_of_classes(infosystem)

# Imprime as uniões descendentes e ascendentes das classes
print("Downward Unions of Classes:\n", tabulate(downward, headers='keys', tablefmt='fancy_grid'), "\n")
print("Upward Unions of Classes:\n", tabulate(upward, headers='keys', tablefmt='fancy_grid'), "\n")

# Calcula os conjuntos de dominância e dominados
dominating = dominating_set(infosystem)
dominated = dominated_set(infosystem)

# Imprime os conjuntos de dominância e dominados
print("Dominating Set:\n", tabulate(dominating, headers='keys', tablefmt='fancy_grid'), "\n")
print("Dominated Set:\n", tabulate(dominated, headers='keys', tablefmt='fancy_grid'), "\n")

# Calcula as aproximações inferiores e superiores
lower_downward = lower_approximation(downward, dominating, union_type='downward')
upper_downward = upper_approximation(downward, dominated, union_type='downward')
lower_upward = lower_approximation(upward, dominated, union_type='upward')
upper_upward = upper_approximation(upward, dominating, union_type='upward')

# Calcula as fronteiras das aproximações
boundaries_downward = boundaries(upper_downward, lower_downward)

# Imprime as aproximações inferiores, superiores e as fronteiras
print("Lower Approximations (Downward Unions):\n", tabulate(lower_downward, headers='keys', tablefmt='fancy_grid'), "\n")
print("Upper Approximations (Downward Unions):\n", tabulate(upper_downward, headers='keys', tablefmt='fancy_grid'), "\n")
print("Lower Approximations (Upward Unions):\n", tabulate(lower_upward, headers='keys', tablefmt='fancy_grid'), "\n")
print("Upper Approximations (Upward Unions):\n", tabulate(upper_upward, headers='keys', tablefmt='fancy_grid'), "\n")
print("Boundaries (Downward Unions):\n", tabulate(boundaries_downward, headers='keys', tablefmt='fancy_grid'), "\n")

# Extrai as regras de decisão usando o algoritmo DOMLEM
rules = domlem(lower_upward, lower_downward, infosystem)

# Converte as regras de decisão para uma lista de dicionários para salvar no Excel
rules_list = [{"Rule": rule} for rule in rules]

# Salva todas as tabelas em um arquivo Excel
file_path = save_file_dialog()
if file_path:
    save_to_excel(file_path, infosystem, downward, upward, dominating, dominated, lower_upward, lower_downward, upper_upward, upper_downward, boundaries_downward, rules_list)
    print(f"Resultados salvos em: {file_path}")
else:
    print("O salvamento do arquivo foi cancelado.")

# Imprime as regras extraídas
print("Decision Rules:\n")
for rule in rules:
    print(rule, "\n")
