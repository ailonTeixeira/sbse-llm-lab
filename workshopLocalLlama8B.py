

import ollama
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from textwrap import dedent


# Troca do modelo para comparação! (instalados localmente no Ollama)
LLM_MODEL = "llama3:latest"      
OLLAMA_HOST = "http://localhost:11434"  # padrão Ollama

def ask_ollama(prompt: str, model: str = LLM_MODEL) -> str:
    """Envia um prompt para o modelo escolhido via Ollama local."""
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(model=model, messages=[
        {"role": "system", "content": "Você é um especialista em otimização e engenharia de software."},
        {"role": "user", "content": prompt},
    ])
    return response['message']['content'] if isinstance(response, dict) else response


DEVELOPERS: Dict[int, Dict] = {
    0: {"name": "Ana (Sênior)", "level": "senior", "cost_hour": 150},
    1: {"name": "Bruno (Pleno)", "level": "mid", "cost_hour": 100},
    2: {"name": "Carla (Júnior)", "level": "junior", "cost_hour": 70},
    3: {"name": "Daniel (Sênior)", "level": "senior", "cost_hour": 160},
}

TASKS: Dict[int, Dict] = {
    0: {"desc": "Configurar CI/CD", "complexity": 20},
    1: {"desc": "Desenvolver tela de login", "complexity": 8},
    2: {"desc": "Criar CRUD de usuários", "complexity": 16},
    3: {"desc": "Refatorar módulo de pagamento", "complexity": 40},
    4: {"desc": "Escrever testes unitários para API", "complexity": 12},
    5: {"desc": "Corrigir bug visual no mobile", "complexity": 4},
    6: {"desc": "Otimizar query do banco de dados", "complexity": 24},
    7: {"desc": "Documentar a arquitetura", "complexity": 10},
}

NUM_DEVS = len(DEVELOPERS)
NUM_TASKS = len(TASKS)

print(f"\n👥 Temos {NUM_DEVS} desenvolvedores e {NUM_TASKS} tarefas.")


def get_fitness_objectives_from_llm(model=LLM_MODEL) -> str:
    """Prompt para LLM para sugerir objetivos de fitness."""
    prompt = dedent(f"""
    Aja como um Gerente de Projetos Ágil experiente e especialista em otimização de equipes.

    Estou tentando otimizar a alocação de {NUM_TASKS} tarefas para uma equipe de {NUM_DEVS} desenvolvedores, cujas características são:
    {DEVELOPERS}

    As tarefas são:
    {TASKS}

    Preciso de sua ajuda para definir uma função de fitness multi-objetivo. Por favor, sugira 3 objetivos conflitantes que definem uma 'boa' alocação. Para cada objetivo, explique:
    1. O nome do objetivo (ex: 'Minimizar Custo Total').
    2. A lógica de negócio por trás dele.
    3. Como calculá-lo matematicamente a partir da lista de alocação.
    4. Se o objetivo deve ser minimizado ou maximizado.

    Formate sua resposta de forma clara e estruturada.
    """)
    return ask_ollama(prompt, model)


print("\n💡 Sugestão do LLM para os Objetivos de Fitness:")
obj_fitness = get_fitness_objectives_from_llm()
print(obj_fitness)


from deap import base, creator, tools, algorithms

def evaluate_allocation(individual: List[int]) -> Tuple[float, float, float]:
    """Calcula MakeSpan, Custo Total, Penalidade por Alocação Incorreta."""
    workload = [0.0] * NUM_DEVS
    total_cost = 0.0
    mismatch_penalty = 0.0

    for task_id, dev_id in enumerate(individual):
        task = TASKS[task_id]
        dev = DEVELOPERS[dev_id]

        workload[dev_id] += task['complexity']
        total_cost += task['complexity'] * dev['cost_hour']

        if dev['level'] == 'junior' and task['complexity'] > 20:
            mismatch_penalty += 100
        elif dev['level'] == 'senior' and task['complexity'] < 8:
            mismatch_penalty += 25

    makespan = max(workload)
    return makespan, total_cost, mismatch_penalty

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()
toolbox.register("attr_dev", random.randint, 0, NUM_DEVS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_dev, n=NUM_TASKS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_allocation)

print("\n✅ Função de fitness e DEAP configurados.")


def get_mutation_operator_from_llm(model=LLM_MODEL) -> str:
    """Prompt para LLM sugerir operador de mutação heurístico."""
    cot_prompt = dedent(f"""
    Estou usando um Algoritmo Genético para o problema de alocação de tarefas, representado por uma lista de alocações: [dev_id_para_tarefa_0, dev_id_para_tarefa_1, ...].

    O operador de mutação padrão é o 'mutUniformInt', re-alocando tarefas aleatoriamente. Sugira uma heurística mais inteligente: por exemplo, direcionar a mutação para desenvolvedores sobrecarregados.

    Descreva:
    1. Nome da heurística.
    2. Lógica passo a passo.
    3. Por que ela é melhor que a mutação aleatória.
    4. Pseudocódigo claro.
    """)
    return ask_ollama(cot_prompt, model)

print("\n💡 Sugestão do LLM para o Operador de Mutação:")
mut_op = get_mutation_operator_from_llm()
print(mut_op)


def bottleneck_mutation(individual: List[int]) -> Tuple[List[int],]:
    """
    Heurística: move tarefa do dev mais ocupado para o menos ocupado.
    """
    workloads = [0.0] * NUM_DEVS
    for task_id, dev_id in enumerate(individual):
        workloads[dev_id] += TASKS[task_id]['complexity']
    bottleneck_dev_id = np.argmax(workloads)
    idle_dev_id = np.argmin(workloads)
    if bottleneck_dev_id == idle_dev_id:   # todos iguais
        task_to_mutate = random.randint(0, NUM_TASKS - 1)
        individual[task_to_mutate] = random.randint(0, NUM_DEVS - 1)
        return individual,
    tasks_of_bottleneck = [i for i, dev in enumerate(individual) if dev == bottleneck_dev_id]
    if not tasks_of_bottleneck:
        return individual,
    task_to_move = random.choice(tasks_of_bottleneck)
    individual[task_to_move] = idle_dev_id
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", bottleneck_mutation)
toolbox.register("select", tools.selNSGA2)

print("\n✅ Operador de mutação inteligente registrado.")


def run_optimization():
    """Executa Algoritmo Genético NSGA-II"""
    population_size = 100
    generations = 50
    cx_prob = 0.7
    mut_prob = 0.2

    pop = toolbox.population(n=population_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=population_size, lambda_=population_size,
        cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )
    return pop, stats, hof

final_pop, log, pareto_front = run_optimization()
print(f"\n🎉 Otimização concluída! Encontradas {len(pareto_front)} soluções na Fronteira de Pareto.")



def prettify_results(hof):
    for idx, ind in enumerate(hof):
        allocs = {dev_id: [] for dev_id in range(NUM_DEVS)}
        workloads = {dev_id: 0.0 for dev_id in range(NUM_DEVS)}
        for t, d in enumerate(ind):
            allocs[d].append(TASKS[t]['desc'])
            workloads[d] += TASKS[t]['complexity']
        fitness = ind.fitness.values
        print(f"\n--- Solução #{idx+1} ---")
        print(f"Fitness: Makespan={fitness[0]}h, Custo=R${fitness[1]:.2f}, Penalidade={fitness[2]}")
        for dev_id, dev_info in DEVELOPERS.items():
            print(f"  - {dev_info['name']}: {workloads[dev_id]}h -> {allocs[dev_id]}")

print("\n--- Análise das Soluções na Fronteira Pareto ---")
prettify_results(pareto_front)

