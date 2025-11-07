import ollama
import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple
from textwrap import dedent

MODEL = ""
HOST = "http://localhost:11434"

DEVELOPERS = {
    0: {"name": "Ana (Senior)", "level": "senior", "cost_hour": 150},
    1: {"name": "Bruno (Mid)", "level": "mid", "cost_hour": 100},
    2: {"name": "Carla (Junior)", "level": "junior", "cost_hour": 70},
    3: {"name": "Daniel (Senior)", "level": "senior", "cost_hour": 160}
}

TASKS = {
    0: {"desc": "Setup CI/CD", "complexity": 20},
    1: {"desc": "Build login screen", "complexity": 8},
    2: {"desc": "Create user CRUD", "complexity": 16},
    3: {"desc": "Refactor payment module", "complexity": 40},
    4: {"desc": "Write API unit tests", "complexity": 12},
    5: {"desc": "Fix mobile visual bug", "complexity": 4},
    6: {"desc": "Optimize DB query", "complexity": 24},
    7: {"desc": "Document architecture", "complexity": 10}
}

NUM_DEVS = len(DEVELOPERS)
NUM_TASKS = len(TASKS)


def query_llm(prompt: str, model: str = MODEL) -> str:
    client = ollama.Client(host=HOST)
    response = client.chat(model=model, messages=[
        {"role": "system", "content": "You are an optimization and software engineering expert."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'] if isinstance(response, dict) else response


def get_fitness_objectives():
    prompt = dedent(f"""
        Act as an experienced Agile Project Manager specializing in team optimization.
        I am trying to optimize the allocation of {NUM_TASKS} tasks for a team of {NUM_DEVS} developers:
        {DEVELOPERS}
        The tasks are:
        {TASKS}
        Please suggest 3 conflicting multi-objective fitness metrics. For each objective:
        1. Name (e.g. 'Minimize Total Cost').
        2. Business logic.
        3. How to calculate it from allocation.
        4. Whether it should be minimized or maximized.
        Format the response clearly.
    """)
    return query_llm(prompt, MODEL)


def evaluate_allocation(individual: List[int]) -> Tuple[float, float, float]:
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


def get_mutation_heuristic():
    prompt = dedent(f"""
        I am using a Genetic Algorithm for the task allocation problem, represented as a list of developer assignments [dev_id_task_0, ...].
        The default mutation operator is random. Suggest a smarter heuristic, for example focusing on overloaded developers.
        Please provide:
        1. Heuristic name.
        2. Logic step by step.
        3. Why this is better than random.
        4. Clear pseudocode.
    """)
    return query_llm(prompt, MODEL)


def bottleneck_mutation(individual: List[int]) -> Tuple[List[int],]:
    workloads = [0.0] * NUM_DEVS
    for task_id, dev_id in enumerate(individual):
        workloads[dev_id] += TASKS[task_id]['complexity']
    bottleneck_dev = np.argmax(workloads)
    idle_dev = np.argmin(workloads)
    if bottleneck_dev == idle_dev:
        task = random.randint(0, NUM_TASKS - 1)
        individual[task] = random.randint(0, NUM_DEVS - 1)
        return individual,
    tasks_bottleneck = [i for i, dev in enumerate(individual) if dev == bottleneck_dev]
    if not tasks_bottleneck:
        return individual,
    task_to_move = random.choice(tasks_bottleneck)
    individual[task_to_move] = idle_dev
    return individual,


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", bottleneck_mutation)
toolbox.register("select", tools.selNSGA2)


def run_optimization():
    pop_size = 100
    generations = 50
    cx_prob = 0.7
    mut_prob = 0.2

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)

    algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=cx_prob, mutpb=mut_prob, ngen=generations,
        stats=stats, halloffame=hof, verbose=True
    )
    return pop, stats, hof


def analyze_solutions(hof):
    for idx, ind in enumerate(hof):
        allocs = {dev_id: [] for dev_id in range(NUM_DEVS)}
        workloads = {dev_id: 0.0 for dev_id in range(NUM_DEVS)}
        for t, d in enumerate(ind):
            allocs[d].append(TASKS[t]['desc'])
            workloads[d] += TASKS[t]['complexity']
        fitness = ind.fitness.values
        print(f"\n--- Solution #{idx + 1} ---")
        print(f"Fitness: Makespan={fitness[0]}h, Cost=R${fitness[1]:.2f}, Penalty={fitness[2]}")
        for dev_id, dev_info in DEVELOPERS.items():
            print(f"  - {dev_info['name']}: {workloads[dev_id]}h -> {allocs[dev_id]}")


def main():
    print("Fitness objectives (LLM):\n", get_fitness_objectives())
    print("Mutation heuristic (LLM):\n", get_mutation_heuristic())
    _, _, pareto_front = run_optimization()
    print("\nPareto Front Solutions:")
    analyze_solutions(pareto_front)


if __name__ == "__main__":
    main()
