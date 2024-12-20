import json
import matplotlib.pyplot as plt

def load_metrics(files):
    """
    Загружает метрики из JSON-файлов.
    
    Args:
        files (list): Список путей к файлам метрик.
        
    Returns:
        dict: Словарь с метриками для каждого эксперимента.
    """
    results = {}
    for file in files:
        with open(file, "r") as f:
            metrics = json.load(f)
            results[file] = metrics
    return results

def plot_metrics(metrics):
    """
    Строит графики метрик для сравнения между экспериментами.
    
    Args:
        metrics (dict): Словарь метрик, загруженный из файлов.
    """
    experiments = list(metrics.keys())
    metrics_names = list(metrics[experiments[0]].keys())

    for metric in metrics_names:
        values = [metrics[exp][metric] for exp in experiments]
        plt.plot(experiments, values, marker='o', label=metric)

    plt.xlabel("Experiments")
    plt.ylabel("Metrics")
    plt.title("Metrics Comparison")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Список файлов метрик
    files = ["metrics.json", "metrics_exp1.json", "metrics_exp2.json"]
    
    # Загрузка и визуализация метрик
    metrics = load_metrics(files)
    plot_metrics(metrics)
