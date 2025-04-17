import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_contour, plot_slice

def load_automl_results(results_path):
    """Carrega resultados de otimização do AutoML"""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_automl_results(results_path, output_dir=None):
    """Gera visualizações dos resultados do AutoML"""
    results = load_automl_results(results_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrair dados dos trials
    trials_df = pd.DataFrame(results['trials'])
    
    # Histórico de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(trials_df['number'], trials_df['value'], 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel(results['metric'])
    plt.title(f'AutoML Optimization History - {results["study_name"]}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{results['study_name']}_history.png"))
    
    # Top parâmetros
    param_cols = list(results['best_params'].keys())
    if len(param_cols) > 0:
        # Correlação entre valores de parâmetros e métricas
        plt.figure(figsize=(12, 8))
        for i, param in enumerate(param_cols[:min(6, len(param_cols))]):
            plt.subplot(2, 3, i+1)
            plt.scatter(trials_df['value'], pd.to_numeric(trials_df['params'].apply(lambda x: x.get(param, 0))))
            plt.xlabel(results['metric'])
            plt.ylabel(param)
            plt.title(f'Impact of {param}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{results['study_name']}_param_impact.png"))

def create_study_from_results(results_path):
    """Recria um objeto de estudo Optuna a partir de resultados salvos"""
    results = load_automl_results(results_path)
    
    # Criar um estudo em memória
    study = optuna.create_study(
        study_name=results["study_name"],
        direction=results["direction"]
    )
    
    # Adicionar trials ao estudo
    for trial_data in results["trials"]:
        study.add_trial(
            optuna.trial.create_trial(
                params=trial_data["params"],
                value=trial_data["value"],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
    
    return study

def generate_optuna_visualizations(results_path, output_dir=None):
    """Gera visualizações do Optuna para os resultados"""
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar resultados e criar estudo
    study = create_study_from_results(results_path)
    results = load_automl_results(results_path)
    
    # Histórico de otimização
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, f"{results['study_name']}_optuna_history.png"))
    
    # Importância dos parâmetros
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, f"{results['study_name']}_param_importance.png"))
    
    # Gráfico de contorno para os 2 parâmetros mais importantes
    try:
        fig = plot_contour(study)
        fig.write_image(os.path.join(output_dir, f"{results['study_name']}_contour.png"))
    except Exception:
        pass
    
    # Slice plot
    try:
        fig = plot_slice(study)
        fig.write_image(os.path.join(output_dir, f"{results['study_name']}_slice.png"))
    except Exception:
        pass