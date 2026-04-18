#!/usr/bin/env python3
"""
Modelo Básico de Swing Classification
Entrena un modelo para clasificar tipo de swing basado en features
Genera curva de aprendizaje y métricas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURACIÓN ========================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = Path(r'c:\Users\USER\Documents\Caddies 2.0\model_results')

# ======================== FUNCIONES ========================

def load_data():
    """Carga datos de GolfDB.csv"""
    print("📊 Cargando datos...")
    
    csv_path = Path(r'c:\Users\USER\Documents\Caddies 2.0\GolfDB.csv')
    df = pd.read_csv(csv_path)
    
    print(f"   ✓ {len(df)} registros cargados")
    print(f"   ✓ Columnas: {list(df.columns)}")
    
    return df

def prepare_features(df):
    """Prepara features para el modelo"""
    print("\n🔧 Preparando features...")
    
    # Selecciona features disponibles
    feature_columns = ['player', 'sex', 'club', 'view', 'slow']
    
    df_features = df[feature_columns].copy()
    
    # Codifica variables categóricas
    label_encoders = {}
    categorical_cols = ['player', 'sex', 'club', 'view']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_features[col] = le.fit_transform(df_features[col])
        label_encoders[col] = le
    
    # Target: Predecir tipo de club
    y = df_features['club'].values
    X = df_features.drop('club', axis=1).values
    
    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Clases a predecir: {len(np.unique(y))}")
    print(f"   ✓ Clases: {label_encoders['club'].classes_}")
    
    return X, y, label_encoders

def split_data(X, y):
    """Divide datos en train/test"""
    print("\n📂 Dividiendo datos...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   ✓ Train set: {len(X_train)} muestras ({100*(1-TEST_SIZE):.0f}%)")
    print(f"   ✓ Test set:  {len(X_test)} muestras ({100*TEST_SIZE:.0f}%)")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entrena el modelo"""
    print("\n🤖 Entrenando modelo...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print(f"   ✓ Modelo entrenado")
    print(f"   ✓ Features importancia:")
    feature_names = ['player', 'sex', 'view', 'slow']
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), 
                           key=lambda x: x[1], reverse=True):
        print(f"      • {name}: {imp:.4f}")
    
    return model

def generate_learning_curve(X_train, y_train, model):
    """Genera curva de aprendizaje"""
    print("\n📈 Generando curva de aprendizaje...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=CV_FOLDS,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    print(f"   ✓ Curva de aprendizaje generada")
    print(f"   ✓ Accuracy train final: {train_mean[-1]:.4f}")
    print(f"   ✓ Accuracy validación final: {val_mean[-1]:.4f}")
    
    return train_sizes, train_mean, train_std, val_mean, val_std

def evaluate_model(model, X_test, y_test, label_encoders):
    """Evalúa el modelo en test set"""
    print("\n🎯 Evaluando modelo...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Accuracy en Test Set: {accuracy:.4f}")
    
    # Reporte detallado
    print("\n📊 Reporte de Clasificación:")
    print("-" * 60)
    
    class_names = label_encoders['club'].classes_
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
    
    # Precisión, Recall, F1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=np.unique(y_test)
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'y_test': y_test,
        'y_pred': y_pred,
        'class_names': class_names
    }
    
    return metrics

def plot_learning_curve(train_sizes, train_mean, train_std, 
                       val_mean, val_std):
    """Grafica la curva de aprendizaje"""
    print("\n📊 Creando gráficos...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train y validation curves
    ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', 
           label='Training Accuracy', linewidth=2, markersize=8)
    ax.fill_between(train_sizes, 
                    train_mean - train_std, 
                    train_mean + train_std, 
                    alpha=0.2, color='#2E86AB')
    
    ax.plot(train_sizes, val_mean, 's-', color='#A23B72', 
           label='Validation Accuracy', linewidth=2, markersize=8)
    ax.fill_between(train_sizes, 
                    val_mean - val_std, 
                    val_mean + val_std, 
                    alpha=0.2, color='#A23B72')
    
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Curva de Aprendizaje - Golf Swing Classifier', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 1.05])
    
    plt.tight_layout()
    
    # Guarda
    output_path = OUTPUT_DIR / 'learning_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path}")
    
    return fig

def plot_confusion_matrix(metrics):
    """Grafica matriz de confusión"""
    y_test = metrics['y_test']
    y_pred = metrics['y_pred']
    class_names = metrics['class_names']
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path}")
    
    return fig

def plot_metrics(metrics):
    """Grafica métricas por clase"""
    class_names = metrics['class_names']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Precision
    axes[0].bar(class_names, precision, color='#2E86AB', alpha=0.7)
    axes[0].set_ylabel('Precision', fontweight='bold')
    axes[0].set_title('Precision por Clase', fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1].bar(class_names, recall, color='#A23B72', alpha=0.7)
    axes[1].set_ylabel('Recall', fontweight='bold')
    axes[1].set_title('Recall por Clase', fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # F1-Score
    axes[2].bar(class_names, f1, color='#F18F01', alpha=0.7)
    axes[2].set_ylabel('F1-Score', fontweight='bold')
    axes[2].set_title('F1-Score por Clase', fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'metrics_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path}")
    
    return fig

def plot_training_history(model, X_train, y_train):
    """Simula epochs/historia de entrenamiento"""
    print("\n⏱️  Simulando historia de entrenamiento por épocas...")
    
    # Entrena modelo múltiples veces para simular épocas
    n_epochs = 20
    train_losses = []
    train_accs = []
    
    # Usa subset para visualización
    sample_size = min(len(X_train), 500)
    X_sample = X_train[:sample_size]
    y_sample = y_train[:sample_size]
    
    for epoch in range(1, n_epochs + 1):
        # Crea nuevo modelo con n_estimators progresivo
        model_epoch = RandomForestClassifier(
            n_estimators=min(10 + epoch * 5, 100),
            max_depth=10 + epoch,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_epoch.fit(X_sample, y_sample)
        
        # Calcula métrica
        y_pred = model_epoch.predict(X_sample)
        acc = accuracy_score(y_sample, y_pred)
        train_accs.append(acc)
        
        # Simula pérdida (inversa de accuracy)
        loss = 1 - acc
        train_losses.append(loss)
    
    # Grafica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss por época
    ax1.plot(range(1, n_epochs + 1), train_losses, 'o-', 
            color='#E63946', linewidth=2, markersize=6)
    ax1.set_xlabel('Épocas', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pérdida (Loss)', fontsize=12, fontweight='bold')
    ax1.set_title('Pérdida por Época', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Accuracy por época
    ax2.plot(range(1, n_epochs + 1), train_accs, 's-', 
            color='#06D6A0', linewidth=2, markersize=6)
    ax2.set_xlabel('Épocas', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy por Época', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'training_epochs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path}")
    
    return fig

def print_summary(metrics):
    """Imprime resumen final"""
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL DEL ENTRENAMIENTO")
    print("="*60)
    
    accuracy = metrics['accuracy']
    precision_avg = np.mean(metrics['precision'])
    recall_avg = np.mean(metrics['recall'])
    f1_avg = np.mean(metrics['f1'])
    
    print(f"\n🎯 Métricas Generales:")
    print(f"   • Accuracy:        {accuracy:.4f}")
    print(f"   • Precision Avg:   {precision_avg:.4f}")
    print(f"   • Recall Avg:      {recall_avg:.4f}")
    print(f"   • F1-Score Avg:    {f1_avg:.4f}")
    
    print(f"\n📈 Datos:")
    print(f"   • Clases:          {len(metrics['class_names'])}")
    print(f"   • Muestras test:   {len(metrics['y_test'])}")
    
    print(f"\n💾 Resultados guardados en:")
    print(f"   {OUTPUT_DIR}")
    
    print("\n" + "="*60 + "\n")

# ======================== MAIN ========================

def main():
    """Ejecuta el pipeline completo"""
    print("\n" + "="*60)
    print("⛳ ENTRENAMIENTO DE MODELO - GOLF SWING CLASSIFIER")
    print("="*60 + "\n")
    
    # Crea directorio de output
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    try:
        # 1. Carga datos
        df = load_data()
        
        # 2. Prepara features
        X, y, label_encoders = prepare_features(df)
        
        # 3. Divide datos
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 4. Entrena modelo
        model = train_model(X_train, y_train)
        
        # 5. Genera curva de aprendizaje
        train_sizes, train_mean, train_std, val_mean, val_std = \
            generate_learning_curve(X_train, y_train, model)
        
        # 6. Evalúa modelo
        metrics = evaluate_model(model, X_test, y_test, label_encoders)
        
        # 7. Crea gráficos
        print("\n📊 Generando visualizaciones...")
        
        # Curva de aprendizaje
        plot_learning_curve(train_sizes, train_mean, train_std, 
                          val_mean, val_std)
        
        # Matriz de confusión
        plot_confusion_matrix(metrics)
        
        # Métricas por clase
        plot_metrics(metrics)
        
        # Historia de épocas
        plot_training_history(model, X_train, y_train)
        
        # 8. Resumen
        print_summary(metrics)
        
        print("✅ ¡Entrenamiento completado exitosamente!")
        print("\nVe a la carpeta 'model_results' para ver los gráficos:")
        print("  • learning_curve.png")
        print("  • training_epochs.png")
        print("  • confusion_matrix.png")
        print("  • metrics_by_class.png")
        
        # Muestra gráficos
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
