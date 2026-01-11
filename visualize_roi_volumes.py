"""
Script pour visualiser les distributions des volumes ROIs.
Crée des graphiques pour analyser les volumes des organes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
CSV_FILE = Path("roi_volumes_analysis.csv")
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('default')
sns.set_palette("husl")

# Mapping des organes
ORGAN_NAMES = {
    1: "GTV",
    2: "PTV", 
    3: "Poumon_Droit",
    4: "Poumon_Gauche",
    5: "Coeur",
    6: "Oesophage",
    7: "Moelle"
}

# Couleurs cohérentes avec les masques
ORGAN_COLORS = {
    "GTV": "#FF0000",           # Rouge
    "PTV": "#FFA500",           # Orange
    "Poumon_Droit": "#00FFFF",  # Cyan
    "Poumon_Gauche": "#00BFFF", # Bleu clair
    "Coeur": "#FF00FF",         # Magenta
    "Oesophage": "#FFFF00",     # Jaune
    "Moelle": "#00FF00"         # Vert
}


def load_data():
    """Charge les données des volumes."""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Fichier {CSV_FILE} introuvable. Exécutez d'abord calculate_roi_volumes.py")
    
    df = pd.read_csv(CSV_FILE)
    print(f"✅ Données chargées: {len(df)} patients")
    return df


def plot_volume_distributions(df):
    """Crée des histogrammes de distribution des volumes."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    for idx, (label, organ_name) in enumerate(ORGAN_NAMES.items()):
        col_name = f"{organ_name}_volume_cm3"
        
        if col_name not in df.columns:
            continue
        
        ax = axes[idx]
        volumes = df[col_name]
        non_zero = volumes[volumes > 0]
        
        if len(non_zero) > 0:
            # Histogramme
            ax.hist(non_zero, bins=30, color=ORGAN_COLORS[organ_name], 
                   alpha=0.7, edgecolor='black')
            
            # Statistiques
            mean_vol = non_zero.mean()
            median_vol = non_zero.median()
            
            ax.axvline(mean_vol, color='red', linestyle='--', linewidth=2, 
                      label=f'Moyenne: {mean_vol:.1f} cm³')
            ax.axvline(median_vol, color='blue', linestyle='--', linewidth=2,
                      label=f'Médiane: {median_vol:.1f} cm³')
            
            ax.set_xlabel('Volume (cm³)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Nombre de patients', fontsize=11, fontweight='bold')
            ax.set_title(f'{organ_name}\n({len(non_zero)}/{len(df)} patients)', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Aucune donnée', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(organ_name, fontsize=12, fontweight='bold')
    
    # Supprimer axes inutilisés
    for idx in range(len(ORGAN_NAMES), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "roi_volumes_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def plot_volume_boxplots(df):
    """Crée des boxplots comparant les volumes des organes."""
    # Préparer les données
    data_for_plot = []
    
    for label, organ_name in ORGAN_NAMES.items():
        col_name = f"{organ_name}_volume_cm3"
        if col_name in df.columns:
            volumes = df[col_name]
            non_zero = volumes[volumes > 0]
            
            for vol in non_zero:
                data_for_plot.append({
                    'Organe': organ_name,
                    'Volume (cm³)': vol
                })
    
    df_plot = pd.DataFrame(data_for_plot)
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Boxplot avec seaborn
    organ_order = list(ORGAN_NAMES.values())
    colors = [ORGAN_COLORS[org] for org in organ_order]
    
    bp = sns.boxplot(data=df_plot, x='Organe', y='Volume (cm³)', 
                     order=organ_order, palette=colors, ax=ax)
    
    # Ajouter points individuels
    sns.stripplot(data=df_plot, x='Organe', y='Volume (cm³)', 
                 order=organ_order, color='black', alpha=0.3, size=3, ax=ax)
    
    ax.set_xlabel('Organe', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volume (cm³)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution des Volumes par Organe (Boxplots)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "roi_volumes_boxplots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def plot_volume_comparison_table(df):
    """Crée un tableau comparatif des volumes moyens."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Préparer données du tableau
    table_data = []
    table_data.append(['Organe', 'N Patients', 'Moyen (cm³)', 'Médian (cm³)', 
                      'Écart-type', 'Min (cm³)', 'Max (cm³)'])
    
    for label, organ_name in ORGAN_NAMES.items():
        col_name = f"{organ_name}_volume_cm3"
        if col_name in df.columns:
            volumes = df[col_name]
            non_zero = volumes[volumes > 0]
            
            if len(non_zero) > 0:
                row = [
                    organ_name,
                    f"{len(non_zero)}/{len(df)}",
                    f"{non_zero.mean():.2f}",
                    f"{non_zero.median():.2f}",
                    f"{non_zero.std():.2f}",
                    f"{non_zero.min():.2f}",
                    f"{non_zero.max():.2f}"
                ]
                table_data.append(row)
    
    # Créer tableau
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style organes (alternance couleurs)
    for i in range(1, len(table_data)):
        organ_name = table_data[i][0]
        color = ORGAN_COLORS.get(organ_name, '#FFFFFF')
        
        cell = table[(i, 0)]
        cell.set_facecolor(color)
        cell.set_text_props(weight='bold')
        
        # Alternance gris clair pour les autres colonnes
        bg_color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
        for j in range(1, len(table_data[0])):
            table[(i, j)].set_facecolor(bg_color)
    
    plt.title('Statistiques des Volumes ROIs par Organe', 
             fontsize=16, fontweight='bold', pad=20)
    
    output_path = OUTPUT_DIR / "roi_volumes_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def plot_organ_presence_chart(df):
    """Crée un graphique de présence des organes."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compter présence
    presence_data = []
    
    for label, organ_name in ORGAN_NAMES.items():
        col_name = f"{organ_name}_volume_cm3"
        if col_name in df.columns:
            volumes = df[col_name]
            count = len(volumes[volumes > 0])
            percentage = (count / len(df)) * 100
            presence_data.append({
                'Organe': organ_name,
                'Count': count,
                'Percentage': percentage
            })
    
    df_presence = pd.DataFrame(presence_data)
    df_presence = df_presence.sort_values('Count', ascending=True)
    
    # Barplot horizontal
    colors_ordered = [ORGAN_COLORS[org] for org in df_presence['Organe']]
    bars = ax.barh(df_presence['Organe'], df_presence['Count'], 
                   color=colors_ordered, edgecolor='black', linewidth=1.5)
    
    # Ajouter pourcentages
    for i, (count, pct) in enumerate(zip(df_presence['Count'], df_presence['Percentage'])):
        ax.text(count + 1, i, f'{count} ({pct:.1f}%)', 
               va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Nombre de patients', fontsize=13, fontweight='bold')
    ax.set_ylabel('Organe', fontsize=13, fontweight='bold')
    ax.set_title(f'Présence des Organes dans le Dataset\n(Total: {len(df)} patients)', 
                fontsize=15, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, len(df) + 10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "roi_organ_presence.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {output_path}")
    plt.close()


def main():
    """Fonction principale."""
    print("=== VISUALISATION DES VOLUMES ROIs ===\n")
    
    # Charger données
    df = load_data()
    
    print("\nGénération des visualisations...")
    
    # 1. Distributions
    print("  1. Histogrammes de distribution...")
    plot_volume_distributions(df)
    
    # 2. Boxplots
    print("  2. Boxplots comparatifs...")
    plot_volume_boxplots(df)
    
    # 3. Tableau statistique
    print("  3. Tableau des statistiques...")
    plot_volume_comparison_table(df)
    
    # 4. Graphique de présence
    print("  4. Graphique de présence...")
    plot_organ_presence_chart(df)
    
    print("\n✅ Toutes les visualisations ont été générées!")
    print(f"📁 Dossier: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
