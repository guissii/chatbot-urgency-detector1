import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def create_limitations_figure():
    # Données
    metrics = {
        'Faux Négatifs': 2.1,
        'Faux Positifs': 1.7,
        'Temps par ticket (ms)': 37.4
    }
    
    class_accuracy = {
        'Urgents': 91.2,
        'Non Urgents': 98.3
    }
    
    # Création de la figure
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    
    # Graphique 1: Taux d'erreur
    ax1 = plt.subplot(gs[0, 0])
    colors = ['#e74c3c', '#f39c12', '#3498db']
    ax1.bar(metrics.keys(), metrics.values(), color=colors)
    ax1.set_title('Taux d\'Erreur et Performance', pad=15, fontweight='bold')
    ax1.set_ylabel('Pourcentage / ms')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Ajout des valeurs
    for i, v in enumerate(metrics.values()):
        suffix = '%' if i < 2 else 'ms'
        ax1.text(i, v + 0.5, f'{v}{suffix}', ha='center')
    
    # Graphique 2: Précision par classe
    ax2 = plt.subplot(gs[0, 1])
    colors = ['#e74c3c', '#2ecc71']
    ax2.bar(class_accuracy.keys(), class_accuracy.values(), color=colors)
    ax2.set_title('Précision par Classe', pad=15, fontweight='bold')
    ax2.set_ylim(85, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    for i, v in enumerate(class_accuracy.values()):
        ax2.text(i, v - 3, f'{v}%', ha='center', color='white', fontweight='bold')
    
    # Texte des limitations
    ax3 = plt.subplot(gs[1, :])
    ax3.axis('off')
    
    limitations = [
        "1. ANALYSE DES ERREURS",
        f"• {int(8349*0.021)} faux négatifs (2.1%) : tickets urgents non détectés",
        f"• {int(8349*0.017)} faux positifs (1.7%) : alertes inutiles",
        "• Traitement : 5min12s pour 8,349 tickets (37.4ms/ticket)",
        "",
        "2. LIMITATIONS TECHNIQUES",
        "• Précision urgents : 91.2% vs 98.3% pour non urgents",
        "• Performance variable selon la complexité des tickets",
        "• Dépendance forte aux données d'entraînement",
        "",
        "3. PERSPECTIVES D'AMÉLIORATION",
        "• Optimisation du temps de traitement (<30ms/ticket)",
        "• Réduction des faux négatifs sous 1.5%",
        "• Amélioration de la précision des urgents à >93%"
    ]
    
    ax3.text(0.02, 0.98, "\n".join(limitations), 
             va='top', 
             linespacing=1.8,
             fontfamily='monospace',
             bbox=dict(facecolor='#f8f9fa', 
                      edgecolor='#dee2e6', 
                      boxstyle='round,pad=1'))
    
    # Titre principal
    plt.suptitle('Figure 9 : Analyse des Limites et Perspectives d\'Amélioration', 
                fontsize=16, 
                y=0.98)
    
    plt.tight_layout()
    plt.savefig('figure9_limitations_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_limitations_figure()