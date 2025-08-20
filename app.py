import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import io
from io import BytesIO
from PIL import Image
import base64

# Configuration du modèle
MODEL_PATH = os.path.join('models', 'urgency_classifier_bert')
MAX_LENGTH = 128

# Vérification de l'existence du modèle
if not os.path.exists(MODEL_PATH):
    st.error(f"Erreur : Le modèle n'a pas été trouvé dans le chemin : {os.path.abspath(MODEL_PATH)}")
    st.stop()

# Charger le logo ALTEN Maroc
def get_logo_base64():
    logo_path = "ALTEN-Logo.wine.png"
    try:
        img = Image.open(logo_path)
        # Redimensionner le logo pour qu'il soit plus grand
        img = img.resize((300, 100), Image.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.warning(f"Logo non trouvé : {str(e)}")
        return None

logo_base64 = get_logo_base64()

# Style CSS personnalisé amélioré
st.markdown(f"""
<style>
    :root {{
        --primary: #003366; /* Bleu ALTEN */
        --primary-hover: #002244;
        --secondary: #FF6600; /* Orange ALTEN */
        --urgent: #e63946;
        --not-urgent: #2a9d8f;
        --bg-color: #ffffff;
        --card-shadow: 0 6px 20px rgba(0, 51, 102, 0.1);
    }}
    
    .main {{
        background-color: var(--bg-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Style pour le conteneur principal de Streamlit */
    .main {{
        padding-top: 0 !important;
    }}
    
    /* Style pour le header personnalisé */
    .header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, var(--primary) 0%, #004080 100%);
        color: white;
        padding: 1rem 2rem 1rem calc(2rem + 16px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        height: 120px; /* Augmenté pour accommoder le logo plus grand */
        display: flex;
        align-items: center;
        z-index: 1001;
    }}
    
    /* Cacher la barre d'en-tête par défaut de Streamlit */
    .stApp > header {{
        display: none;
    }}
    
    /* Ajuster le padding du contenu principal */
    .stApp > .block-container {{
        padding-top: 140px !important; /* Augmenté pour le header plus grand */
    }}
    
    .header-title {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    
    .header-subtitle {{
        font-size: 1rem;
        opacity: 0.9;
    }}
    
    .stTextArea>div>div>textarea {{
        min-height: 150px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        padding: 14px;
        font-size: 15px;
        transition: all 0.3s ease;
    }}
    
    .stTextArea>div>div>textarea:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(0, 51, 102, 0.2);
    }}
    
    .stButton>button {{
        width: 100%;
        border-radius: 10px;
        background: var(--primary);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.85rem 1.2rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }}
    
    .stButton>button:hover {{
        background: var(--primary-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .prediction-card {{
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: var(--card-shadow);
        border-top: 6px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .prediction-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 51, 102, 0.15);
    }}
    
    .urgent {{ 
        border-top-color: var(--urgent);
        background: linear-gradient(to right, #ffffff 0%, #fff5f5 100%);
    }}
    
    .not-urgent {{ 
        border-top-color: var(--not-urgent);
        background: linear-gradient(to right, #ffffff 0%, #f5fffd 100%);
    }}
    
    .confidence-bar {{
        height: 12px;
        background: #e5e7eb;
        border-radius: 6px;
        margin: 0.75rem 0 1.75rem;
        overflow: hidden;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 6px;
        transition: width 1s ease-in-out;
    }}
    
    .urgent-fill {{ background: linear-gradient(90deg, #ff7b00, var(--urgent)); }}
    .not-urgent-fill {{ background: linear-gradient(90deg, var(--not-urgent), #4cc9f0); }}
    
    .footer {{
        text-align: center;
        margin-top: 3rem;
        padding: 2rem 0;
        color: #6b7280;
        font-size: 0.9rem;
        border-top: 1px solid #e5e7eb;
    }}
    
    .tabs {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .stTab {{
        transition: all 0.3s ease;
    }}
    
    .stTab:hover {{
        background-color: #f0f7ff;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .result-animation {{
        animation: fadeIn 0.6s ease-out forwards;
    }}
    
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--primary);
    }}
    
    .stExpander > div {{
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
    }}
    
    .stExpander > div:hover {{
        border-color: var(--primary) !important;
    }}
    
    /* Nouveaux styles pour le header amélioré */
    .alten-header {{
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 0 2rem;
    }}
    
    .logo-container {{
        flex-shrink: 0;
    }}
    
    .header-text {{
        flex-grow: 1;
    }}
    
    .header-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
        line-height: 1.2;
    }}
    
    .header-subtitle {{
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Charge le tokenizer et le modèle BERT"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        st.stop()

def predict_urgency(text, tokenizer, model):
    """Effectue une prédiction d'urgence sur le texte donné"""
    try:
        if not text or not text.strip():
            return "Non urgent", 0.0, 0.0  # Retourne 0% pour les deux classes
            
        text = ' '.join(str(text).split())
        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Obtenir les probabilités pour les deux classes
            not_urgent_prob = probs[0][0].item()  # Probabilité pour la classe 0 (Non urgent)
            urgent_prob = probs[0][1].item()      # Probabilité pour la classe 1 (Urgent)
            
            # S'assurer que les probabilités sont dans [0, 1]
            not_urgent_prob = max(0.0, min(1.0, not_urgent_prob))
            urgent_prob = max(0.0, min(1.0, urgent_prob))
            
            # Normaliser pour s'assurer que la somme fait 100%
            total = not_urgent_prob + urgent_prob
            if total > 0:
                not_urgent_prob /= total
                urgent_prob /= total
            
            # Déterminer la classe prédite
            predicted_class = 1 if urgent_prob > not_urgent_prob else 0
            prediction = "Urgent" if predicted_class == 1 else "Non urgent"
            confidence = urgent_prob if predicted_class == 1 else not_urgent_prob
        
        return prediction, not_urgent_prob, urgent_prob
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return "Non urgent", 0.0

def process_excel(file):
    """Traite un fichier Excel et retourne un DataFrame"""
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel : {str(e)}")
        return None

def export_to_excel(df):
    """Convertit un DataFrame en fichier Excel pour téléchargement"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Résultats')
    return output.getvalue()

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="ALTEN Urgence Détecteur",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Masquer le header par défaut de Streamlit
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stApp > header {
                display: none !important;
            }
            .stApp > .block-container {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Header ALTEN avec logo - Design professionnel amélioré
    header_html = f"""
    <div class="header">
        <div class="alten-header">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64 if logo_base64 else ''}" alt="ALTEN Logo" style="height: 80px;">
            </div>
            <div class="header-text">
                <h1 class="header-title">Système d'Analyse et de Gestion des Urgences</h1>
                <p class="header-subtitle">Projet de Stage • ALTEN Maroc</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Chargement du modèle
    tokenizer, model = load_model()
    
    # Styles pour les onglets
    st.markdown("""
    <style>
        /* Style pour les onglets */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0;
            padding: 0;
            border-bottom: none;
            margin-bottom: 2rem;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 4px;
            display: inline-flex;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            padding: 0 2rem;
            font-size: 1rem;
            font-weight: 600;
            color: #4a4a4a;
            background: transparent;
            border: none;
            margin: 0;
            position: relative;
            transition: all 0.2s ease;
            border-radius: 6px;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(0, 0, 0, 0.05);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: white !important;
            color: #003366 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .stTabs [data-baseweb="tab"] > div {{
            padding: 0;
            display: flex;
            align-items: center;
            height: 100%;
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] span {{
            font-size: 1.2em;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Création des onglets
    tab1, tab2, tab3 = st.tabs([
        "🔍 Analyse manuelle", 
        "📊 Analyse par lot",
        "❓ Aide"
    ])
    
    with tab3:
        # Conteneur principal avec largeur maximale
        with st.container():
            # En-tête
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2.5rem;'>
                <h1 style='color: #003366; margin-bottom: 1rem;'>Centre d'Aide</h1>
                <p style='color: #4a5568; font-size: 1.1rem;'>Trouvez rapidement les réponses à vos questions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Section Guide d'utilisation
            with st.expander("📋 Guide d'utilisation", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container():
                        st.markdown("### 🔍 Analyse Manuelle")
                        st.markdown("""
                        1. Saisissez votre demande dans la zone de texte
                        2. Cliquez sur "Analyser" pour évaluer le niveau d'urgence
                        3. Consultez les résultats détaillés et les explications
                        """)
                
                with col2:
                    with st.container():
                        st.markdown("### 📊 Analyse par Lot")
                        st.markdown("""
                        1. Préparez un fichier CSV avec une colonne "demande"
                        2. Téléchargez le fichier pour traitement
                        3. Téléchargez le rapport complet
                        """)
                
                st.info("💡 **Astuce** : Pour des résultats optimaux, formulez vos demandes de manière claire et précise en incluant tous les détails pertinents.")
            
            # Section FAQ
            with st.expander("❓ Foire aux questions", expanded=True):
                faq1 = st.expander("Comment le niveau d'urgence est-il déterminé ?", expanded=False)
                with faq1:
                    st.write("Notre système utilise une intelligence artificielle avancée pour analyser le contenu de votre demande et évaluer son niveau d'urgence en fonction de plusieurs facteurs clés.")
                
                faq2 = st.expander("Quel format dois-je utiliser pour l'analyse par lot ?", expanded=False)
                with faq2:
                    st.write("Préparez un fichier CSV avec une colonne intitulée 'demande' contenant une demande par ligne.")
                
                faq3 = st.expander("Comment interpréter les résultats d'analyse ?", expanded=False)
                with faq3:
                    st.write("Chaque demande est classée comme 'Urgente' ou 'Non urgente' avec un indice de confiance. Plus l'indice est proche de 100%, plus la certitude du modèle est élevée.")
            
            # Section Support
            with st.expander("📞 Contact & Support", expanded=True):
                col_sup1, col_sup2, col_sup3 = st.columns(3)
                
                with col_sup1:
                    st.subheader("📞 Support technique")
                    st.write("Notre équipe est là pour vous aider du lundi au vendredi, de 8h30 à 18h.")
                    st.write("📧 support@alten.ma")
                    st.write("📞 +212 535-730000")
                
                with col_sup2:
                    st.subheader("🏢 Site de Fès")
                    st.write("""
                    ALTEN Maroc
                    
                    Shore Harazem, Route de Fès
                    Gzoula Sidi Mbarek
                    Fès 30000, Maroc
                    """)
                
                with col_sup3:
                    st.subheader("🏢 Siège Social")
                    st.write("""
                    ALTEN Maroc
                    
                    Avenue des FAR, Hay Riad
                    Fès 30000, Maroc
                    """)
            
            # Pied de page
            st.markdown("""
            <div style='text-align: center; padding: 1.5rem; color: #718096; font-size: 0.9rem; margin-top: 2rem;'>
                <p style='margin: 0 0 0.5rem 0;'>Version 1.0.0 | Dernière mise à jour : Juillet 2025</p>
                <p style='margin: 0;'>© 2025 ALTEN Maroc - Tous droits réservés</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab1:
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h1 style="color: #003366; margin-bottom: 0.5rem;">Analyse d'urgence</h1>
            <p style="color: #4a5568; font-size: 1.1rem; margin-bottom: 2rem;">Saisissez votre demande ci-dessous pour évaluer son niveau d'urgence</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("ticket_form"):
            ticket_text = st.text_area(
                "Décrivez le problème ou la demande :",
                height=250,
                placeholder="Exemple : 'URGENT - Le système de production est arrêté depuis ce matin, impactant 50 opérateurs et causant une perte estimée à 15 000€/heure. Intervention immédiate requise.'",
                label_visibility="collapsed",
                help="Décrivez votre demande de manière détaillée pour une analyse plus précise."
            )
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                submitted = st.form_submit_button("Analyser l'urgence", type="primary")
            with col2:
                example_btn = st.form_submit_button("Charger un exemple")
            with col3:
                clear_btn = st.form_submit_button("Effacer")
                
            if example_btn:
                ticket_text = """
                URGENT - Problème critique production
                
                Bonjour équipe support,
                
                Notre ligne de production est complètement arrêtée depuis 8h ce matin 
                suite à une panne du système de contrôle. 
                
                Impact :
                - 50 opérateurs au chômage technique
                - Perte estimée : 15 000€/heure
                - Clients prioritaires affectés
                
                Requiert une intervention immédiate.
                
                Cordialement,
                Responsable Usine - Site Fès
                """
                st.rerun()
            
            if clear_btn:
                ticket_text = ""
                st.rerun()
    
    if submitted and ticket_text.strip():
        with st.spinner("Analyse en cours avec le modèle BERT..."):
            prediction, not_urgent_prob, urgent_prob = predict_urgency(ticket_text, tokenizer, model)
            
            st.markdown("## 📊 Résultats de l'analyse")
            st.markdown('<div class="result-animation">', unsafe_allow_html=True)
            
            prediction_class = "urgent" if prediction == "Urgent" else "not-urgent"
            st.markdown(f'<div class="prediction-card {prediction_class}">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                confidence = urgent_prob if prediction == "Urgent" else not_urgent_prob
                if prediction == "Urgent":
                    st.markdown("""
                    <div style="text-align: center;">
                        <h2 style="color: var(--urgent); margin-bottom: 0.5rem;">
                            🔥 URGENT
                        </h2>
                        <div style="background: #ffecec; border-radius: 50%; width: 100px; height: 100px; margin: 0 auto; display: flex; align-items: center; justify-content: center; border: 4px solid var(--urgent);">
                            <h1 style="color: var(--urgent); margin: 0;">
                                {:.0f}%
                            </h1>
                        </div>
                        <p style="color: #6b7280; margin-top: 0.5rem;">Confiance de la prédiction</p>
                    </div>
                    """.format(confidence * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center;">
                        <h2 style="color: var(--not-urgent); margin-bottom: 0.5rem;">
                            ✅ NON URGENT
                        </h2>
                        <div style="background: #e6f7f4; border-radius: 50%; width: 100px; height: 100px; margin: 0 auto; display: flex; align-items: center; justify-content: center; border: 4px solid var(--not-urgent);">
                            <h1 style="color: var(--not-urgent); margin: 0;">
                                {:.0f}%
                            </h1>
                        </div>
                        <p style="color: #6b7280; margin-top: 0.5rem;">Confiance de la prédiction</p>
                    </div>
                    """.format(confidence * 100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Niveaux de confiance")
                urgent_pct = urgent_prob * 100
                not_urgent_pct = not_urgent_prob * 100
                
                st.markdown(f"""
                <div style="margin-bottom: 2rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: var(--urgent);">Urgent</span>
                        <span style="font-weight: 600;">{urgent_pct:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill urgent-fill" style="width: {urgent_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: var(--not-urgent);">Non urgent</span>
                        <span style="font-weight: 600;">{not_urgent_pct:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill not-urgent-fill" style="width: {not_urgent_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("📝 Analyse détaillée et recommandations", expanded=True):
                if prediction == "Urgent":
                    st.markdown("""
                    <div style="background: #fff5f5; border-left: 4px solid var(--urgent); padding: 1.5rem; border-radius: 0 12px 12px 0;">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div style="background: var(--urgent); color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                !
                            </div>
                            <h4 style="color: var(--urgent); margin: 0;">Attention requise - Priorité Maximale</h4>
                        </div>
                        <p>Ce ticket a été classé comme <strong style="color: var(--urgent);">URGENT</strong> en raison des éléments suivants :</p>
                        <ul style="margin-left: 1.5rem;">
                            <li>Termes indiquant une situation critique (URGENT, panne, arrêt)</li>
                            <li>Impact financier quantifié (15 000€/heure)</li>
                            <li>Effet sur les opérations et le personnel</li>
                            <li>Nécessité d'une intervention rapide</li>
                        </ul>
                        <div style="background: #ffecec; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h5 style="margin: 0 0 0.5rem 0; color: var(--urgent);">🔔 Recommandation ALTEN</h5>
                            <p style="margin: 0;">Traiter en priorité absolue dans les plus brefs délais. Contacter immédiatement l'équipe technique concernée et le responsable de site.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: #f0fdf9; border-left: 4px solid var(--not-urgent); padding: 1.5rem; border-radius: 0 12px 12px 0;">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div style="background: var(--not-urgent); color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                ✓
                            </div>
                            <h4 style="color: var(--not-urgent); margin: 0;">Situation normale - Priorité Standard</h4>
                        </div>
                        <p>Ce ticket a été classé comme <strong style="color: var(--not-urgent);">NON URGENT</strong> car il semble concerner :</p>
                        <ul style="margin-left: 1.5rem;">
                            <li>Une demande d'information ou de documentation</li>
                            <li>Un problème mineur sans impact immédiat</li>
                            <li>Une question de routine ou de maintenance préventive</li>
                            <li>Une requête sans délai critique spécifié</li>
                        </ul>
                        <div style="background: #e6f7f4; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <h5 style="margin: 0 0 0.5rem 0; color: var(--not-urgent);">ℹ️ Recommandation ALTEN</h5>
                            <p style="margin: 0;">Traiter selon les délais standards définis dans le SLA. Planifier la résolution lors du prochain créneau disponible.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📤 Importer des tickets en lot")
        st.markdown("""
        <div style="background: #f8fafc; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
            <h4 style="margin-top: 0;">Comment utiliser l'analyse par lot ?</h4>
            <ol style="margin-bottom: 0;">
                <li>Préparez un fichier Excel avec vos tickets</li>
                <li>Assurez-vous qu'une colonne contient le texte des tickets</li>
                <li>L'application détectera automatiquement les colonnes pertinentes</li>
                <li>Visualisez et exportez les résultats</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Glissez-déposez votre fichier Excel ici",
            type=["xlsx", "xls"],
            key="file_uploader",
            help="Le fichier doit contenir au moins une colonne avec le texte des tickets",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            df = process_excel(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Fichier chargé avec succès ! {len(df)} tickets détectés.")
                
                text_columns = [col for col in df.columns if 'text' in col.lower() or 'ticket' in col.lower() or 'description' in col.lower() or 'contenu' in col.lower()]
                
                if text_columns:
                    selected_col = st.selectbox(
                        "Sélectionnez la colonne contenant le texte à analyser :",
                        text_columns,
                        index=0
                    )
                    
                    if st.button("🚀 Lancer l'analyse des tickets", type="primary", use_container_width=True):
                        with st.spinner("Analyse en cours avec l'IA ALTEN..."):
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, text in enumerate(df[selected_col]):
                                if pd.isna(text) or str(text).strip() == '':
                                    continue
                                    
                                text = str(text).strip()
                                prediction, not_urgent_prob, urgent_prob = predict_urgency(text, tokenizer, model)
                                confidence = urgent_prob if prediction == "Urgent" else not_urgent_prob
                                
                                display_text = text[:100] + '...' if len(text) > 100 else text
                                
                                results.append({
                                    'ID': i + 1,
                                    'Ticket': display_text,
                                    'Prédiction': prediction,
                                    'Confiance (%)': confidence * 100,
                                    'Urgence': '🟢' if prediction == 'Non urgent' else '🔴',
                                    'Texte complet': text
                                })
                                
                                progress = (i + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"Analyse en cours... {i + 1}/{len(df)} tickets traités")
                            
                            results_df = pd.DataFrame(results)
                            
                            st.markdown('<div class="result-animation">', unsafe_allow_html=True)
                            st.success("🎉 Analyse terminée avec succès !")
                            
                            st.markdown("### 📈 Statistiques globales")
                            col1, col2, col3 = st.columns(3)
                            
                            urgent_count = (results_df['Prédiction'] == 'Urgent').sum()
                            avg_confidence = results_df['Confiance (%)'].mean()
                            
                            with col1:
                                st.markdown("""
                                <div class="metric-card">
                                    <h3 style="margin-top: 0; color: var(--urgent);">Tickets Urgents</h3>
                                    <h1 style="color: var(--urgent); margin-bottom: 0;">{}</h1>
                                    <p>{:.1f}% du total</p>
                                </div>
                                """.format(urgent_count, urgent_count/len(df)*100), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                <div class="metric-card">
                                    <h3 style="margin-top: 0; color: var(--not-urgent);">Tickets Non Urgents</h3>
                                    <h1 style="color: var(--not-urgent); margin-bottom: 0;">{}</h1>
                                    <p>{:.1f}% du total</p>
                                </div>
                                """.format(len(df)-urgent_count, (len(df)-urgent_count)/len(df)*100), unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown("""
                                <div class="metric-card">
                                    <h3 style="margin-top: 0; color: var(--primary);">Confiance Moyenne</h3>
                                    <h1 style="color: var(--primary); margin-bottom: 0;">{:.1f}%</h1>
                                    <p>Précision globale</p>
                                </div>
                                """.format(avg_confidence), unsafe_allow_html=True)
                            
                            st.markdown("### 📊 Répartition des tickets")
                            fig = px.pie(
                                results_df, 
                                names='Prédiction',
                                title='',
                                color='Prédiction',
                                color_discrete_map={'Urgent': '#e63946', 'Non urgent': '#2a9d8f'},
                                hole=0.4
                            )
                            fig.update_layout(
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📋 Résultats détaillés")
                            st.dataframe(
                                results_df[['ID', 'Ticket', 'Prédiction', 'Confiance (%)', 'Urgence']],
                                column_config={
                                    'ID': 'N°',
                                    'Ticket': 'Contenu',
                                    'Prédiction': 'Statut',
                                    'Confiance (%)': st.column_config.ProgressColumn(
                                        'Confiance',
                                        format="%.1f%%",
                                        min_value=0,
                                        max_value=100,
                                    ),
                                    'Urgence': 'Niveau'
                                },
                                hide_index=True,
                                use_container_width=True,
                                height=min(500, 50 + len(results_df) * 35)
                            )
                            
                            st.markdown("### 📤 Exporter les résultats")
                            st.markdown("""
                            <div style="background: #f8fafc; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
                                <h4 style="margin-top: 0;">Options d'export</h4>
                                <p>Exportez les résultats pour les partager avec votre équipe ou les intégrer à vos systèmes.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                excel_data = export_to_excel(results_df)
                                st.download_button(
                                    label="💾 Excel complet",
                                    data=excel_data,
                                    file_name=f"alten_urgence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col2:
                                csv_data = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📄 CSV simplifié",
                                    data=csv_data,
                                    file_name=f"alten_urgence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col3:
                                st.button(
                                    "📊 Générer un rapport PDF",
                                    help="Fonctionnalité à venir",
                                    use_container_width=True,
                                    disabled=True
                                )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Aucune colonne de texte n'a été détectée automatiquement.")
                    with st.expander("Afficher les colonnes disponibles"):
                        st.write(df.columns.tolist())
    
    # Pied de page
    st.markdown(f"""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_base64 if logo_base64 else ''}" style="height: 40px; opacity: 0.8;">
        </div>
        <p><strong>Détecteur d'Urgence des Tickets</strong> - Solution IA pour la gestion des priorités</p>
        <p>ALTEN Maroc - Site de Fès • © {datetime.now().year} Tous droits réservés</p>
        <div style="margin-top: 1rem; font-size: 0.8rem; color: #9ca3af;">
            <p>Propulsé par BERT • Version 1.0 • Dernière mise à jour : {datetime.now().strftime("%d/%m/%Y")}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()