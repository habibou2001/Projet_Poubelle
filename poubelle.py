# app.py
import streamlit as st
import os
import numpy as np
from PIL import Image
import io
import time

# Configuration pour éviter les problèmes OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Import sécurisé d'OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ OpenCV non disponible: {e}")
    CV2_AVAILABLE = False

# Import sécurisé d'Ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Ultralytics non disponible: {e}")
    ULTRALYTICS_AVAILABLE = False

# ---------------------------------------
# 🎨 CONFIG INTERFACE MODERNE
# ---------------------------------------
st.set_page_config(
    page_title="Détection Intelligente de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 CSS custom - Design moderne avec cartes orange/corail
custom_css = """
<style>
    /* Reset et fond principal */
    .main {
        background: linear-gradient(135deg, #2a1a0f 0%, #3d2815 100%);
        background-attachment: fixed;
    }
    
    /* Container principal élargi */
    .main .block-container {
        background: #2a1a0f;
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.4);
        max-width: 95%;
        border: 1px solid #7c5c34;
    }
    
    /* Header principal centré */
    .main-header {
        background: linear-gradient(135deg, #dd6b20 0%, #ed8936 100%);
        color: white;
        padding: 4rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 2px solid #f6ad55;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        position: relative;
        color: #fffaf0;
    }
    
    .main-subtitle {
        font-size: 1.6rem;
        opacity: 0.95;
        font-weight: 300;
        position: relative;
        color: #ffe8d6;
    }
    
    /* Barre d'outils supérieure */
    .toolbar {
        background: rgba(221, 107, 32, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 1px solid #f6ad55;
        color: white;
    }
    
    /* Boutons modernes */
    .stButton>button {
        background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 25px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.6);
        background: linear-gradient(135deg, #f6ad55 0%, #fbd38d 100%);
    }
    
    /* Bouton de téléchargement */
    .download-btn {
        background: linear-gradient(135deg, #d69e2e 0%, #ecc94b 100%) !important;
        color: #2a1a0f !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 12px 25px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(214, 158, 46, 0.4) !important;
    }
    
    .download-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(214, 158, 46, 0.6) !important;
        background: linear-gradient(135deg, #ecc94b 0%, #f6e05e 100%) !important;
    }
    
    /* Cartes de contenu en ORANGE/CORAIL */
    .content-card {
        background: linear-gradient(135deg, #dd6b20 0%, #ed8936 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 2px solid #f6ad55;
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    /* Zone d'upload stylisée */
    .upload-section {
        background: linear-gradient(135deg, #dd6b20 0%, #ed8936 100%);
        color: white;
        border: 3px dashed #f6ad55;
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-section:hover {
        background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
        border-color: #fbd38d;
        transform: scale(1.02);
    }
    
    /* Badges de résultats */
    .detection-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        margin: 8px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(237, 137, 54, 0.4);
        font-size: 1.1rem;
        border: 1px solid #f6ad55;
    }
    
    .confidence-bar-container {
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #f6ad55;
        backdrop-filter: blur(10px);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        height: 12px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Statistiques */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        min-width: 150px;
        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.4);
        border: 1px solid #f6ad55;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        display: block;
        color: #fffaf0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        color: #ffe8d6;
    }
    
    /* Textes dans les cartes */
    .content-card h1, .content-card h2, .content-card h3, 
    .content-card h4, .content-card h5, .content-card h6 {
        color: #fffaf0 !important;
    }
    
    .content-card p, .content-card div {
        color: #ffe8d6 !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------
# 🧠 CHARGEMENT DU MODEL YOLO
# ---------------------------------------
MODEL_PATH = "models/best.pt"

def ensure_models_directory():
    """Crée le dossier models s'il n'existe pas"""
    os.makedirs("models", exist_ok=True)
    return os.path.exists("models")

# Clear cache pour forcer le rechargement
@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle YOLO avec gestion du cache"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modèle non trouvé: {MODEL_PATH}")
        return None
    
    try:
        model = YOLO(MODEL_PATH)
        st.success("✅ Modèle YOLO chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None

# Initialisation
ensure_models_directory()
model = None

# Tentative de chargement du modèle seulement si disponible
if ULTRALYTICS_AVAILABLE:
    model = load_model()
else:
    st.error("❌ Ultralytics non disponible - Impossible de charger le modèle")

# ---------------------------------------
# 🖥️ HEADER PRINCIPAL
# ---------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🗑️ Détection Intelligente</div>
    <div class="main-subtitle">IA Avancée · Détection en Temps Réel · Classification Automatique</div>
</div>
""", unsafe_allow_html=True)

# Avertissements de dépendances
if not CV2_AVAILABLE:
    st.warning("""
    ⚠️ **OpenCV non disponible** 
    - L'affichage des images annotées sera limité
    - La détection fonctionne normalement
    """)

if not ULTRALYTICS_AVAILABLE:
    st.error("""
    ❌ **Ultralytics non disponible**
    - Impossible de charger les modèles YOLO
    - Vérifiez l'installation des dépendances
    """)

# ---------------------------------------
# 📥 SECTION TÉLÉCHARGEMENT DU MODÈLE
# ---------------------------------------
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("### 🚀 Configuration du Modèle IA")

if model is None:
    st.error("""
    ❌ **Modèle introuvable**
    
    Pour utiliser l'application :
    1. Placez votre fichier `best.pt` dans le dossier `models/`
    2. Le modèle doit s'appeler `best.pt` et être placé dans le dossier `models/`
    """)
    
    # Option pour uploader un modèle
    st.markdown("### 📤 Uploader un modèle")
    uploaded_model = st.file_uploader(
        "Téléchargez votre modèle YOLO (.pt)",
        type=["pt"],
        key="model_uploader"
    )
    
    if uploaded_model is not None:
        try:
            # Sauvegarder le modèle uploadé
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.success("✅ Modèle téléchargé avec succès!")
            st.rerun()  # Recharger la page
        except Exception as e:
            st.error(f"❌ Erreur lors du téléchargement: {str(e)}")
            
else:
    st.success("✅ **Modèle chargé avec succès!**")
    
    # Informations sur le modèle
    col_info, col_download = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        ### 📋 Informations du Modèle
        - **Type**: YOLOv8
        - **Fonction**: Détection de poubelles
        - **Statut**: ✅ Opérationnel
        """)
        
        # Affichage des classes détectables
        if hasattr(model, 'names') and model.names:
            st.markdown("### 🏷️ Classes Détectables")
            classes = list(model.names.values())
            classes_text = ", ".join(classes)
            st.markdown(f"**Objets reconnus:** {classes_text}")
    
    with col_download:
        st.markdown("### 📥 Téléchargement")
        
        # Bouton de téléchargement du modèle actuel
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model_data = f.read()
            
            st.download_button(
                label="💾 Télécharger le Modèle",
                data=model_data,
                file_name="best.pt",
                mime="application/octet-stream",
                help="Téléchargez le modèle YOLO de détection de poubelles",
                use_container_width=True,
                key="download_model"
            )
            
            # Informations sur le modèle
            file_size = len(model_data) / (1024 * 1024)  # Taille en MB
            st.info(f"**Taille du modèle:** {file_size:.1f} MB")
    
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 📸 SECTION UPLOAD D'IMAGE
# ---------------------------------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### 📸 Analyse d'Image")
st.markdown("""
<div style='text-align: center;'>
    <h3 style='color: #fffaf0; margin-bottom: 1rem;'>⬆️ Glissez-déposez votre image ici</h3>
    <p style='color: #ffe8d6; font-size: 1.1rem;'>Formats supportés: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

# File uploader avec gestion de cache
uploaded_img = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    key="image_uploader",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🖼️ AFFICHAGE DES RÉSULTATS
# ---------------------------------------
if uploaded_img is not None:
    try:
        # DEBUG: Afficher des informations sur le fichier
        st.write(f"📄 Fichier: {uploaded_img.name}")
        st.write(f"📏 Taille: {uploaded_img.size} bytes")
        st.write(f"🎨 Type: {uploaded_img.type}")
        
        # Lecture directe des bytes
        image_bytes = uploaded_img.getvalue()
        
        if len(image_bytes) == 0:
            st.error("❌ Le fichier est vide")
        else:
            # Ouvrir l'image depuis les bytes
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")  # Conversion en RGB
            
            st.success("✅ Image chargée avec succès!")
            
            if ULTRALYTICS_AVAILABLE and model is not None:
                # Layout principal pour images
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown("### 🖼️ Image Originale")
                    st.image(image, caption="Image source uploadée", use_container_width=True)
                    
                    # Informations sur l'image
                    st.markdown(f"""
                    **📏 Dimensions:** {image.size[0]} × {image.size[1]} pixels
                    **💾 Taille:** {len(image_bytes) / 1024:.1f} KB
                    **✅ Statut:** Prête pour l'analyse
                    """)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                # Bouton d'analyse centré
                st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
                analyze = st.button(
                    "🚀 Lancer l'Analyse IA", 
                    type="primary", 
                    use_container_width=True,
                    key="analyze_button"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                if analyze:
                    with st.spinner("🔍 **Analyse en cours...** L'IA scanne l'image"):
                        try:
                            # Conversion en numpy array
                            img_array = np.array(image)
                            
                            # Vérification des dimensions
                            if img_array.size == 0:
                                st.error("❌ L'image est vide")
                            else:
                                # Prédiction
                                results = model.predict(img_array, conf=0.25, imgsz=640)

                                if results and len(results) > 0:
                                    r = results[0]

                                    # Affichage résultats dans colonne 2
                                    with col2:
                                        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                                        st.markdown("### 📊 Résultats de Détection")
                                        
                                        if CV2_AVAILABLE:
                                            try:
                                                # Annotation avec OpenCV
                                                annotated = r.plot()
                                                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                                st.image(annotated_rgb, caption="🟠 Détections YOLOv8", use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"⚠️ Annotation OpenCV non disponible: {e}")
                                                st.image(image, caption="Image originale", use_container_width=True)
                                        else:
                                            st.image(image, caption="Image originale (OpenCV non disponible)", use_container_width=True)
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)

                                    # Statistiques de détection
                                    dets = getattr(r, "boxes", None)
                                    if dets and len(dets) > 0:
                                        st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
                                        st.markdown(f"""
                                        <div class="stat-item">
                                            <span class="stat-number">{len(dets)}</span>
                                            <span class="stat-label">Poubelles Détectées</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-number">{len(dets)}</span>
                                            <span class="stat-label">Analyses Effectuées</span>
                                        </div>
                                        <div class="stat-item">
                                            <span class="stat-number">YOLOv8</span>
                                            <span class="stat-label">Modèle IA</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.markdown("</div>", unsafe_allow_html=True)

                                        # Détails des détections
                                        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                                        st.markdown("### 🔍 Détails des Analyses")
                                        
                                        for i, box in enumerate(dets, start=1):
                                            cls_idx = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            cls_name = model.names[cls_idx] if hasattr(model, "names") else str(cls_idx)
                                            
                                            # Affichage avec barre de confiance
                                            conf_percent = int(conf * 100)
                                            st.markdown(f"""
                                            <div class="confidence-bar-container">
                                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                                    <span class="detection-badge">🔍 Détection #{i} • {cls_name.upper()}</span>
                                                    <strong style="font-size: 1.3rem; color: #fffaf0;">{conf_percent}%</strong>
                                                </div>
                                                <div class="confidence-bar" style="width: {conf_percent}%;"></div>
                                                <div style="text-align: center; color: #ffe8d6; font-size: 0.9rem; margin-top: 5px;">
                                                    Niveau de confiance de l'IA
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    else:
                                        st.warning("❌ Aucune poubelle détectée dans l'image")
                                else:
                                    st.error("❌ Aucun résultat d'analyse obtenu")

                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse: {str(e)}")

            elif not ULTRALYTICS_AVAILABLE or model is None:
                st.error("❌ Modèle non disponible - Impossible d'analyser l'image")
                
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'image: {str(e)}")
        st.info("💡 **Conseil de dépannage:** Essayez de redémarrer l'application avec `Ctrl+C` puis relancez-la")

else:
    # Section d'instructions quand aucune image n'est uploadée
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    
