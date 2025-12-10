import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Configuration de la page
st.set_page_config(
    page_title="🌐 CLASSICATION DU CANCER DU SEIN ",
    page_icon="🌐",
    layout="centered",
)

# fond bleu professionnel
st.markdown("""
<style>
    /* Fond bleu professionnel */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #90caf9 100%);
        background-attachment: fixed;
    }
    
    /* Style pour les cartes */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15);
        border: 2px solid #2196f3;
    }
    
    /* Style pour les titres */
    .custom-title {
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Style pour les sous-titres */
    .custom-subtitle {
        color: #0d47a1;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Uploader stylisé */
    .stFileUploader > div > div {
        border: 3px dashed #1565c0;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #0d47a1;
        background: rgba(255, 255, 255, 1);
    }
    
    /* Style pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e3f2fd;
        border-radius: 10px 10px 0px 0px;
        gap: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2196f3 !important;
        color: white !important;
    }
    
    /* Footer personnalisé */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: white;
        padding: 10px 0;
        text-align: center;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .footer-content {
        display: flex;
        justify-content: space-around;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    
    .footer-icon {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Ajout du footer avec vos coordonnées
st.markdown("""
<div class="custom-footer">
    <div class="footer-content">
        <div class="footer-item">
            <span class="footer-icon">📞</span>
            <span><strong>Téléphone :</strong> +237 659 06 06 81</span>
        </div>
        <div class="footer-item">
            <span class="footer-icon">📧</span>
            <span><strong>Email :</strong> louiskngn01@gmail.com</span>
        </div>
        <div class="footer-item">
            <span class="footer-icon">©</span>
            <span>Yaoundé-Cameroun:2025</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

#Entête 
st.markdown("""
    <div class="custom-title">🌐 CLASSIFICATION DU CANCER DU SEIN</div>
    <div class="custom-subtitle"> Sein normal & Sein malin</div>
""", unsafe_allow_html=True)

#CHARGER LE MODÈLE 
MODEL_PATH = "/models/mon_CNN_final.h5"
model = tf.keras.models.load_model(MODEL_PATH)

#CLASSES 
class_names = ["[0]=malignant", "[1]=normal"]

#PRÉTRAITEMENT 
def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  
        image = image[..., :3]
    image = np.expand_dims(image, 0)
    return image

#PREDICTION 
def predict_image(image):
    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)[0]

    if pred.shape == () or len(pred) == 1:
        prob1 = float(pred)
        prob0 = 1 - prob1
        probs = [prob0, prob1]
    else:
        probs = pred.tolist()

    predicted_index = int(np.argmax(probs))
    predicted_class = class_names[predicted_index]
    confidence = probs[predicted_index] * 100

    return predicted_class, confidence, probs, predicted_index

# Fonction pour capturer l'image de la caméra
def capture_camera_image():
    """Capture une image depuis la caméra"""
    camera_image = st.camera_input("📸 Prendre une photo avec la caméra")
    if camera_image is not None:
        return Image.open(camera_image)
    return None

# Fonction principale pour afficher les résultats
def display_results(image):
    """Affiche les résultats de l'analyse"""
    #Section image
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.image(image, caption="✅Image analysée✅", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #Analyse
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.write(" **ANALYSE EN COURS...**")
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
    st.write("✅ Analyse terminée !")
    st.markdown('</div>', unsafe_allow_html=True)

    #PREDICTION
    predicted_class, confidence, probs, predicted_index = predict_image(image)

    #RÉSULTAT
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    if predicted_index == 0:  # malignant
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                     border-radius: 10px; padding: 20px; border-left: 6px solid #f44336;">
                <h2 style="color:#d32f2f;">⚠️ RÉSULTAT : {predicted_class}</h2>
                <h3 style="color:#b71c1c;">Confiance : {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    else:  # normal
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                     border-radius: 10px; padding: 20px; border-left: 6px solid #4caf50;">
                <h2 style="color:#2e7d32;">✅ RÉSULTAT : {predicted_class}</h2>
                <h3 style="color:#1b5e20;">Confiance : {confidence:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #PROBABILITÉS
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔻**PROBABILITÉS DÉTAILLÉES**🔻")
    
    for i, cls in enumerate(class_names):
        if i < len(probs):
            bar_color = "#f44336" if i == 0 else "#4caf50"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{cls}**")
            with col2:
                st.write(f"**{probs[i]*100:.2f}%**")
            
            st.markdown(f"""
                <div style="background: #f5f5f5; border-radius: 10px; height: 25px; margin: 5px 0;">
                    <div style="background: {bar_color}; width: {probs[i]*100}%; 
                             height: 100%; border-radius: 10px;"></div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #CONCLUSION
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    if predicted_index == 0:
        st.markdown("""
            <div style="text-align: center; padding: 20px; background: #ffebee; 
                     border-radius: 10px; border: 2px solid #f44336;">
                <h2 style="color:#d32f2f;">⚠️⚠️ CONCLUSION</h2>
                <h3 style="color:#b71c1c;">Votre sein est Cancereux.</h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 20px; background: #e8f5e9; 
                     border-radius: 10px; border: 2px solid #4caf50;">
                <h2 style="color:#2e7d32;">✅ CONCLUSION</h2>
                <h3 style="color:#1b5e20;">Votre sein est en forme normal.</h3>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    #MESSAGE FINAL
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    if predicted_index == 0:
        st.markdown("""
            <div style="padding: 15px; background: #fff3e0; border-radius: 10px;">
                <p style="color:#e65100;">
                    ⚠️⚠️ <b>IMPORTANT :</b> Consultez un professionnel de santé rapidement.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="padding: 15px; background: #f1f8e9; border-radius: 10px;">
                <p style="color:#33691e;">
                    ✅ <b>INFORMATION :</b> Continuez vos examens de routine.
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== NOUVELLE SECTION : ONGLETS POUR CHOIX DE MÉTHODE ==========

# Créer des onglets pour choisir la méthode d'importation
tab1, tab2 = st.tabs(["📁 **Importer une image**", "📸 **Prendre une photo**"])

# Onglet 1 : Importer une image
with tab1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📁 **IMPORTER UNE IMAGE DEPUIS VOTRE APPAREIL**")
    uploaded_file = st.file_uploader("**Sélectionnez une image**", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Format accepté : JPG, JPEG, PNG")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        display_results(image)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <h3 style="color: #1565c0;">📁 Sélectionnez une image depuis votre appareil</h3>
                <p style="color: #666;">Format accepté : JPG, JPEG, PNG</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Onglet 2 : Prendre une photo
with tab2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📸 **PRENDRE UNE PHOTO AVEC VOTRE CAMÉRA**")
    st.markdown("""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <p style="color: #1565c0;">
                <b>Instructions :</b><br>
                1. Autorisez l'accès à votre caméra<br>
                2. Placez-vous face à la caméra<br>
                3. Cliquez sur le bouton pour prendre la photo<br>
                4. L'analyse commencera automatiquement
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Bouton pour activer la caméra
    use_camera = st.checkbox("✅ Activer la caméra", value=False)
    
    if use_camera:
        st.info("🎥 La caméra est activée. Prenez une photo ci-dessous.")
        camera_image = st.camera_input("**Prendre une photo**", 
                                      key="camera_capture",
                                      help="Cliquez sur le bouton de l'appareil photo pour capturer l'image")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            display_results(image)
        else:
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #1565c0;">📸 Préparez-vous à prendre une photo</h3>
                    <p style="color: #666;">Positionnez-vous face à la caméra et cliquez sur le bouton de capture</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <h3 style="color: #1565c0;">📸 Activez la caméra pour prendre une photo</h3>
                <p style="color: #666;">Cochez la case ci-dessus pour activer votre caméra</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Message d'information générale
st.markdown('<div class="custom-card">', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        
    </div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
