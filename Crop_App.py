import streamlit as st
import pickle
import sklearn
from joblib import dump, load
import pandas as pd
from PIL import Image

image = Image.open('Spring_Crop_PNG_Clipart.png')
image_resized = image.resize((200, 300))

#st.image(image_resized, caption='Image redimensionnée')




def pred(feat):
    feat=pd.DataFrame(feat)
    clf = pickle.load(open('/Users/borgou/Documents/COURS/MASTERE 1/SEMESTRE 2/PROJET_INTERDISCIPLINAIRE/Projet_interdisciplinaire/RandomForest.pkl','rb'))

    pred=clf.predict(feat)
    st.success('La culture recommandée est {}'.format(str(pred[0]).upper()))
    st.success('La précision de la prédiction est: {}%'.format(99.54))

def main():
    page_bg_img = """
    <style>div.stButton > button:first-child {
    background-color: #7aaf04;color:black;font-size:20px;height:3em;width:100%;border-radius:10px 10px 10px 10px;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html= True)

    html_temp = """ 
    <div style ="padding:13px;margin:15px 0px;background-color:#7aaf04  ;">
    <h1 style ="color:black;text-align:center;">Recommandation de culture à l'aide de ML</h1>
    </div>
    <h4 style ="text-align:center;padding:20px 0px">Entrez les détails de votre sol et nous vous recommanderons quelle est la culture la mieux adaptée à votre sol et vous aiderons à maximiser vos profits!</h4>
    """
    st.image(image_resized) # l'insertion de l'image
    st.markdown(html_temp, unsafe_allow_html = True)
    N=st.text_input("Rapport de teneur en azote dans le sol", "")
    P=st.text_input("Rapport de teneur en phosphore dans le sol", "")
    K=st.text_input("Rapport de teneur en potassium dans le sol", "")
    temp=st.text_input("Température en degré Celsius", "")
    hum=st.text_input("Humidité (relative en %)", "")
    ph=st.text_input("pH du sol", "")
    rain=st.text_input("Précipitations (en mm)", "")
    l=[[N, P, K, temp, hum, ph, rain]]
    col1,col2,col3=st.columns([0.3,1.2,0.3])

    with col1:

        st.empty()
    with col2:
        if st.button("Recommander"):
            pred(l)
    with col3:
        st.empty()


if __name__=='__main__':
    main()

