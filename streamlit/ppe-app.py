import streamlit as st
from PIL import Image

from backend import (
    load_simple_cnn, load_resnet50, load_yolo,
    predict_cnn, predict_resnet, predict_yolo
)

st.set_page_config(page_title="Safety Engineering Assistant", layout="wide")

# Loading models 
cnn_model = load_simple_cnn()
resnet_model = load_resnet50()
yolo_model = load_yolo()

# Main
def main():
    st.markdown(
    """
    <h1 style='color:#009774; font-weight:500;'>
        Safety Engineering Assistant 
    </h1>
    <h3 style='color:gray; font-weight:200; margin-top:-10px;'>
        AI system for Detecting PPE and Improving Workplace Safety
    </h3>
    """,
    unsafe_allow_html=True
    )   

    uploaded_image = st.file_uploader("**Upload your image 👷:**", type=["jpg", "jpeg"])
    model_selection = st.selectbox("**Select a model:**", options= ("Simple CNN", "ResNet50", "Yolov8"))

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")

        if st.button("Run model"):
            if model_selection == "Simple CNN":
                preds = predict_cnn(cnn_model, img)
                st.success(", ".join(preds))
                st.image(img)

            elif model_selection == "ResNet50":
                preds = predict_resnet(resnet_model, img)
                st.success(", ".join(preds))
                st.image(img)

            else:
                annotated = predict_yolo(yolo_model, img)
                st.image(annotated, caption="YOLOv8")

    with st.sidebar:
        st.image("ppe-logo.png")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("### Contact:") 
        st.markdown("[Raquel Colares](https://www.linkedin.com/in/raquel-colares)")

if __name__ == "__main__":
    st.sidebar.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #009774;
        }
    </style>
    """, unsafe_allow_html=True)
    main()
