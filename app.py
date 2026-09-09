import streamlit as st
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


# -----------------------------
# Page
# -----------------------------
st.title("Face Recognition")


# -----------------------------
# Load saved embeddings
# -----------------------------
try:
    with open("HLJ.pkl", "rb") as f:
        database = pickle.load(f)
except FileNotFoundError:
    st.error("embedding.pkl not found.")
    st.stop()


# -----------------------------
# Load InsightFace
# -----------------------------
@st.cache_resource
def load_model():

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return app


app = load_model()


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------
# Recognition
# -----------------------------
if uploaded_file is not None:

    # Convert uploaded image to NumPy array
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Detect faces
    faces = app.get(img)

    # -----------------------------
    # Check number of faces
    # -----------------------------
    if len(faces) == 0:

        st.error("No face detected.")

    elif len(faces) > 1:

        st.warning(
            "More than one face detected. "
            "Please upload an image with one face."
        )

    else:

        # -----------------------------
        # Get face embedding
        # -----------------------------
        embedding = faces[0].embedding

        embedding = np.asarray(
            embedding
        ).reshape(1, -1)


        # -----------------------------
        # Compare with database
        # -----------------------------
        max_score = -1
        max_name = None

        for name, stored_embedding in database.items():

            stored_embedding = np.asarray(
                stored_embedding
            ).reshape(1, -1)

            score = cosine_similarity(
                stored_embedding,
                embedding
            )[0][0]

            if score > max_score:

                max_score = score
                max_name = name


        # -----------------------------
        # Recognition result
        # -----------------------------
        threshold = 0.55
        def fun(name):
            result = ""
            for char in name:
                if not char.isdigit():
                    result += char
                
        
           
            return result

        if max_score >= threshold:
            clean_name = fun(max_name)
            st.success(
                f"Person: {clean_name}"
            )

            st.write(
                f"Similarity: {max_score:.3f}"
            )

        else:

            st.warning(
                "Unknown person"
            )

            st.write(
                f"Best similarity: {max_score:.3f}"
            )
