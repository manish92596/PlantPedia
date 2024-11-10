
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class PlantClassifier:
    def __init__(self, model_path, class_labels_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.class_labels = self._load_class_labels(class_labels_path)
        self.transform = self._setup_transform()

    def _load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_class_labels(self, file_path):
        try:
            with open(file_path, "r") as file:
                return [line.strip() for line in file.readlines()]
        except Exception as e:
            raise RuntimeError(f"Failed to load class labels: {str(e)}")

    def _setup_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        try:
            image = Image.open(image).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)  
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            return self.class_labels[predicted.item()], confidence.item() * 100
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

class PlantInfoRetriever:
    def __init__(self, api_key):
        self.chat_model = ChatGoogleGenerativeAI(
            api_key=api_key, 
            model="gemini-pro", 
            temperature=0.3
        )
        self.prompt_template = PromptTemplate(
            template="""
            Please answer the given question using the context provided. Make the response short and concise, about 150 words. 
            You may enhance the response using relevant knowledge only if it aligns with the context.\n\n
            Question: \n{question}\n
            Answer:
            """,
            input_variables=["question"]
        )

    def get_information(self, query):
        try:
            formatted_query = self.prompt_template.format(question=query)
            response = self.chat_model.invoke(formatted_query)
            return response.content
        except Exception as e:
            return f"An error occurred while fetching information: {str(e)}"


def main():
    st.set_page_config(page_title="PlantPedia 🌿", layout="wide")
    
    if not GEMINI_API_KEY:
        st.error("Gemini API Key not found. Please check your environment variables.")
        return

   
    if 'image_uploaded' not in st.session_state:
        st.session_state.image_uploaded = False
    if 'requests' not in st.session_state:
        st.session_state.requests = []
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'predicted_label' not in st.session_state:
        st.session_state.predicted_label = None
    if 'initial_question_asked' not in st.session_state:
        st.session_state.initial_question_asked = False
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

    
    try:
        classifier = PlantClassifier(
            model_path='/fab3/btech/2021/manish.kumar21b/manish/PlantPedia/endsem/model/trained_model.pkl',
            class_labels_path="/fab3/btech/2021/manish.kumar21b/manish/PlantPedia/endsem/class_labels.txt"
        )
        info_retriever = PlantInfoRetriever(GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        return

    st.title("PlantPedia 🌿🧠")
    
   
    if st.sidebar.button("New Chat"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

   
    st.subheader("Upload an image of a plant leaf and click 'Predict' to learn more about it.")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image and st.button("Predict"):
        try:
            st.session_state.uploaded_image = uploaded_image
            st.session_state.image_uploaded = True

            with st.spinner("Predicting..."):
                predicted_label, confidence_score = classifier.predict(uploaded_image)
                st.session_state.predicted_label = predicted_label
                st.session_state.confidence_score = confidence_score

                if not st.session_state.initial_question_asked:
                    initial_question = f"What is a {predicted_label} plant leaf and can you provide some interesting facts about it?"
                    response = info_retriever.get_information(initial_question)
                    st.session_state.requests.append(initial_question)
                    st.session_state.responses.append(response)
                    st.session_state.initial_question_asked = True

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            predicted_label = "Prediction failed"
            confidence_score = 0.0

    
    if st.session_state.image_uploaded and st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.session_state.predicted_label:
            st.success(f"Predicted Label: {st.session_state.predicted_label} (Confidence: {st.session_state.confidence_score:.2f}%)")



  
    if st.session_state.requests:
        for request, response in zip(st.session_state.requests, st.session_state.responses):
            with st.container():
                st.markdown(f"**🙋 You:** {request}")
                st.markdown(f"**🌿 Chatbot:** {response}")
            st.markdown("---")

    
    if st.session_state.predicted_label:
        with st.form(key='question_form', clear_on_submit=True):
            user_question = st.text_input("Ask a question about the plant leaf:")
            if st.form_submit_button("Submit") and user_question:
                with st.spinner("Generating answer..."):
                    response = info_retriever.get_information(user_question)
                    st.session_state.requests.append(user_question)
                    st.session_state.responses.append(response)
                st.rerun()  

if __name__ == "__main__":
    main()