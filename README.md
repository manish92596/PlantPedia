![Screenshot (5412)](https://github.com/user-attachments/assets/3bb69ed3-e542-4d03-a7ae-5427aa8565ea)


</br>

![Screenshot (5413)](https://github.com/user-attachments/assets/5913e09b-326d-4fd0-af40-ffa38a9ec25b)



</br>

![Screenshot (5414)](https://github.com/user-attachments/assets/e7598977-4ba0-4fe0-89e3-6a9b676bb648)





# 🌿 PlantPedia: Plant Leaf Classification and Information Retrieval 

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [Results](#-results)
- [Future Work](#-future-work)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🌟 Introduction
PlantPedia is an AI-powered system designed for classifying plant species using **Vision Transformers (ViTs)** combined with **transfer learning**. The project aims to automate plant species identification by analyzing leaf images, significantly reducing manual efforts, and enhancing ecological and agricultural research. The system includes an AI chatbot powered by **Google Generative AI (Gemini Pro)** for interactive information retrieval on identified plants.

## 🚀 Features
* **Automated Plant Leaf Classification** using Vision Transformers
* **Real-time chatbot** for detailed plant information powered by Google Gen-AI
* **Interactive GUI** developed using **Streamlit**
* **High accuracy of 95.97%** with precision, recall, and F1-score each at 0.96

## 🌿 Dataset
* **Swedish Leaf Dataset**: 1,125 images of 15 plant species
* **Extended Dataset**: Added 10 additional plant species, bringing the total to **1,875 images across 25 classes**

## 🔧 Methodology
1. **Image Preprocessing**:
   * Resizing to `224x224` pixels
   * Normalization using ImageNet statistics
   * Data augmentation (rotations, flips, shifts) for better generalization

2. **Feature Extraction**:
   * Leveraged pre-trained Vision Transformers to extract high-level features
   * Added a custom classifier: `Linear layer (512 units) → ReLU → Dropout (0.5) → Output layer`

3. **Transfer Learning & Fine-Tuning**:
   * Freezed initial ViT layers and fine-tuned the final layers
   * Used an **Adam optimizer** with learning rate scheduling and early stopping

4. **Model Testing**:
   * Evaluated on the test dataset using metrics like accuracy, precision, recall, F1-score, and a confusion matrix

## 🛠️ Installation
1. **Clone the Repository**:
```bash
git clone https://github.com/manish92596/PlantPedia.git
cd PlantPedia
```

2. **Set Up a Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## 🖥️ Usage
1. **Run the Streamlit Application**:
```bash
streamlit run app.py
```

2. **Upload a Plant Leaf Image**:
   * Get species classification and interact with the chatbot for more information

## 📊 Results
* **Model Accuracy**: 95.97%
* **Precision**: 0.96
* **Recall**: 0.96
* **F1 Score**: 0.96
* **Confusion Matrix**: Visualizes model performance across all 25 classes

## 🔮 Future Work
* **Multilingual Support**: Extend the chatbot's capabilities to support multiple languages
* **Disease Detection**: Add functionality to detect plant diseases from leaf images
* **Scalability**: Expand the dataset to include more plant species for broader classification

## 💻 Technologies Used
* **Framework**: PyTorch with Hugging Face Transformers
* **Python Version**: 3.11.5
* **Libraries**: 
  - torch
  - torchvision
  - transformers
  - scikit-learn
  - matplotlib
  - seaborn
* **Generative AI**: 
  - langchain-google-genai
  - langchain[openai,all]
  - langchain-community
  - sentence-transformers
* **GUI**: Streamlit
* **Hardware**: NVIDIA A100 GPUs with CUDA support

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-branch`
5. Open a pull request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact
Feel free to reach out for collaboration or queries:
* **Manish Kumar** - [GitHub](https://github.com/manish92596) | [Email](mailto:manishkumar92596@gmail.com)

---
Enjoy using PlantPedia! 🌱🌿🌳