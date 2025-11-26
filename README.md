# AI and Safety Engineering
---

Ensuring workplace safety is a critical priority across many environments, such as construction sites, industrial settings, and any other place where workers are exposed to potential hazards. Because of that, Personal Protective Equipment (PPE) acts as the first line of defense against injuries, making accurate and consistent monitoring essential for preventing accidents and promoting safer working conditions. With AI, more specifically in computer vision, it is now possible to detect and monitor PPE in a much more frequent and consistent way, reducing human error and helping safety teams.

### Objective

The objective of this project is to build an AI system capable of detecting Personal Protective Equipment (PPE) using three deep learning approaches. Two of these models are CNN multi-label classification designed to determine which PPE items are present in an image, one model a simple CNN architecture built from scratch and the other ResNet50 using transfer learning. The third model on the other hand is YOLOv8 focused on object detection and localization of each equipment component. By training, evaluating, and comparing these three models, the project aims to identify the most effective approach for practical and reliable PPE monitoring in real-world scenarios.

### Data 

The dataset used in this project is based on the Construction PPE dataset provided by Ultralytics, referenced at the end of this project. It includes labeled images for various safety equipment such as helmet, gloves, vest, boots, goggles, none, person, no_helmet, no_goggle, no_gloves, and no_boots. 


### Simple CNN 



### ResNet50



### Yolov8



### Project structure

.
├── images/                              # Image dataset
├── labels/                              # Label dataset
├── models/                              # Saved model's weights 
│   ├── model_3/                         # Yolov8 model files
│   ├── model_1_weights.pth              # Simple CNN model weights
│   └── model_2_weights.pth              # ResNet50 model weights
│
├── notebook/
│   └── AI_and_Safety_Engineering.ipynb  # Jupyter notebook containing all project
│
├── streamlit/
│   └── ppe-app.py                       # Streamlit interface
│
├── demo/                                # Demo video 
│
├── backend.py                           # Backend pipeline 
├── requirements.txt                     # Dependencies list
└── README.md                            # Project documentation




### Visualization
- **Project visualization:** https://ppe-safe-detection.streamlit.app/

The streamlit can be seen on the link above and also accessing by the following command line on the Anaconda prompt:

`streamlit run ppe-app.py`

Or on VS Code:

`py -m streamlit run ppe-app.py` 

### Demo




### References 