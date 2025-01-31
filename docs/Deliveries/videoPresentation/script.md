Here is your translated and structured script in English, keeping within the specified time constraints:  

---

### **Introduction and Context (30-45 seconds)**  

**Opening:**  
*"Hello everyone. My dissertation addresses a critical healthcare issue: lung cancer, which remains the leading cause of cancer-related deaths worldwide. Late diagnosis results in a 5-year survival rate of less than 10% in 70% of cases, on contrary early detection can increase this rate to over 90%."*  

**Context:**  
*"In 2022, lung cancer had the highest incidence and mortality of all cancers. Efforts to reduce mortality through screening have been hindered by the aggressive and diverse nature of the disease. Currently, lung nodule classification relies on assessing growth rate over two years, avoiding biopsies, which carry risks and costs."*  

**Need:**  
*"There is an increasing need for models that can effectively integrate and analyze complex medical data to support clinical decision-making.
CAD systems have emerged as a promising alternative."*  

---

### **Problem Statement (20-30 seconds)**  

**Problem:**  
*"This thesis addresses the need for a more accurate and reliable diagnostic tool. Existing diagnostic systems, based on deep learning models, face limitations in terms of accuracy and generalization across different medical imaging datasets. These models tend to emphasize deep features while neglecting important surface-level characteristics such as texture and shape."*  

**Challenges:**  
*"Additionally, the lack of interpretability in these models poses a challenge in clinical contexts, limiting their reliability in critical medical decision-making."*  

---

### **Research Questions (20-30 seconds)**  

**Questions:**  
*"My research aims to answer three key questions:"*  

1. *Does fusing information from shallow and deep feature extractors bring any improvement (classification performance, generalization, reduction in the number of model parameters) compared to using only a deep feature extractor?*  
2. *How does this approach compare with an approach that only uses a deep feature extractor when varying the dataset? (e.g. training on one set of data and testing on another, using different amounts of data, among others)*  
3. *In what ways can information-fusion-based models contribute to improving the explainability of lung nodule malignancy predictions?*  

---

### **Literature Review and Theoretical Framework (60-75 seconds)**  

**Review Objectives:**  
*"The literature review aims to define a strategy for analyzing the state of research on information fusion models for lung nodule characterization. The goal is to examine techniques used for nodule characterization, methods for automating this process, and assess their effectiveness."*  

**Research Strategy:**  
*"The search was conducted in databases such as IEEE Xplore, PubMed, and Google Scholar using keywords like 'lung nodule characterization,' 'information fusion,' 'shallow feature extraction,' and 'deep feature extraction.' The reference chaining technique was also applied."*  

**Eligibility Criteria:**  
*"Studies published in the last five years (2019-2024), in English, focusing on lung nodule characterization or other medical conditions revealed through CT scans, were included, provided they used relevant methodologies for the study."*  

---

### **State of the Art and Technological/Methodological Advances (30-45 seconds)**  

**Advances:**  
*"Recent advancements in deep learning and information fusion techniques have significantly improved lung nodule diagnosis. Convolutional Neural Networks (CNNs) have become essential for automatically extracting high-dimensional features from medical images, reducing the reliance on manual feature engineering. Additionally, radiomics, which quantifies imaging data, has gained traction, as its integration with deep learning models enhances classification performance."*  

**Emerging Techniques:**  
*"Information fusion techniques are increasingly used to combine shallow features (such as texture and shape) with deep features. This can be done at the feature level (feature-level fusion) or decision level (decision-level fusion), with decision-level fusion proving more effective in integrating CNN-derived features with handcrafted radiomics features. Transfer learning, where pre-trained models are fine-tuned for lung nodule classification, is another widely used approach to improve accuracy and efficiency, particularly given the limited size of medical imaging datasets."*  

**Impact of These Innovations:**  
*"By integrating these techniques, recent models have achieved sensitivities above 90% while reducing false positives, leading to more accurate and clinically reliable lung nodule classification. My research builds upon these advancements, further exploring fusion-based models to enhance accuracy, interpretability, and generalization, ultimately contributing to early lung cancer detection and improved public health outcomes."*  


---


### **Methodology and Experimentation (60-75 seconds)**  

**Population and Sampling:**  
*"This study utilizes publicly available and clinically validated CT imaging datasets, such as LIDC-IDRI and LUNA16, ensuring reliable ground-truth labels for lung nodule classification. To assess data quality and balance, we analyze statistical metrics—including nodule size, patient demographics, and class distribution—informing potential sampling strategies. Preprocessing steps, such as resampling and noise reduction, are applied to enhance image uniformity and improve model performance. The dataset is then randomly split into training and testing sets to prevent overfitting and ensure robust model evaluation."*  

**Evaluation Strategy:**  
*"Each model configuration undergoes rigorous cross-validation to guarantee reliable and generalizable results. We employ multiple performance metrics to assess classification effectiveness: accuracy offers a general measure of correctness, but in imbalanced datasets, sensitivity (recall) is crucial for detecting malignant nodules and minimizing false negatives. Specificity ensures benign nodules are correctly classified, reducing unnecessary interventions. Additionally, the AUC-ROC metric evaluates the model’s overall discriminative power across different classification thresholds, balancing sensitivity and specificity."*  

**Deep Learning Selection:**  
*"The first step is identifying state-of-the-art deep learning models tailored for pulmonary nodule classification. These models will establish a baseline for performance comparison, after which the top-performing architectures will be selected for further enhancement through complementary methodologies."*  

**Shallow Extractors:**  
*"In parallel, shallow feature extractors are employed to capture critical texture, shape, gradient, and moment-based information. These features provide valuable complementary insights not always captured by deep learning models. The selected extractors, aligned with state-of-the-art methods, ensure compatibility with fusion techniques, allowing for an integrated approach to lung nodule characterization."*  

**Fusion Methods:**  
*"To leverage the strengths of both deep learning and shallow feature extraction, advanced fusion techniques will be implemented. These include decision-level fusion—such as weighted and majority voting—and feature-level fusion, where extracted feature vectors are concatenated or averaged, often using Principal Component Analysis (PCA) for dimensionality reduction. This modular fusion framework enables seamless integration across different model configurations."*  

**Experimentation:**  
*"The experimentation phase systematically evaluates various model combinations. The workflow begins by selecting one of the top three deep learning models, integrating it with one or more shallow feature extractors, and applying a chosen fusion method. The complete pipeline processes CT images, extracts features, applies fusion, and trains the classification model. Performance is assessed using metrics including accuracy, sensitivity, specificity, F1-score, AUC, and false positive rate (FPR). This structured approach ensures that the most effective combinations are identified, surpassing standalone deep learning models and traditional classifiers. The insights gained will contribute to advancing pulmonary nodule classification and improving early lung cancer detection."*  


---

### **Expected Contributions and Impact (30-45 seconds)**  

**Contributions:**  
*"This dissertation aims to contribute a more accurate and reliable model for lung nodule characterization. The expected outcome is improved diagnostic accuracy, better clinical decision support, and a reduction in the need for invasive procedures."*  

**Impact:**  
*"The anticipated impact is progress in early lung cancer diagnosis, potentially leading to reduced mortality and improved treatment strategies. This work aligns with Goal 3 of the United Nations’ Sustainable Development Goals, which aims to ensure healthy lives and promote well-being for all."*  

---

### **Conclusion (10-20 seconds)**  

**Relevance:**  
*"The accuracy of lung nodule diagnosis is crucial for public health. My research aims to contribute to solving this problem through an innovative and effective approach."*  

**Next Steps:**  
*"I am confident that the proposed approach will yield significant results and have a positive impact on healthcare."*  

