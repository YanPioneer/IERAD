# EIRAD
Code repository for the paper : EIRAD: An evidence-based dialogue system with highly interpretable reasoning path for automatic diagnosis

# train
run the train.py

# test
run the predict.py

# requirements
torch==1.8.0 tensorboardX==2.5.1 numpy==1.20.1

# Dialogue System for Medical Diagnosis (DSMD)

Typically, DSMD simulates the real doctor's consultation process. During the consultation, the doctor first determines a set of suspected diseases based on the patient's chief complaints and selects symptoms associated with these diseases for further inquiry. They then incorporate medical knowledge and diagnostic experience to gather critical diagnostic evidence and confirm/exclude relevant diseases (updating the set of suspected diseases) according to patient's responses until the final preliminary diagnosis is given [9], [10].

![img](https://github.com/YanPioneer/IERAD/blob/main/Fig_Process_DSMD%202(1)_00.jpg)

Fig. 1 DSMD involves multi-turn conversations where doctors inquire further based on patient complaints and ultimately provide a preliminary diagnosis.

# Decision-making module

To imitate doctors' evidence-based reasoning and generate interpretable diagnostic paths from the patient's chief complaint to the final diagnosis, EIRAD decouples symptom inquiry and disease diagnosis.} At the $t$-th dialogue turn, the disease diagnosis module $M_\varphi$ first infers the probabilities of suspected diseases $D_t = M_\varphi(s_{1:t})$ based on the patient symptom information $s_{1:t}$, where $D_t \in \mathbb{R}^L$. If the highest probability of potential diseases exceeds the threshold $\epsilon$ or $t$ reaches the maximum turn $T_{max}$, the module returns a diagnosis result and concludes the current dialogue. Otherwise, the model will predict the most crucial symptom for the next inquiry based on MKG.

### Disease diagnosis module

Considering that label attention can effectively model the correlation between known symptoms and predicted disease labels using a label attention mechanism, we employed label attention for disease diagnosis. Additionally, taking into account the timeliness and deployment cost in practical applications, we initially utilized word2vec for encoding symptom information. However, during model analysis, we realized that the performance of the diagnostic model might affect the overall performance of the model. Therefore, we briefly explored the effects of using different disease diagnosis methods without being constrained by parameters or time, and employed BERT, RoBERT, and ERNIE for encoding, along with neural network linear layers (LLN) for implementation.

**The comparison of parameter quantity and inference speed for different disease diagnosis methods is as follows**: 

Table 1 The comparison of parameter quantity and inference speed for different disease diagnosis methods.

| Methods                                 | Parameter Quantity(MB) | Inference Speed(second) |
| --------------------------------------- | ---------------------- | ----------------------- |
| Label Attention (encoded with Word2Vec) | ***\*0.02\****         | ***\*0.001531\****      |
| BERT with NNL                           | 102                    | 0.01458                 |
| ROBERT with NNL                         | 102                    | 0.01435                 |
| ERNIE with NNL                          | 118                    | 0.01407                 |



# Datasets

To evaluate the overall performance of the proposed model framework, we test our system on three public medical dialogue datasets, MZ , DXY, and GMD. These datasets are derived from real medical diagnostic dialogues, the languages involved, types of diseases and symptoms, and the number of dialogue samples as depicted in Table I in the paper. Same as Liu et al., 80\% for training and 20\% for testing in MZ and DXY; 80\% for training, 10\% for validating, and 10\% for testing in GMD. The distribution of diseases across the datasets is fairly uniform.
Table 1: Medical Dialogue Datasets. "# Avg. Symptoms/Patient" signifies the average number of symptoms per patient in the dataset.

| Dataset                 | MZ                                                           | DXY                                                          | GMD                                                          |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Language                | Chinese                                                      | Chinese                                                      | Chinese/English                                              |
| Disease Type            | upper respiratory infection(130/30),pediatric bronchitis(166/34),pediatric diarrhea(155/45),pediatric indigestion(117/33) | allergic rhinitis(82/20),upper respiratory infection(97/23),pneumonia(79/20),Hand Foot and Mouth Disease(81/20),pediatric diarrhea(82/20) | esophagitis(157/27/16),enteritis(156/24/20),asthma(158/19/23),coronary heart disease(163/19/18),pneumonia(153/20/27),rhinitis(164/15/21),thyroiditis(154/19/27),traumatic brain injury(161/19/20),dermatitis(160/20/20),otitis externa(168/17/15),conjunctivitis(152/21/17),mastitis(166/19/15) |
| #Diseases               | 4                                                            | 5                                                            | 12                                                           |
| #Symptoms               | 66                                                           | 41                                                           | 118                                                          |
| #Dialogue Samples       | 710                                                          | 524                                                          | 2390                                                         |
| # Avg. Symptoms/Patient | 5.61                                                         | 4.77                                                         | 5.47                                                         |



# Evaluation metrics

Our evaluation metrics include disease diagnosis accuracy (Acc.), Macro-Precision/Recall/F1 (Macro-P/R/F1) for disease diagnosis (Multi-classification standards Micro and Macro can be used to measure disease diagnosis prediction, as Micro is consistent with the disease diagnosis accuracy, we only report Macro.), symptom inquiry efficiency (Eff.), symptom recall rate (Rec.), and average dialogue turns (Len.). The disease diagnosis accuracy is the probability of the diseases that the model diagnoses correctly, serving as a crucial indicator of the dialogue system's utility; Macro-P assesses the model's diagnostic accuracy for diseases, Macro-R ensures minimal misdiagnosis of cases, and Macro-F1 comprehensively evaluates the model's performance in diagnosing different categories of diseases; the symptom inquiry efficiency is the average proportion of the patient's symptoms successfully hit by the model to the total symptoms inquired, and measures the ability of the model to obtain diagnostic evidence, depicted in (15); symptom recall rate refers to the average proportion of the patient's symptoms successfully queried by the dialogue agent, calculated as (16); average dialogue turns is the average length of all conversational interactions at the end of the dialogue.

$$
Accuracy=\frac{TP_1+TP_2+...+TP_L}{TP_1+FP_1+TP_2+FP_2+...+TP_L+FP_L}\ \ \   \  \  \     \
$$

$$
{Macro-P}=\frac{1}{L}\sum_{i=1}^{L} \frac{TP_i}{TP_i+FP_i}\ \ \   \  \  \     \ 
$$

$$
{Macro-R}=\frac{1}{L}\sum_{i=1}^{L} \frac{TP_i}{TP_i+FN_i}\ \ \   \  \  \     \
$$

$$
{Macro-F1}=\frac{2 \times {Macro-P}\times {Macro-R}}{{Macro-P}+{Macro-R}}\ \ \   \  \ 
$$

where $L$ represents the total number of disease categories in the dataset, $TP_i$ denotes the count of samples correctly classified within category $i$, $FP_i$ indicates the count of samples from other categories erroneously classified as $i$, and $FN_i$ indicates the count of samples belonging to category $i$ but incorrectly classified into other categories.

$$
Efficiency= \frac{1}{K} \sum_{i=1}^{K} \frac{Sym\_Num_i^*}{Sym\_Num_i}\ \ \   \  \  \     \ (15)
$$

$$
Recall= \frac{1}{K} \sum_{i=1}^{K} \frac{Sym\_Num_i^*}{|s_i^p|}\ \ \   \  \  \     \      (16)
$$

where $K$ represents the total number of user goals, i.e., the number of dialogue samples. $Sym\_Num_i$ signifies the count of symptoms queried by the model during each dialogue, while $Sym\_Num_i^*$ represents the count of symptoms queried by the model and confirmed by the patient (either clear denial or acknowledgment), $|s_i^p|$ represents the number of potential symptoms of the $i$-th patient, that is, the number of implicit symptoms of the $i$-th user goal in the dataset.

$$
Length=\frac{1}{K} \sum_{i=1}^{K}len_i\ \ \   \  \  
$$

where $len_i$ represents the communication rounds between the agent and the patient throughout the dialogue process, from the patient's self-report to the agent delivers the preliminary diagnosis for the $i$-th user goal.

# Analyzing interpretability from a statistical perspective

In section 4.5.1, we demonstrat and analyze EIRAD's generation process of diagnostic paths through a case study. Compared to other methods, the symptoms inquired by EIRAD are highly related to the suspected diseases both structurally and semantically, and the calculation process is relatively transparent. To further illustrate the reasonableness of each node in EIRAD's diagnostic paths, we conduct a statistical analysis of symptom inquiries performed for all test user goals in DXY, as shown in Fig. 9. Fig. 9-a depicts a medical knowledge subgraph corresponding to DXY, where the values on edges represent the co-occurrence frequencies of diseases and symptoms in the dataset, reflecting the criticality of symptoms for diseases from a statistical perspective. Fig. 9-b and Fig. 9-c illustrate the statistical illustrations of symptom inquiries during the consultation process by the EIRAD and KRDS, respectively. The size of symptom nodes is positively correlated with inquiry frequency, and the color and width of edges are positively related to the number of times the symptoms are asked during disease diagnosis. It can be seen from the Fig. 9 that the symptoms queried by our proposed model during the consultation process substantially overlap with the key symptoms highly relevant to disease diagnosis, as reflected in the knowledge graph (frequently co-occurring and helpful for differential diagnosis), further demonstrating the rationality of the symptom inquiries and diagnostic paths of EIRAD.

![sym_statistic_com1](https://github.com/YanPioneer/IERAD/blob/main/sym_statistic_com1.svg)

Fig. 9. The statistical analysis results of symptom inquiries and disease diagnosis.

At the same time, the figure illustrates the diagnostic outcomes for various diseases across the dataset, indicating that our model has the lowest misdiagnosis rate. This in turn suggests that the symptoms EIRAD inquires about are all beneficial for disease diagnosis.
