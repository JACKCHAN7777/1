
Waste classification detection and identification system based on YOLO V8 algorithm






A Dissertation Presented 
By





Submitted to The University of The West of England in partial fulfilment of the requirements for the degree of: 

Software Engineering for Business

April 2025
Department of Environment and Technology – Computer Science and Creative Technologies


ABSTRACT 

With the increasing importance of environmental protection and resource recycling, waste classification has become a global concern. Traditional waste classification methods suffer from inefficiency and misclassification, while deep learning-based target detection technology shows great promise in improving classification accuracy and efficiency. In this study, a waste classification system based on the YOLOv8 (You Only Look Once version 8) target detection algorithm is proposed, which is capable of accurately identifying and classifying 36 types of waste. In order to improve the generalisation ability and classification effect of the model, this paper obtains 3490 rubbish images through web crawlers, constructs a homemade dataset containing 36 categories of rubbish, and uses the LabelImg tool for data annotation. The YOLOv8 algorithm is used to train the dataset, and the training strategy of freezing and unfreezing phases is adopted, and the model training is carried out for a total of 300 Epochs, with an initial learning rate of 1e-2, and the learning rate is gradually adjusted by the learning rate decay strategy of cosine annealing. The experimental results show that the system is able to achieve high mean average precision (mAP) in different categories of waste classification tasks, demonstrating good classification performance and application prospects. This study further demonstrates the efficiency and accuracy of the YOLOv8 algorithm in multi-category target detection tasks such as waste classification.

Keywords: Waste classification, YOLOv8, target detection, deep learning, classification system
 

CONTENTS

CHAPTER 1 INTRODUCTION	1
1.1 Current Situation and Problems of Waste Classification	1
1.2 Research on Intelligent Waste Classification System Based on YOLOv8	1
CHAPTER 2 RELATED WORK	3
2.1 Current Status of Waste Classification Technology	3
2.2 Development of YOLO Target Detection Algorithm	4
2.3 Advantages of YOLOv8	6
CHAPTER 3 METHOD	8
3.1 Data Set Construction	8
3.2 YOLOv8 Algorithm	9
3.2.1 Network Architecture of YOLOv8	10
3.3 Model Training Setup	11
CHAPTER 4 RESEARCH	13
4.1 Secondary Research	13
4.1.1 Literature Review	13
4.1.2 Standards and Regulations	13
4.2 Informal Sources: User Stories & Case Studies	13
4.2.1 User Stories	13
4.2.2 Case Studies	14
4.3 Technology Selection: Pugh Matrix	14
4.4 Main Argument and Alternative Arguments	15
4.4.1 Main Argument	15
4.4.2 Alternative Arguments	15
CHAPTER 5 EXPERIMENTS AND RESULTS	16
5.1 Experimental Environment	16
5.2 Data Set	16
5.3 Model Training Process	16
5.3.1 Training Phase Loss Curves	17
5.3.2 Training Phase Map Curve	17
5.4 Model Performance Evaluation	18
5.4.3 F1 Chart	21
5.4.4 Precision and Recall	24
5.4.5 ground-truth	26
5.4.6 Log-average Miss Rate	27
5.4.7 Confusion Matrix	28
5.5 System Interface and User Experience	29
CHAPTER 6 CONCLUSION	31
6.1 Analysis of Model Strengths	31
6.1.1 High Accuracy and Efficiency	31
6.1.2 Real-time and Deployment Potential	31
6.2 Problems and Limitations	32
6.3 Directions for Future Improvement	32
REFERENCES	34







 

CHAPTER 1
INTRODUCTION

1.1 Current Situation and Problems of Waste Classification

With the growing global awareness of environmental protection, waste separation has received widespread attention and promotion as an important means of achieving resource recovery and reducing environmental pollution. By reasonably classifying rubbish, it can not only effectively reduce the difficulty of rubbish disposal but also maximise the reuse of resources. Modern waste sorters using intelligent tools have the ability to efficiently enhance the task load of manual sortation and diminish the rate of errors in sortation while enhancing the degree of automation, thereby offering efficient aid to environmental protection and recycle systems.
Instead, the current waste sorting depends largely on manual operation or simple machinery for sorting, which is not only inefficient but also susceptible to errors in sorting. Particularly when it comes to the vast range of waste, the traditional approach cannot guarantee the accuracy of the sorted and real-time. Thus, the key to overcoming such a problem lies in the development of an intelligent classification system of waste.

1.2 Research on Intelligent Waste Classification System Based on YOLOv8

In recent years, deep learning-based target detection technology has made significant progress in the field of image processing, and the YOLO (You Only Look Once) series of algorithms, as a kind of efficient real-time target detection model, has been widely used in various kinds of image recognition tasks due to its fast speed and high accuracy. The YOLO series of algorithms have been continuously updated and changed since they were proposed. The latest version, YOLOv8, has been significantly improved in terms of structure optimisation and detection accuracy. YOLOv8 not only inherits the advantages of its predecessor in terms of computing speed, but also improves the detection of small and multi-category targets by optimising the structure of the model and adding a more efficient loss function. This makes it a promising candidate for a wide range of applications in complex rubbish sorting tasks.
The aim of this study is to design and implement an intelligent waste classification system based on the YOLOv8 algorithm. In order to improve the classification ability of the system, we first obtained 3490 pictures of different types of waste through web crawlers, constructed a dedicated dataset containing 36 waste categories, and labelled the data using the LabelImg tool. Subsequently, we trained the model based on the YOLOv8 model using the training strategy of freezing and unfreezing phases, and finally completed the development of the waste classification system. By adjusting the learning rate, optimiser and other key parameters, the model demonstrated high classification accuracy and stability in the waste classification task.
The main contribution of this paper is that an efficient rubbish classification method is proposed based on YOLOv8, which is able to realise real-time classification and identification of multiple categories of rubbish. With this system, we not only improve the automation of rubbish classification, but also verify the superiority and practicality of the YOLOv8 algorithm in the rubbish classification task.













CHAPTER 2
RELATED WORK

2.1 Current Status of Waste Classification Technology

With the increasingly serious environmental problems, waste classification has become an important means to improve the efficiency of resource recovery. At present, technical solutions for waste classification are mainly divided into traditional methods and artificial intelligence-based methods.
Traditional waste sorting techniques mostly rely on manual operations, simple mechanical devices or regularised image processing algorithms. The main advantage of these methods is that they are low cost and have some application tasks for small-scale waste classification scenarios. Common traditional sorting methods include: 
(1) Manual sorting
Targeting and sorting of waste by post workers, although it can achieve better sorting, it is inefficient and prone to errors in high volume processing scenarios.
(2) Mechanical sorting
Sorting waste by mechanical means such as magnetic sorting, percolation sorting, concentration sorting, etc. It is usually used in large-scale recycling systems, but has limited effectiveness in sorting multiple types of waste and small waste.
(3) Regularised image processing
Target recognition of waste by image pre-processing and outputting motion features, usually combined with regular expressions and generic classification models, but the processing efficiency and accuracy are not high and there are significant gaps when dealing with complex environments.
While traditional methods maintain a certain level of efficiency for certain types of waste sorting, there is a need to implement smarter sorting solutions for more complex and high accuracy demanding scenarios. Deep learning models have the advantage of extracting sophisticated feature information from enormous datasets and thus are especially used in target detection and image classification. The original convolutional neural networks like AlexNet and VGG were a great leap in classification work but were not very efficient in multi-target detection and sometimes get confused with images of similar shapes, the primary reason being that these networks have to identify each target individually, which cannot match the speed requirements in real applications. Thus, the question of how to enhance the speed while maintaining high classification accuracy has been of major research interest.

2.2 Development of YOLO Target Detection Algorithm

 
Target detection in the computer vision domain has been a challenging and fundamental task, of identifying target classes in an image or video and deciding upon their place in the image. Sliding window-based traditional target methods involve sliding over different sizes of a window over the image and making classification decisions based on the contents of each window, and hence it becomes time and resource-intensive. With the advent of deep learning, target detection methods through CNN-based target algorithms have become the mainstream. One of the significant developments among them is the YOLO (You Only Look Once) algorithm, presented by Redmon et al. in 2016.
The principle of the YOLO algorithm lies in analyzing the target detecting issue as a regression problem and outputting the target's category and location from a neural network that processes the image in a single pass. Particularly, YOLO divides the input image into a number of S×S grids, each predicting the targets falling into the grid. Each of the grids produces B prediction boxes and their respective confidence scores, and C category probabilities. This structure allows YOLO to significantly enhance the detecting speed while guaranteeing high accuracy, and it is applicable to target detecting in real-world applications like smart security, self-driving cars, robot vision, etc.
As the YOLO algorithm becomes increasingly sophisticated, numerous versions have emerged to enhance its capability. YOLOv2, published in 2017, introduces the mechanism of anchor boxes. In YOLOv1, the mesh directly predicts the position coordinates of the bounding box, while YOLOv2 adopts the concept of anchor boxes from Faster R-CNN, predefining a batch of anchor boxes of different scales and aspect ratios, then letting each mesh predict the offsets and sizing parameters of the anchor boxes. This optimization further ameliorates the target recognition at different scales, particularly for small targets. In addition, YOLOv2 applies Batch Normalization to accelerate the convergence of the network and enhance the generalisation of the model. At the same time, it trains the network on high-resolution images, enabling the network to learn finer information.
YOLOv3, proposed in 2018, adds a residual network (ResNet) into the network structure. ResNet resolves the issue of gradient vanishing and explosion in deep neural networks, so the network can be designed deeper and hence extract more sophisticated semantic features. YOLOv3 enhances the accuracy of detection based on multi-scale prediction methods, and performs best in detecting small objects especially. It makes predictions from three different scales of feature maps, that is, large, medium and small targets, such that it can make full use of feature information at different scales to enhance the recall and precision of detection. Furthermore, YOLOv3 also adopts logistic regression to predict target categories, in place of the usage of the softmax function, for the reason that in real-world applications, there may exist overlap or correlative relationship between target categories, and logistic regression can handle the latter better.
YOLOv4, released in 2020, optimizes the network and loss function architectures. It incorporates some of the most recent techniques like CSPNet (Cross Stage Partial Network), a network structure that minimizes the amount of computations by partially connecting between stages while enhancing the network and feature reuse capabilities. In terms of data enhancement, YOLOv4 introduces the Mosaic data enhancement technique, which splices four images together for training, which not only enhances the diversity of the data, but also strengthens the model's ability to learn a variety of scenes and objects. In addition, YOLOv4 also makes use of some new optimisation strategies, such as the CIoU (Complete Intersection over Union) loss function, in which it takes into account factors such as the distance between bounding boxes, the overlap area, and the aspect ratio, in order to improve the accuracy of bounding box regression.
Even though YOLOv5 isn't a release by the official Redmon team but is optimised and improved further based on YOLOv4, it has garnered great popularity and usage. YOLOv5 also takes the CSPNet structure and lightens it, thus the model becomes faster and more efficient to run on resource-constrained devices while not sacrificing accuracy. As for data enhancement, YOLOv5 also includes Mosaic data augmentation and adds adaptive anchor frame calculations and adaptive image scaling, and all these enhance the generalisation capability and detection speed of the model. Also, YOLOv5 offers sizes of the model to be selected as per the user's requirements, i.e., in the case of higher speed requirements, they select a smaller one; in the case of higher accuracy requirements, they select a larger one.

2.3 Advantages of YOLOv8

YOLOv8, the newest in the YOLO series of models announced by Ultralytics in 2023, follows in the footsteps of the previous models with numerous improvements. Compared to the network architecture of the earlier models, YOLOv8 introduces a new Backbone and Neck design that provides improved feature extraction and fusing capabilities. Its Backbone network is based on the Extended Efficient Layer Aggregation Network (ELAN), which is able to extract feature information from images more efficiently, while the Neck network introduces an improved version of the PAN (Path Aggregation Network) structure, which strengthens the transfer of information between features of different scales, enabling the model to be more accurate in detecting targets of different sizes. In terms of training and optimisation, YOLOv8 introduces new loss functions and training strategies. For example, it adopts VFL (Varifocal Loss), which is a loss function that can better balance the weights of positive and negative samples to improve the accuracy of the model in target localisation and classification. Meanwhile, in terms of data enhancement, in addition to traditional methods such as Mosaic, it also introduces richer and more diverse enhancement strategies to further improve the generalisation ability of the model.
There are several important reasons for choosing the YOLOv8 model for the waste classification system. Firstly, there are different sizes, shapes and colours of waste targets in the waste sorting scenario, and YOLOv8’s powerful multi-scale target detection capability can accurately identify different sizes of waste, whether it is a large cardboard box or a small bottle cap, and so on. Secondly, waste sorting systems usually need to operate in real-time scenarios, such as on smart bins or waste sorting conveyors, YOLOv8’s optimised network structure and efficient inference speed can meet the real-time requirements and quickly identify the waste to be sorted. Furthermore, since there are many different types of rubbish classification, and there may be some similar-looking rubbish categories, YOLOv8’s high accuracy and improved loss function can better distinguish these similar rubbish categories and reduce the classification error rate. In addition, YOLOv8 has good model scalability and flexibility, which allows users to fine-tune or re-train the model according to the actual rubbish classification needs, in order to adapt to different rubbish datasets and application scenarios.




































CHAPTER 3
METHOD

In this study, an intelligent waste classification system based on YOLOv8 target detection algorithm is designed and implemented. In order to give full play to the performance advantages of YOLOv8 in waste classification, this paper optimises the model by constructing a homemade dataset and combining freeze-training and unfreeze-training strategies. This section will introduce the dataset construction process, model training settings, and the network structure of YOLOv8 algorithm with its advantages in detail.

3.1 Data Set Construction

In the Waste classification task, the quality of the dataset directly affects the classification effect of the model. In order to ensure the system’s ability to classify multiple categories of Waste, this paper collects 3490 Waste images from multiple public data sources via web crawlers, covering 36 different categories of Waste types. Each image is annotated by LabelImg tool in order to generate an annotation file in VOC format. The labelling information includes the category of each trash and its bounding box position in the image.
 
 

This dataset has a large category imbalance problem, and the number of samples in some spam categories is significantly less than others. To alleviate this problem, this paper performs data enhancement operations in the data preprocessing stage, including random cropping, rotating, scaling and other methods to increase the diversity of the training data and improve the generalisation ability of the model.

3.2 YOLOv8 Algorithm

YOLOv8 is the latest version of the YOLO target detection algorithm, which combines the advantages of the YOLO series algorithms and introduces a number of structural and technological improvements, and is characterised by fast speed, high accuracy and lightweight network structure. The following is the detailed structure of the YOLOv8 network and its advantages in the Waste classification task.
 
3.2.1 Network Architecture of YOLOv8
The network structure of YOLOv8 has been deeply optimised compared to its predecessor version and is divided into the following sections:
(1) Input Layer
YOLOv8 provides dynamic input resolution with a shared input resolution of dimension [640, 640], and it has the ability to lower the computational requirements while retaining the accuracy of the detection such that the inference rate increases. The input image is normalized to the defined dimensions and then fed into the network to extract the features. 
(2) Backbone Network
The backbone network of YOLOv8 adopts the CSP (Cross Stage Partial) architecture, which effectively reduces computational complexity by introducing segmentation and merging operations in each convolutional block while maintaining efficient feature representation. The CSP network can better retain the global information of the input image, thus improving the detection of complex targets.
(3) Feature Pyramid Network, FPN
In order to enhance the detection of multi-scale targets, YOLOv8 introduces an improved FPN structure. The FPN enables the model to focus on both large and small targets by fusing features from different scales. The network not only improves the accuracy of small target detection, but also ensures that targets at different scales are effectively recognised.
(4) Detection Head
The detection head of YOLOv8 is further optimised on the basis of the original YOLO series with multi-scale output. The network outputs three different scales of feature maps, each corresponding to a different size of target detection task. For example, the small feature map is employed to identify the large targets and the large feature map to identify the small targets. This not only enhances the network's detection accuracy, but also increases the accuracy in multi-class classification.
(5) Loss Function
YOLOv8 employs the GIoU (Generalised Intersection over Union) loss function for the optimization of bounding box prediction. Compared to the traditional IoU, GIoU is more effective in dealing with bounding boxes with little or no overlap, thus enhancing the model’s ability to predict target locations in complex backgrounds. In addition, YOLOv8 introduces improved classification and confidence loss functions, which makes the model perform more robustly in detecting multi-category targets.
3.2.2 Advantages of YOLOv8 in Waste Classification Tasks
(1) Real-time
Compared to other target detection algorithms, the biggest advantage of YOLOv8 is its ability to achieve real-time detection at a low computational cost. This makes the algorithm very suitable for scenarios that require fast response, such as Waste classification terminal equipment. With a lightweight backbone network and optimised detection head design, YOLOv8 is able to run at a high frame rate on low-configuration hardware, thus meeting the real-time requirements of real applications.
(2) High precision small target detection
There are a wide variety of targets in the waste classification task and some of them are small in size. YOLOv8 is able to effectively improve the detection accuracy of small targets and reduce the leakage rate through the improved FPN and GIoU loss functions. In addition, the application of CSP architecture further enhances the feature extraction capability and ensures the recognisability of small targets in complex backgrounds.
(3) Multi-category support
The waste classification system in this project requires accurate identification of 36 categories of waste classification. YOLOv8 is able to maintain high classification accuracy in the face of multiple categories through its multi-scale feature fusion and optimised classification loss function. Compared with its predecessor YOLO algorithm, YOLOv8 shows better stability and robustness in complex tasks with multiple categories.
(4) Flexible training strategies
YOLOv8 supports switching between two training phases, freezing and unfreezing, which allows us to firstly stabilise the basic part of the training network in the freezing phase, and then further optimise the network’s advanced feature extraction capability in the unfreezing phase. In this study, the training strategy of 100 frozen Epochs and 300 unfrozen Epochs is adopted, which enables the model to converge in a shorter time and obtain better classification accuracy.

3.3 Model Training Setup

During the model training process, we set the following key parameters to ensure the effectiveness and efficiency of the model:
(1) Input size: [640, 640] to ensure that the model can effectively detect targets at multiple scales.
(2) Freeze phase training: Freeze_Epoch = 100, freeze batch_size = 8.
(3) Unfreeze phase training: UnFreeze_Epoch = 200, unfreeze batch_size = 8.
(4) Learning rate: Initial learning rate (Init_lr) is set to 1e-2, Minimum learning rate is set to 1e-4, and Cosine Annealing (Cosine Annealing) is used to gradually reduce the learning rate, to ensure that the model can converge better in the later stages of training.
(5) Optimiser: an SGD optimiser with momentum set to 0.937 is used to accelerate the convergence of the model.
With these settings, we succeeded in achieving efficient training of the model and achieved satisfactory results in the waste classification task.




























CHAPTER 4
RESEARCH

4.1 Secondary Research

4.1.1 Literature Review
Several research papers and white papers were analyzed to understand existing methods and their limitations. Key findings include:
(1) Traditional classification methods such as manual sorting and mechanical separation lack efficiency and accuracy.
(2) Machine learning-based approaches such as CNNs (AlexNet, VGG) improve accuracy but struggle with real-time processing.
(3) YOLO-based models, especially YOLOv8, offer an ideal balance between accuracy and speed due to their optimized architecture.
4.1.2 Standards and Regulations
To align the system with industry practices, relevant waste classification standards were reviewed:
(1) European Waste Classification (EWC) standards for categorizing waste materials.
(2) ISO 14001 environmental management guidelines.
(3) National smart recycling policies in China and the UK.
These standards helped define waste categories and ensure compliance with international environmental practices.

4.2 Informal Sources: User Stories & Case Studies

4.2.1 User Stories
User requirements were gathered from recycling centers and municipal waste management facilities. Key epics included:
(1) As a waste management operator, I need an automated system to identify waste categories in real-time to reduce manual labor.
(2) As a smart bin manufacturer, I want an embedded AI model to classify waste and provide disposal guidance.
(3) As a local government official, I need an AI-driven waste classification system to improve recycling efficiency and policy compliance.
4.2.2 Case Studies
(1) Smart waste Bins in Shanghai: AI-powered bins with image recognition improved sorting accuracy by 30%.
(2) Google’s AI waste Sorting Initiative: Demonstrated a deep learning-based approach for waste classification in corporate settings.
(3) Municipal AI Sorting Systems in the UK: Showed that automation could reduce waste processing costs by 20%.

4.3 Technology Selection: Pugh Matrix

A Pugh matrix was used to compare YOLOv8 with alternative models such as Faster R-CNN, SSD, and ResNet-based classifiers.
Criteria	YOLOv8	Faster R-CNN

	SSD		ResNet Classifier
	

Accuracy
	9/10
	9/10	8/10	7/10
Speed
	10/10	6/10	9/10	5/10
Model Size
	8/10	6/10	9/10	7/10
Complexity
	7/10	5/10	7/10	6/10
Real-time Performance
	10/10	6/10	9/10	4/10
	
Ease of Training
	9/10	7/10	8/10	7/10
Total Score
	53	39	50	36
YOLOv8 outperformed other models in speed, real-time performance, and ease of training, making it the most suitable choice for this application.

4.4 Main Argument and Alternative Arguments

4.4.1 Main Argument
This study argues that YOLOv8 is the optimal deep learning model for real-time waste classification due to its high accuracy, fast inference speed, and ability to detect multiple waste categories efficiently. The research demonstrates that:
(1) YOLOv8 achieves superior classification accuracy compared to other models.
(2) The model operates efficiently in real-time applications such as smart bins and sorting facilities.
(3) The combination of deep learning with waste classification enhances automation and reduces reliance on manual labor.
4.4.2 Alternative Arguments
(1) Faster R-CNN offers higher accuracy: While this is true, Faster R-CNN is computationally expensive and unsuitable for real-time applications.
(2) Rule-based systems are more interpretable: Although rule-based classification is transparent, it lacks adaptability to new waste types.
(3) Hybrid approaches could yield better results: Combining deep learning with traditional sorting may improve performance, but it increases system complexity and cost.












CHAPTER 5
EXPERIMENTS AND RESULTS

In order to verify the performance of the YOLOv8-based waste classification system in real tasks, this paper conducts a large number of experiments on the model and evaluates the system in detail by a number of metrics. The experiments mainly focus on the training process of the model, classification accuracy, and performance evaluation. The experimental setup and the analysis of its results are described in detail below.

5.1 Experimental Environment

All experiments were conducted on NVIDIA GeForce GTX 3060 GPUs using the main framework PyTorch 1.8.0 and Python version 3.8. The implementation of the YOLOv8 algorithm was carried out based on open-source libraries, and the training parameters were adjusted accordingly to the task requirements.

5.2 Data Set

The experimental dataset is a self-made waste classification dataset, which contains 3490 images covering 36 waste categories. All images are labelled by LabelImg tool, and the labelling information of each image includes the category of the target and its bounding box position. In order to improve the generalisation ability of the model, the dataset was subjected to data enhancement operations during the training process, including random cropping, rotation and scaling. The dataset is randomly divided into training and validation sets in the ratio of 8:2 to ensure that the model can learn sufficiently during the training process, while its generalisation performance is evaluated on the validation set.

5.3 Model Training Process

5.3.1 Training Phase Loss Curves
In the initial stage of model training, the backbone network of YOLOv8 is trained using a freezing strategy to ensure the stability of the base feature extraction part. In this stage, the training lasts for 100 Epochs, the batch size (batch_size) is set to 8, the initial learning rate is 1e-2, and the learning rate is gradually reduced by using the Cosine Annealing (Cosine Annealing) strategy.
As the training proceeds, the loss value gradually decreases and the model gradually converges. At the end of the freezing phase, the loss of the model on the validation set stabilises, indicating that the backbone network has learned effective features.
In the unfreezing phase, all network layers were involved in the training, with the main objective of further optimising the high-level feature extraction part. The unfreezing phase training lasted for 200 Epochs, the batch size was kept at 8, and the initial learning rate was set to 1e-3 and gradually reduced to 1e-4.
 

5.3.2 Training Phase Map Curve
With the increasing number of trainings, the map value of the model is increasing
 

5.4 Model Performance Evaluation

5.4.1 Mean Accuracy (mAP)
Mean Average Precision (mAP) is the main index to evaluate the performance of target detection models. mAP considers the detection precision of different categories and combines Precision and Recall, which can effectively measure the classification ability of the model. In this paper, mAP is calculated under different threshold conditions, and the results are shown in the Table:
Thresholds	Map value
IoU=0.50:0.95	0.705
IoU=0.50	0.958
IoU=0.75	0.820
It can be seen that the model achieves a mean accuracy of 95.8% at a threshold of 0.5, indicating that the model has high detection accuracy on most categories. As the threshold value increases, the mAP decreases, but generally stays above 75%, indicating that the model has a high recognition ability for the 36 categories of waste classification task.
Below is the map value performance IoU=0.50 specific to each category:
 
5.4.2 AP Chart
In target detection task evaluation, the Precision - Recall curve and Average Precision (AP) are the key metrics to measure the model’s detection performance for a specific class of targets. The following is a detailed analysis of the AP graphs for six categories of targets (bags, cigarettes, power banks, cans, nappies, thermometers):


bags: The AP value of the bags category reaches 100.00%. As can be seen from the figure, as the recall rate gradually increases from 0 to 1.0, the precision rate always stays stable at 1.0. This shows that the model has very high accuracy and completeness in detecting the targets of bags. That is to say, the model determines that all the targets of bags are correct, and it can successfully detect all the targets of bags that actually exist, and there are almost no false detections and omissions, so the detection performance is extremely excellent.
cigarette: The AP value in the current category was 86.29%. When the precision rate holds at 1.0 at the level of the low recall rate, all of the detected cigarette targets at the first period are correct. With the growth of the recall rate, the precision rate also exhibits a clear declining trend, and after the recall rate grows to a certain degree, the precision rate fluctuates and drops. This indicates that when the model raises the number of detected cigarette targets (the recall rate also increases), the number of false recognitions also increases, and the precision rate becomes lower, and the entire detection performance needs to be optimized again.
power bank: The AP value of power bank category is 93.01%. Its precision-recall curve shows the characteristics of smooth, then decreasing, and then fluctuating. When the recall rate starts to increase, the precision rate stays at 1.0, then decreases and then fluctuates. This shows that the model performs well in detecting power bank targets in the early stage, but as more targets are detected, the precision rate is affected, although the overall performance is still relatively good compared to some categories, but there is still room for improvement.
cans: The AP value for the cans class is 68.89%. When the recall is low, the precision rate stays at 1.0, but when the recall increases to a certain level (after about 0.6), the precision rate decreases sharply. This suggests that the model’s detection performance is poor when detecting cans targets, as more targets are detected, the false detections increase dramatically, and measures need to be taken in model optimisation, data processing, etc. to improve the performance.
nappies: The AP value for the nappies category is 100.00%. The precision rate is always stable at 1.0 throughout the recall change process. This means that the model accurately and comprehensively identifies all targets when detecting nappies class targets, with excellent detection results, both in terms of accuracy and completeness.
thermometers: The AP value of thermometers class is 100.00%. The precision rate is always maintained at 1.0 during the change of the recall rate from 0 to 1.0, which indicates that the model is very capable of detecting the targets of thermometers class, and it can detect all the targets of this class accurately and correctly without any misdetection or omission problem, and the detection performance is perfect.
5.4.3 F1 Chart
The F1 score is an evaluation metric that combines precision and recall to give a more comprehensive picture of the model performance. The six graphs show how the F1 scores of bags, cans, cigarettes, nappies, power banks, and thermometers vary with the Score_Threshold, which is set to 0.5, and are analysed below:

bags: The model performed well in detecting the ‘bags’ category, maintaining a near-perfect F1 score (close to 1.0) at all confidence thresholds. This means that it consistently maintains a balance between accuracy (correctly identifying bags without too many false positives) and recall (capturing most of the actual bags without missing too many). However, when the confidence threshold is set very high (close to 1.0), the F1 score drops drastically as the model becomes overly cautious and only makes predictions when it is almost certain, resulting in fewer overall detections. This behaviour is to be expected, as the requirement for near-absolute confidence naturally reduces the number of predictions. In practice, the model remains highly reliable as long as the thresholds are not pushed to unrealistic extremes.
cans: The F1 value of cans class was 0.75. The F1 score curve shows a trend of fluctuation, then increase, and then a sharp decrease. When the confidence threshold is low, the F1 score fluctuates, indicating that the model detection performance is unstable; with the increase of the threshold, the F1 score rises, and reaches a relative high point when the threshold is close to 0.8; and then falls sharply, indicating that the model detection ability is greatly limited by the high confidence threshold, on the whole, the model detection performance of cans targets needs to be improved, and the optimisation of the selection of confidence thresholds is needed for the stable performance. Stable performance.
cigarette: The F1 value for the cigarettes category is 0.82. The curve shows a gradual increase, with several small fluctuations in the process, and begins to decline after the confidence threshold has risen to a certain level. This indicates that within a certain threshold range, the detection performance of the model gradually improves, but due to large fluctuations, the detection stability is not good; beyond a certain threshold, the performance decreases due to the decline in the precision rate or recall rate, and there is still room to improve the detection performance of the model for the category of cigarettes, and it is necessary to further adjust the parameters in order to enhance the stability.
nappies: The F1 value for the nappies category reaches 1.0. As the confidence threshold increases from 0 to nearly 0.8, the F1 score remains high and relatively stable, with a sharp drop only when the threshold approaches 1.0. This shows that the model is very effective in detecting nappies, with excellent detection performance in terms of precision and recall over a wide range of thresholds.
power bank: The F1 value for the power bank category is 0.86. The curve first fluctuates at a high level, and then begins to decline after the confidence threshold increases to a certain level. The fluctuation indicates that the detection performance of the model varies under different thresholds, but the overall level is maintained at a good level; the downward trend indicates that too high a threshold will deteriorate the performance of the model, and that the model has a certain performance for the detection of power bank targets, but needs to be optimised in order to enhance the stability and cope with high thresholds.
thermometers: The model achieves strong results for thermometers, with an F1 score of 0.95, reflecting a good balance between precision and recall. Performance remains stable across moderate confidence thresholds, meaning the model reliably detects thermometers without excessive false positives or missed detections. However, as the confidence threshold approaches 1.0, the F1 score declines sharply—likely because the model becomes too conservative, only making predictions when it’s extremely confident, which drastically reduces detection rates. While the overall performance is solid, this sensitivity to high thresholds suggests room for improvement, particularly in maintaining robustness when stricter confidence levels are applied.
5.4.4 Precision and Recall
In order to further evaluate the performance of the model in the waste classification task, this paper calculates the precision rate and recall rate for each category separately. The precision rate indicates the proportion of the model that truly belongs to a category in the prediction results, while the recall rate indicates the ability of the model to correctly identify all instances of the category. The results of precision and recall for some key categories are shown in Fig:
  


It can be seen that the model possesses high precision and recall in classifying recyclable categories such as newspapers and napkins, showing good classification results. However, the model has some errors in identifying food waste categories such as banana skin and green vegetables, which may be related to the fact that there are fewer samples of these categories in the dataset. The recognition of these categories can be improved in the future by dataset expansion and further optimization of the network.









5.4.5 Ground-truth
 

This is a bar chart showing the number of instances of each target category, covering 349 documents and 36 categories. The horizontal coordinate is the number of targets corresponding to each category and the vertical coordinate is the target category name. It can be seen from the graph:
The number of targets per category varies considerably. The ‘Bags’ category had the highest number of targets, with 40, while the “Dolls” and ‘Bread’ categories also had a higher number of targets, with 37 and 29, respectively. The ‘greens’ and ‘chocolate’ categories had the lowest number of targets, with only one.
Unbalanced data may affect model training. A large number of categories allows the model to learn richer features and be more adequately trained; with a small number of categories, the model learns only a limited number of features, which may lead to poor performance in detecting the targets in these categories, such as missed detection. Data enhancement, oversampling, and other methods can be considered to balance the data subsequently.
5.4.6 Log-average Miss Rate
 
The figure shows the log mean missed detection rate for different categories of targets. The horizontal coordinate is the value of the log mean leakage rate and the vertical coordinate is the name of the target category. Specific analyses are presented below:
‘Egg shellsdddaa’ has the highest log-mean miss rate of 0.80, implying that targets in this category are more difficult to detect and are easily missed by the model; “Shoes” has a miss rate of 0.53, “cans” 0.40, “cigarette” 0.34, and these categories also have relatively high miss rates. Shoes has a miss rate of 0.53, “cans” has a miss rate of 0.40, and “cigarette” has a miss rate of 0.34, which are also relatively high. Many categories such as ‘masks’, ‘power bank’, ‘cups’, etc. have a leakage rate of 0.00, which indicates that the model is very effective in detecting these categories and there is almost no leakage problem. There is almost no leakage detection problem.
For the categories with high leakage detection rate, it may be due to the complexity of their appearance features, their small number in the dataset, and their easy confusion with other categories. Subsequently, the model can be further optimized for these categories, such as adjusting the network structure, increasing the training data, improving the feature extraction method, etc., in order to reduce the leakage detection rate and improve the overall detection performance of the model.
5.4.7 Confusion Matrix 
This confusion matrix shows the predictions of the model for different categories of targets, the rows represent the Ground - truth label and the columns represent the Predicted label, the values in the matrix represent the number of predicted results for the corresponding category, and the colour shades reflect the size of the values.
The diagonal elements show correct predictions—for instance, the value 26 for ‘dolls’ means the model accurately identified 26 doll samples. However, most diagonal values are small, suggesting frequent false positives (incorrectly labeling other items as belonging to these categories). The off-diagonal elements reveal misclassifications. For example:
1.	12 ‘Egg shellsdddaa’ samples were wrongly predicted as ‘rice’,
2.	10 ‘plastic bags’ samples were confused with ‘bags’.
These errors highlight the model’s struggle with visually or semantically similar categories (e.g., plastic bags vs. bags). To improve performance, refining distinctions between such overlapping classes—through better training data or model adjustments—could help.
Looking at the matrix as a whole, the model misclassifies some categories (e.g. ‘Shoes’ ‘green vegetables’ etc.) less often, while it misclassifies categories such as ‘Egg shellsdddaa ‘cigarette’ etc. are misclassified more often. This indicates that the performance of the model in identifying these confusing categories needs to be improved, and it may be necessary to optimise the feature extraction method, increase the diversity of the training data, or adjust the model structure in order to improve the ability to differentiate between the categories, especially the confusing ones, and thus improve the overall detection accuracy.

5.5 System Interface and User Experience

Based on the experimental results, we developed two user interfaces based on PyQt5, both of which have the same functionality, just different layouts. It is used to show the actual application effect of the system. The system interface is simple and intuitive, and the user can select a spam image to be detected through the interface, and the system will automatically detect it and display the prediction results. The response speed of the system is fast, with an average recognition time of about 0.5 seconds per image, which indicates that the system has good real-time performance and user experience in practical applications. The following figure shows an example of the system interface and the recognition results:
 
 



 

CHAPTER 6
CONCLUSION

In this study, a waste classification system based on the YOLOv8 algorithm was designed and implemented, and the effectiveness of the system in the multi-category waste classification task was verified by training and testing the homemade dataset. The experimental results show that the system exhibits high performance in terms of classification accuracy and recall, especially when dealing with major categories such as recyclables and food waste. However, there is still room for improvement in the classification effectiveness of the model on certain categories, and the experimental results are analysed and discussed in depth below. 

6.1 Analysis of Model Strengths

6.1.1 High Accuracy and Efficiency
The experimental results show that the YOLOv8-based waste classification system achieves a good balance between accuracy and efficiency. By introducing a multi-scale feature pyramid (FPN) and CSP architecture, the system is able to effectively improve the detection accuracy of small targets while maintaining a high inference speed. The system achieves an average accuracy of 95.8% at a mAP of 0.5, which proves the excellent performance of YOLOv8 in complex multi-category classification tasks. In addition, the system is still able to train stably and converge rapidly with a batch size of only 8, showing the computational efficiency of the YOLOv8 network structure.
6.1.2 Real-time and Deployment Potential
As the goal of the waste classification system is to achieve efficient automated classification in real-world applications, real-time performance is an important indicator of system performance. YOLOv8 adopts a lightweight backbone network design and optimises the processing speed of the system in the inference phase by dynamically adjusting the input resolution. In the experimental environment of this study, the system is able to achieve high inference speeds on NVIDIA GTX 3060 GPUs, indicating good real-time performance and deployment potential. Compared to traditional target detection algorithms, the system is able to better meet the demand for real-time and high efficiency in waste classification applications.

6.2 Problems and Limitations
Although the system achieves good results in terms of overall performance, there are still some limitations in the performance of the model in some specific categories of classification tasks. In particular, the system classification results are relatively poor for similar objects in the waste categories, such as ‘Food waste’ and ‘Other waste’. Analysis of the confusion matrix shows that there is a high rate of confusion between some of the categories, especially for those with similar appearance characteristics, which are easily misclassified. This problem may be caused by the following factors:
(1) Data Set Imbalance
The uneven distribution of categories in the dataset samples leads to insufficient samples in some categories, which affects the classification accuracy of the model. For some junk categories with fewer samples, it is difficult for the model to learn enough feature information, resulting in a weak recognition ability on these categories. One of the future directions for improvement is to expand the dataset, especially to increase the training data of the under-sampled categories to further enhance the generalisation ability of the model.
(2) Confusion of Features Between Similar Categories
In the waste classification task, different categories of waste may have similar appearance characteristics. For example, it is often difficult to distinguish between ‘food waste’ and ‘other waste’ in terms of appearance, leading to model confusion during detection. To solve this problem, more complex feature extraction networks or the use of additional sensor data (e.g., weight, material composition, etc.) can be considered to help the model differentiate between these hard-to-identify categories.
(3) Limitations of Small Target Detection Performance
Despite the optimisation of YOLOv8 for small target detection, the detection capability of the system still needs to be improved for certain spam objects with very small sizes or in complex backgrounds. Further improvement of the system’s detection accuracy for small targets can be considered in combination with super-resolution image generation techniques or by using a finer multi-scale feature fusion mechanism.
6.3 Directions for Future Improvement
Future work can further extend the dataset for waste classification to include more types of waste and image collection in different scenarios. For example, more real scene data can be collected from sensors in real applications to improve the applicability and robustness of the system. In addition, data enhancement techniques can be further optimised, e.g., by introducing more sophisticated image enhancement methods (e.g., GAN-generated synthetic data) to increase the model’s adaptability to unknown scenes.
Although YOLOv8 performs well in terms of detection performance, there is still room for optimisation. Future research can consider further simplifying the network structure and reducing the computational complexity to adapt to resource-limited hardware deployment scenarios. At the same time, the fusion of Transformer structure or the introduction of self-supervised learning methods can be considered to enhance the generalisation ability of the model to different scenarios and categories.
The current system is only based on image data for classification, but in practical applications, the waste classification task can be combined with data from other modalities, such as weight, sound, and other multi-sensor information. By fusing multimodal data, the accuracy and robustness of the model for complex waste classification tasks can be further improved. The multimodal system can better cope with waste categories that are similar in appearance but different in nature, and improve the classification accuracy of the system.
The YOLOv8-based waste classification system has good real-time performance and high efficiency, demonstrating great potential in practical waste classification applications. The system can be integrated into smart waste classification devices or robots to achieve the goal of automated waste classification. In the future, with further improvement in model accuracy and hardware performance, the system is expected to be widely used in large-scale waste classification projects. Through cooperation with governments and environmental organisations, the system can promote the further development of waste treatment automation and help global environmental protection.
In summary, the waste classification system based on YOLOv8 shows high accuracy and real-time performance in multi-category waste classification tasks, and has potential for practical application. Although there are some classification errors on a few categories, the performance of the system is expected to be further improved by further optimising the dataset, improving the model structure and introducing multimodal data fusion. This study provides new ideas and technical support for the development of intelligent systems for automated waste classification.

REFERENCES

[1] Bochkovskiy, Alexey & Wang, Chien-Yao & Liao, Hong-yuan. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection.
[2] Chen Yan. [陈妍]. (2024). 基于YOLOv8的深度学习目标检测研究[J]. 消费电子, (5), 34-36.
[3] Gao Hao [高昊]. (2024). 基于YOLOv8的垃圾分类方法研究[J]. 环境保护科学, (2), 36-40.
[4] Jocher, G.& Chaurasia, A. (2023). YOLOv8: Real-Time Object Detection Model. Ultralytics Documentation.
[5] Li Jia. [李佳]. (2023). 基于YOLOv8的室内物品分类与检测[J]. 计算机应用技术, (4), 66-71.
[6] Li Ming Jie, Wang Peng. [李明杰][王鹏]. (2023). 基于改进YOLOv8算法的遥感图像目标检测[J]. 激光与光电子学进展, (12), 100-106.  
[7] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(2), 318-327.
[8] Liu C, Tao Y, Liang J, et al. (2018). Object detection based on YOLO network[C]. In 2018 IEEE 4th information technology and mechatronics engineering conference (ITOEC). (pp. 799-803). IEEE. 
[9] Liu Zhi Cheng, Chen Jiang [刘志成][陈江].(2024). 基于深度学习的YOLO目标检测综述[J]. 电子信息与通信技术, (3), 12-18.
[10] M. Shroff, A. Desai and D. Garg. (2023). YOLOv8-based Waste Detection System for Recycling Plants: A Deep Learning Approach. In 2023 International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS) (pp. 01-09). Erode, India. 
[11] Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.
[12] Sun Z, Chen B. (2022). Research on Pedestrian Detection and Recognition Based on Improved YOLOv6 Algorithm[C]. In International Conference on Artificial Intelligence in China. (pp. 281-289). Singapore: Springer Nature Singapore.
[13] Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2020). Scaled-YOLOv4: Scaling Cross Stage Partial Network. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[14] Wang Yu, Chen Yong [王宇][陈勇]. (2023). 基于改进YOLOv8的火灾目标检测系统[J]. 计算机应用研究, (8), 123-128.
[15] Wen, Guihao, et al. (2024). The improved YOLOv8 algorithm based on EMSPConv and SPE-head modules. Multimedia Tools and Applications, 1-17.
[16] Zhang De Ying, Zhao Zhi Heng [张德银][赵志恒]. (2023). 基于YOLOv8的遥感图像飞机目标检测[J]. 遥感学报, (9), 89-92.  
[17] Zhang Wei, Liu Hui [张伟][刘辉]. (2023). 基于YOLOv8的可回收垃圾识别方法研究[J]. 环境科学与技术, (7), 55-58.  
[18] Zhao H, Zhang H, Zhao Y. (2023). Yolov7-sea: Object detection of maritime uav images based on improved yolov7[C]. Proceedings of the IEEE/CVF winter conference on applications of computer vision, 233-238. 
[19] Zhu X, Lyu S, Wang X, et al. (2021). TPH-YOLOv5: Improved YOLOv5 based on transformer prediction head for object detection on drone-captured scenarios[C]. Proceedings of the IEEE/CVF international conference on computer vision, 2778-2788. 
[20] Z. Nie, W. Duan and X. Li. (2021). Domestic garbage recognition and detection based on Faster R-CNN. Journal of Physics: Conference Series, 1738(1), 012089.

