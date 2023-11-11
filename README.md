# T81 558:Applications of Deep Neural Networks
[Washington University in St. Louis](http://www.wustl.edu)

Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/)

*This branch contains the evolving PyTorch version of this course.  It is a work in progress and is not yet complete.*

**The content of this course changes as technology evolves**, to keep up to date with changes [follow me on GitHub](https://github.com/jeffheaton).

* Section 1. Fall 2023, Monday, 2:30 PM, Location: Louderman/461
* Section 2. Fall 2023, Online

# Course Description

Deep learning is a group of exciting new technologies for neural networks. Through a combination of advanced training techniques and neural network architectural components, it is now possible to create neural networks that can handle tabular data, images, text, and audio as both input and output. Deep learning allows a neural network to learn hierarchies of information in a way that is like the function of the human brain. This course will introduce the student to classic neural network structures, Convolution Neural Networks (CNN), Long Short-Term Memory (LSTM), Gated Recurrent Neural Networks (GRU), General Adversarial Networks (GAN) and reinforcement learning. Application of these architectures to computer vision, time series, security, natural language processing (NLP), and data generation will be covered. High Performance Computing (HPC) aspects will demonstrate how deep learning can be leveraged both on graphical processing units (GPUs), as well as grids. Focus is primarily upon the application of deep learning to problems, with some introduction to mathematical foundations. Students will use the Python programming language to implement deep learning using PyTorch. It is not necessary to know Python prior to this course; however, familiarity of at least one programming language is assumed. This course will be delivered in a hybrid format that includes both classroom and online instruction.

# Objectives

1. Explain how neural networks (deep and otherwise) compare to other machine learning models.
2. Determine when a deep neural network would be a good choice for a particular problem.
3. Demonstrate your understanding of the material through a final project uploaded to GitHub.

# Syllabus
This syllabus presents the expected class schedule, due dates, and reading assignments.  [Download current syllabus.](https://data.heatonresearch.com/wustl/jheaton-t81-558-fall-2023-syllabus.pdf)

Module|Content
---|---
[Module 1](t81_558_class_01_1_overview.ipynb)<br>**Meet on 08/28/2023** | **Module 1: Python Preliminaries**<ul><li>Part 1.1: Course Overview<li>Part 1.2: Introduction to Python<li>Part 1.3: Python Lists, Dictionaries, Sets & JSON<li>Part 1.4: File Handling<li>Part 1.5: Functions, Lambdas, and Map/ReducePython Preliminaries<li>**We will meet on campus this week! (first meeting)**</ul>
[Module 2](t81_558_class_02_1_python_pandas.ipynb)<br>Week of 09/11/2023 | **Module 2: Python for Machine Learning**<ul><li>	Part 2.1: Introduction to Pandas for Deep Learning<li>Part 2.2: Encoding Categorical Values in Pandas<li>Part 2.3: Grouping, Sorting, and Shuffling<li>Part 2.4: Using Apply and Map in Pandas<li>Part 2.5: Feature Engineering in Padas<li>[Module 1 Program](./assignments/assignment_yourname_class1.ipynb) due: 09/12/2023<li> Icebreaker due: 09/12/2023</ul>
[Module 3](t81_558_class_03_1_neural_net.ipynb)<br>Week of 09/18/2023 | **Module 3: PyTorch for Neural Networks**<ul><li>Part 3.1: Deep Learning and Neural Network Introduction<li> Part 3.2: Introduction to PyTorch<li>Part 3.3: Encoding a Feature Vector for PyTorch Deep Learning<li>Part 3.4: Early Stopping and Network Persistence<li>Part 3.5: Sequences vs Classes in PyTorch<li>[Module 2: Program](./assignments/assignment_yourname_class2.ipynb) due: 09/19/2023</ul>
[Module 4](t81_558_class_04_1_kfold.ipynb)<br>Week of 09/25/2023 |**Module 4: Training for Tabular Data**<ul><li>Part 4.1: Using K-Fold Cross-validation with PyTorch<li>Part 4.2: Training Schedules for PyTorch<li>Part 4.3: Dropout Regularization<li>Part 4.4: Batch Normalization<li>Part 4.5: RAPIDS for Tabular Data<li>[Module 3 Program](./assignments/assignment_yourname_class3.ipynb) due: 09/26/2023</ul>
[Module 5](t81_558_class_05_1_python_images.ipynb)<br>**Meet on 10/02/2023** | **Module 5: CNN and Computer Vision**<ul><li>5.1 Image Processing in Python<li>5.2 Using Convolutional Neural Networks<li>5.3 Using Pretrained Neural Networks<li>5.4 Looking at Generators and Image Augmentation<li>5.5 Recognizing Multiple Images with YOLO<li>[Module 4 Program](./assignments/assignment_yourname_class4.ipynb) due: 10/03/2023<li>**We will meet on campus this week! (second meeting)**</ul>
[Module 6](t81_558_class_06_1_transformers.ipynb)<br>Week of 10/16/2023 | **Module 6: ChatGPT and Large Language Models**<ul><li>6.1: Introduction to Transformers<li>6.2: Accessing the ChatGPT API<li>6.3: Llama, Alpaca, and LORA<li>6.4: Introduction to Embeddings<li>6.5: Prompt Engineering<li>[Module 5 Program](./assignments/assignment_yourname_class5.ipynb) due: 10/17/2023</ul>
[Module 7](t81_558_class_07_1_img_generative.ipynb)<br>Week of 10/23/2023 | **Module 7: Image Generative Models**<ul><li>7.1: Introduction to Generative AI<li>7.2: Generating Faces with StyleGAN3<li>7.3: GANS to Enhance Old Photographs Deoldify<li>7.4: Text to Images with StableDiffusion<li>7.5: Finetuning with Dreambooth<li>[Module 6 Assignment](./assignments/assignment_yourname_class6.ipynb) due: 10/24/2023</ul>
[Module 8](t81_558_class_08_1_kaggle_intro.ipynb)<br>**Meet on 10/30/2023** | **Module 8: Kaggle**<ul><li>8.1 Introduction to Kaggle<li>8.2 Building Ensembles with Scikit-Learn and PyTorch<li>8.3 How Should you Architect Your PyTorch Neural Network: Hyperparameters<li>8.4 Bayesian Hyperparameter Optimization for PyTorch<li>8.5 Current Semester's Kaggle<li>[Module 7 Assignment](./assignments/assignment_yourname_class7.ipynb) due: 10/31/2023<li>**We will meet on campus this week! (third meeting)**</ul>
[Module 9](t81_558_class_09_1_faces.ipynb)<br>Week of 11/06/2023 | **Module 9: Facial Recognition**<ul><li>9.1 Detecting Faces in an Image<li>9.2 Detecting Facial Features<li>9.3 Image Augmentation<li>9.4 Application: Emotion Detection<li>9.5 Application: Blink Efficiency<li>[Module 8 Assignment](./assignments/assignment_yourname_class8.ipynb) due: 11/07/2023</ul>
[Module 10](t81_558_class_10_1_timeseries.ipynb)<br>Week of 11/13/2023 | **Module 10: Time Series in PyTorch**<ul><li>Time Series Data Encoding for Deep Learning, PyTorch<li>Seasonality and Trend<li>LSTM-Based Time Series with PyTorch<li>CNN-Based Time Series with PyTorch<li>Predicting with Meta Prophet<li>[Module 9 Assignment](./assignments/assignment_yourname_class9.ipynb) due: 11/14/2023</ul>
[Module 11](t81_558_class_11_1_hf.ipynb)<br>Week of 11/20/2023 | **Module 11: Natural Language Processing**<ul><li>11.1 Introduction to Natural Language Processing<li>11.2 Hugging Face Introduction<li>11.3 Hugging Face Tokenizers<li>11.4 Hugging Face Data Sets<li>11.5 Training a Model in Hugging Face<li>[Module 10 Assignment](./assignments/assignment_yourname_class10.ipynb) due: 11/21/2023</ul>
[Module 12](t81_558_class_12_1_ai_gym.ipynb)<br>Week of 11/27/2023 | **Module 12: Reinforcement Learning**<ul><li>Kaggle Assignment due: 11/27/2023 (approx 4-6PM, due to Kaggle GMT timezone)<li>Introduction to Gymnasium<li>Introduction to Q-Learning<li>Stable Baselines Q-Learning<li>Atari Games with Stable Baselines Neural Networks<li>Future of Reinforcement Learning</ul>
[Module 13](t81_558_class_13_1_automl.ipynb)<br>**Meet on 12/04/2023** | **Module 13: Deployment and Monitoring**<ul><li>Part 13.1: Flask and Deep Learning Web Services <li>Part 13.2: Interrupting and Continuing Training<li>Part 13.3: Using a PyTorch Deep Neural Network with a Web Application<li>Part 13.4: When to Retrain Your Neural Network<li>Part 13.5: Tensor Processing Units (TPUs)<li>**We will meet on campus this week! (fourth meeting)**<li>Final project due: 12/05/2023</ul>

# Datasets

* [Datasets can be downloaded here](https://data.heatonresearch.com/data/t81-558/index.html)

