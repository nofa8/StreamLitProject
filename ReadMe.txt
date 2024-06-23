Project: Image Classification
Objectives
In this project, students are required to:

Use an image dataset for classification tasks.
Train models based on convolutional neural networks (CNNs).
Write a report using markdown annotations in the developed notebooks. The report must include:
All steps taken to build the models.
A description of all experiments and results. This includes analyzing metrics (e.g., confusion matrices, accuracy, precision, recall, F1 score), suitable graphs, and result analysis.
Dataset
The provided dataset is divided into 6 directories: 5 training directories (train#) and 1 test directory. All groups use the same test directory. Each group uses 4 training directories for training and 1 directory for validation. The validation directory for each group is determined by summing the last digit of each group member's student number, taking the modulo 5 of the sum, and adding 1. For example:

Student 1: 2200783
Student 2: 2243929
Sum: 12 % 5 + 1 = 3
Validation set: train3
Training set: train1, train2, train4, and train5
Requirements
The project must meet the following requirements:

Use and describe training, validation, and test datasets.
Use RGB images (three channels).
Develop at least one model from scratch (model S), different from the one developed in class.
Explore at least two different optimizers.
Train S models with and without data augmentation.
Develop models using transfer learning (model T) with feature extraction and fine-tuning.
Train T models with and without data augmentation.
Evaluation
Data Processing: 5%
S Models: 35%
T Models: 30%
Report: 20%
Extras: 10%
Innovation and going beyond the course content (e.g., regularization techniques, deploying models as applications, custom data augmentation) are encouraged and will be rewarded.

Deadlines and Rules
Submission deadline: June 22, 2024, 23:59.
Projects are to be done in groups of 2 students. Individual projects are allowed only with written permission from the course instructor.
Submit the project as a zip file named dl_project_#1_#2.zip, where #1 and #2 are the student numbers. The zip file must contain:
Completed notebooks (.ipynb) after execution.
Notebooks (.ipynb) without markdown content before execution.
PDFs of notebooks with execution results.
The models.
Calculated features (for models using transfer learning without data augmentation).
An oral exam may be required in certain cases. The oral exam grade will multiply the project grade. The list of students required to take the oral exam will be published on Moodle after the submission.
