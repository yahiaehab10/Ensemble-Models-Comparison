# Banknote Authentication

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [License](#license)

## Introduction
This project demonstrates the use of ensemble machine learning models to classify banknote authentication data. The dataset used contains various features extracted from banknote images and a class label indicating whether the banknote is authentic or not.

## Dataset
The dataset used for this project is the Banknote Authentication dataset, which contains the following features:
- Variance of Wavelet Transformed image
- Skewness of Wavelet Transformed image
- Curtosis of Wavelet Transformed image
- Image Entropy
- Class (0 for authentic, 1 for inauthentic)

## Installation
To install the required dependencies, use the following commands:
```sh
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `BanknoteAuthentication.ipynb` notebook to execute the code.

## Project Structure
- `BanknoteAuthentication.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and visualization.

## Data Preprocessing
The project starts with loading and preprocessing the banknote authentication data.

### Code Example:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("./AMLAss1Datasets/data_banknote_authentication.csv")

# Rename columns
df.columns = [
    "variance_wavelet", "skewness_wavelet", "curtosis_wavelet", "image_entropy", "class"
]

# Scale numerical features
scaler = StandardScaler()
df[["variance_wavelet", "skewness_wavelet", "curtosis_wavelet", "image_entropy"]] = scaler.fit_transform(
    df[["variance_wavelet", "skewness_wavelet", "curtosis_wavelet", "image_entropy"]]
)
```

## Model Training and Evaluation
The project includes training several ensemble models such as Random Forest, AdaBoost, and Gradient Boosting. It also performs hyperparameter tuning and evaluates the models.

### Code Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into training and testing sets
X = df.drop("class", axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_train, y_train)
ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_ada = ada.predict(X_test)
y_pred_gb = gb.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
```

## Visualization
The project includes various visualizations to analyze the model performance, such as model comparison and hyperparameter tuning heatmaps.

### Code Example:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Model comparison
models = ["Random Forest", "AdaBoost", "Gradient Boosting"]
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_ada), accuracy_score(y_test, y_pred_gb)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

# Glass Type Prediction

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [License](#license)

## Introduction
This project demonstrates the use of ensemble machine learning models to classify glass types based on their chemical composition. The dataset used contains various features representing the chemical elements present in the glass and a class label indicating the type of glass.

## Dataset
The dataset used for this project is the Glass Type Prediction dataset, which contains the following features:
- Refractive Index (RI)
- Sodium (Na)
- Magnesium (Mg)
- Aluminum (Al)
- Silicon (Si)
- Potassium (K)
- Calcium (Ca)
- Barium (Ba)
- Iron (Fe)
- Type (glass type classification)

## Installation
To install the required dependencies, use the following commands:
```sh
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `GlassTypePrediction.ipynb` notebook to execute the code.

## Project Structure
- `GlassTypePrediction.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and visualization.

## Data Preprocessing
The project starts with loading and preprocessing the glass type prediction data.

### Code Example:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("./AMLAss1Datasets/glasstypePrediction.csv")

# Rename columns
df.columns = ["ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe", "type"]

# Scale numerical features
scaler = StandardScaler()
df[["ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe"]] = scaler.fit_transform(
    df[["ri", "na", "mg", "al", "si", "k", "ca", "ba", "fe"]]
)
```

## Model Training and Evaluation
The project includes training several ensemble models such as Random Forest, AdaBoost, and Gradient Boosting. It also performs hyperparameter tuning and evaluates the models.

### Code Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into training and testing sets
X = df.drop("type", axis=1)
y = df["type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_train, y_train)
ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_ada = ada.predict(X_test)
y_pred_gb = gb.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
```

## Visualization
The project includes various visualizations to analyze the model performance, such as model comparison and hyperparameter tuning heatmaps.

### Code Example:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Model comparison
models = ["Random Forest", "AdaBoost", "Gradient Boosting"]
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_ada), accuracy_score(y_test, y_pred_gb)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

# BankLoan Dataset

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [License](#license)

## Introduction
This project demonstrates the use of ensemble machine learning models to predict personal loan acceptance based on various customer attributes. The dataset used contains information about customers and whether they have accepted a personal loan offer.

## Dataset
The dataset used for this project is the BankLoan dataset, which contains the following features:
- Age
- Experience
- Income
- Family
- CCAvg (Average spending on credit cards per month)
- Education
- Mortgage
- Personal Loan (Target variable: 1 if accepted, 0 if not)
- Securities Account (Binary: 1 if the customer has a securities account, 0 otherwise)
- CD Account (Binary: 1 if the customer has a certificate of deposit account, 0 otherwise)
- Online (Binary: 1 if the customer uses online banking, 0 otherwise)
- Credit Card (Binary: 1 if the customer has a credit card issued by the bank, 0 otherwise)

## Installation
To install the required dependencies, use the following commands:
```sh
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `Bankloan_Dataset.ipynb` notebook to execute the code.

## Project Structure
- `Bankloan_Dataset.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and visualization.

## Data Preprocessing
The project starts with loading and preprocessing the bank loan dataset.

### Code Example:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv("./AMLAss1Datasets/bankloan.csv")

# Selecting relevant columns
df = df[
    [
        "Age", "Experience", "Income",
        "Family", "CCAvg", "Education",
        "Mortgage", "Personal.Loan", "Securities.Account",
        "CD.Account", "Online", "CreditCard",
    ]
]

# Renaming the columns
df.columns = [
    "age", "experience", "income", "family",
    "cc_avg", "education", "mortgage", 
    "personal_loan", "securities_account", 
    "cd_account", "online", "credit_card"
]

# Scaling the numerical features
scaler = StandardScaler()
df[["age", "experience", "income", "cc_avg", "mortgage"]] = scaler.fit_transform(
    df[["age", "experience", "income", "cc_avg", "mortgage"]]
)

# Encoding categorical features
label_encoder = LabelEncoder()
df["education"] = label_encoder.fit_transform(df["education"])
```

## Model Training and Evaluation
The project includes training several ensemble models such as Random Forest, AdaBoost, and Gradient Boosting. It also performs hyperparameter tuning and evaluates the models.

### Code Example:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split data into training and testing sets
X = df.drop("personal_loan", axis=1)
y = df["personal_loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_train, y_train)
ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Evaluate models
y_pred_rf = rf.predict(X_test)
y_pred_ada = ada.predict(X_test)
y_pred_gb = gb.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
```

## Visualization
The project includes various visualizations to analyze the model performance, such as model comparison and hyperparameter tuning heatmaps.

### Code Example:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Model comparison
models = ["Random Forest", "AdaBoost", "Gradient Boosting"]
accuracies = [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_ada), accuracy_score(y_test, y_pred_gb)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

### Note
This project was managed and provided by the German International University.

These README files provide comprehensive guides for understanding, installing, and using the `BanknoteAuthentication`, `GlassTypePrediction`, and `Bankloan_Dataset` projects. You can copy and paste this content into `README.md` files for your GitHub repository.
