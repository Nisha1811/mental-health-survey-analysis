

# Mental Health in Tech – Survey Analysis & Power BI Dashboard

This project is of data analytics project that analyzes **mental health perceptions in the tech industry** using a public survey dataset.  
It includes:

- Data cleaning and exploratory data analysis (EDA) in **Python**
- A simple **machine learning model** to predict perceived consequences
- An interactive **Power BI dashboard** for visual insights

***

## 1. Project Overview

Mental health is a critical issue in the tech and student community, with rising levels of stress, anxiety, and depression. However, stigma and fear of negative consequences at work often prevent people from seeking help.

This project uses a real survey dataset to:

- Understand **who** the respondents are (age, gender, family history, etc.)
- Analyze how they **perceive the consequences** of having a mental health issue at work
- Explore how factors like **gender**, **family history**, **benefits**, and **work interference** are related to mental health perceptions
- Present the findings through a **Python notebook** and a **Power BI dashboard**

***

## 2. Objectives

- Perform **data cleaning** and **EDA** on a mental health survey dataset.
- Analyze the distribution of key variables like age, gender, family history, benefits, and work interference.
- Study relationships such as:
  - `Gender` vs `mental_health_consequence`
  - `family_history` vs `mental_health_consequence`
  - `work_interfere` vs mental health perceptions
- Build a **basic classification model** to predict perceived consequences.
- Design a **Power BI dashboard** summarizing key findings in an interactive way.

***

## 3. Dataset

- **Source:** Kaggle – *Mental Health in Tech Survey* (OSMI)  
  Example: `mental-health-in-tech-2016` (OSMI survey)
 
- **Format:** CSV  
- **Examples of key columns:**
  - `Age` – Age of respondent
  - `Gender` – Raw gender text
  - `Gender_clean` – Cleaned gender (Male / Female / Other)
  - `family_history` – Family history of mental illness (Yes/No)
  - `mental_health_consequence` – Perceived consequences if mental health issues are discussed at work (Yes/No/Maybe)
  - `work_interfere` – How often mental health interferes with work
  - `benefits` – Whether employer provides mental health benefits
  - `remote_work`, `care_options`, etc. (if present in the version used)

> Note: The dataset is not owned by me; it is used only for educational and academic purposes.

***

## 4. Tools & Technologies

- **Programming Language:** Python
- **Libraries:**
  - `pandas` – data manipulation
  - `numpy` – numerical operations
  - `matplotlib`, `seaborn` – visualizations
  - `scikit-learn` – machine learning (Logistic Regression, train-test split, etc.)
- **Environment:** Google Colab
- **Dashboard:** Power BI Desktop

***

## 5. Project Structure

```text
mental-health-survey-analysis/
│
├── Mental_health_da.ipynb        # Main analysis notebook (Python, Colab)
├── mental_health_dashboard.pbix  # Power BI dashboard file (optional, if included)
├── mental_health_dashboard.pdf   # Exported dashboard for quick view
├── data/
│   └── mental-health-in-tech.csv # Survey dataset (OR instructions in README if not shared)
└── README.md                     # Project documentation
```

> If you cannot share the raw dataset due to license, keep a `data/README.md` with the Kaggle link and usage instructions.

***

## 6. Methodology / Workflow

1. **Data Loading**
   - Load the CSV file in Google Colab using `pandas.read_csv`.
2. **Initial Inspection**
   - View shape, columns, head, missing values, and data types.
3. **Data Cleaning**
   - Filter out unrealistic ages (e.g., keep 15–70 years).
   - Clean and standardize `Gender` into `Gender_clean`:
     - Male / Female / Other
   - Drop columns with >50% missing values.
   - Fill remaining missing values:
     - Categorical → `"Unknown"`
     - Numeric → median values
4. **Exploratory Data Analysis (EDA)**
   - Univariate analysis:
     - Age distribution (histogram)
     - Gender distribution (bar chart)
     - Family history counts
   - Mental health consequence analysis:
     - Distribution of `mental_health_consequence` responses
   - Bivariate analysis:
     - `Gender_clean` vs `mental_health_consequence` (crosstab + bar chart)
     - `family_history` vs `mental_health_consequence`
     - `work_interfere` vs counts
5. **Basic Machine Learning**
   - Label encode categorical features (Gender_clean, family_history, mental_health_consequence).
   - Select features: `Age`, `Gender_clean`, `family_history` (and others if needed).
   - Train/test split.
   - Train a **Logistic Regression** model.
   - Evaluate using:
     - Classification report (accuracy, precision, recall, F1-score)
     - Confusion matrix
6. **Dashboard Design (Power BI)**
   - Import the cleaned CSV into Power BI.
   - Create interactive visuals (see next section).
   - Export dashboard as PDF for sharing.

***

## 7. Key Python Steps (Summary)

Inside `Mental_health_da.ipynb`, the main steps are:

### 7.1 Data Loading

```python
import pandas as pd

df = pd.read_csv('mental-health-in-tech-2016.csv')
print(df.shape)
print(df.columns)
```

### 7.2 Cleaning Age and Gender

```python
# Filter realistic ages
df = df[(df['Age'] >= 15) & (df['Age'] <= 70)]

# Clean gender
def clean_gender(g):
    if pd.isnull(g):
        return 'Other'
    g = str(g).strip().lower()
    if 'male' in g or g in ['m', 'man']:
        return 'Male'
    if 'female' in g or g in ['f', 'woman']:
        return 'Female'
    return 'Other'

df['Gender_clean'] = df['Gender'].apply(clean_gender)
```

### 7.3 Handling Missing Values

```python
missing_pct = df.isnull().mean()

cols_many_missing = missing_pct[missing_pct > 0.5].index
df = df.drop(columns=cols_many_missing)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
```

### 7.4 EDA Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Age distribution
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Age Distribution of Respondents')
plt.show()

# Gender distribution
df['Gender_clean'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.show()

# Mental health consequence distribution
col_mh = 'mental_health_consequence'
df[col_mh].value_counts().plot(kind='bar')
plt.title('Perceived Mental Health Consequences')
plt.show()
```

```python
# Gender vs mental_health_consequence
ct_gender = pd.crosstab(df['Gender_clean'], df[col_mh], normalize='index') * 100
print(ct_gender)
ct_gender.plot(kind='bar')
plt.title('Perceived Consequences by Gender (%)')
plt.show()
```

***

## 8. Machine Learning Model (Summary)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

feature_cols = ['Age', 'Gender_clean', 'family_history']
data_ml = df[feature_cols + ['mental_health_consequence']].copy()

# Label encode
le_dict = {}
for col in data_ml.columns:
    if data_ml[col].dtype == 'object':
        le = LabelEncoder()
        data_ml[col] = le.fit_transform(data_ml[col])
        le_dict[col] = le

X = data_ml[feature_cols]
y = data_ml['mental_health_consequence']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

The model gives a basic idea of how features like age, gender, and family history relate to perceived mental health consequences.

***

## 9. Power BI Dashboard

The Power BI dashboard includes:

- **Cards**
  - Total number of respondents
  - Count (or % ) of respondents with family history = “Yes”

- **Bar Charts**
  - Gender distribution (`Gender_clean`)
  - Distribution of `mental_health_consequence`
  - `Gender_clean` vs `mental_health_consequence`
  - `family_history` vs `mental_health_consequence`
  - `work_interfere` distribution

- **Pie/Donut Chart**
  - Employer mental health `benefits` (Yes/No/Don’t know)

- **Line Chart**
  - `AgeGroup` vs rate of “Yes” for `mental_health_consequence`

- **Slicers**
  - `Gender_clean`
  - `family_history`
  - Optional: `remote_work`, country, etc.

This dashboard helps quickly understand:

- Who the respondents are
- How they perceive the consequences of mental health issues
- How factors like gender, family history, and work interference are related

***

## 10. Results & Insights (Example)

Some sample insights you might derive (replace with your actual findings):

- The majority of respondents are in the **25–34 age group** and identify as **Male**.
- Most respondents are unsure or do not clearly expect negative consequences at work (“Maybe” or “No” responses dominate).
- Respondents with a **family history** of mental illness show a **slightly higher rate** of expecting negative consequences compared to those without family history.
- A significant number report that mental health issues **sometimes or often interfere** with work.
- Awareness of employer-provided **mental health benefits** is not uniform; many are unsure if such benefits exist.

***

## 11. How to Run This Project

### Option 1 – Google Colab

1. Clone this repository:
   ```bash
   git clone <your_repo_link>.git
   ```
2. Upload `Mental_health_da.ipynb` and the dataset CSV to Google Colab.
3. Open the notebook in Colab and run all cells step by step.

### Option 2 – Local Jupyter Notebook

1. Clone the repo and install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `Mental_health_da.ipynb` and execute cells.

***

## 12. Future Work

- Use more advanced models (Random Forest, XGBoost) and compare performance.
- Include additional features like `remote_work`, `benefits`, `work_interfere` in the ML model.
- Extend analysis to multiple years of the OSMI survey.
- Deploy an interactive web app (Streamlit / Flask) for mental health exploration.

***

## 13. Acknowledgments

- **Dataset:** Open Sourcing Mental Illness (OSMI) – Mental Health in Tech Survey (via Kaggle).
