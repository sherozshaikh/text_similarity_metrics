## 📝 Textual Similarity Assessment Suite

Textual Similarity Assessment Suite is a Python class designed for computing various text similarity metrics between elements in pandas dataframe. It supports a wide range of metrics and provides functionalities for preprocessing, computing similarity scores, and exporting results.

### Overview

Welcome to the Textual Similarity Assessment Suite repository! This project provides a robust toolkit for evaluating and comparing text similarity using advanced Natural Language Processing (NLP) techniques. It enables users to compute a variety of similarity metrics between textual elements extracted from datasets, facilitating detailed analysis and assessment of text similarity.

This suite supports a wide range of functionalities, including preprocessing of text for normalization, and computation of metrics such as Jaccard similarity, Levenshtein distance, Dice coefficient, and more. Users can customize metric selection, analyze results, and export findings in CSV and ZIP formats for further insights and applications.

Explore the capabilities of the Textual Similarity Assessment Suite to enhance your text analysis workflows and gain deeper insights into textual similarity relationships.

---

### ✨ Key Features

- Compute similarity metrics such as Jaccard similarity, Levenshtein distance, Dice coefficient, and more.
- Flexible metric selection and output customization.
- Preprocessing utilities for text normalization and character trimming.
- Export results to CSV and ZIP formats.

---

### 📦 Requirements

- Python 3.x
- pandas
- nltk
- jellyfish
- fuzzywuzzy
- numpy
- scipy
- pandarallel (optional for parallel processing)

---

### 💻 Installation

```bash
pip install --quiet numpy pandas fuzzywuzzy jellyfish python-Levenshtein pandarallel
```

---

### 🚀 Usage

#### Using `text_scoring.py`

```python
import pandas as pd
from text_scoring import TextScoring

# Example usage with DataFrame
df = pd.read_csv('your_data.csv')

# Create an instance of TextScoring
text_scorer = TextScoring(
    dataframe_object=df,
    output_folder='Mapped_Attributes',
    col_name_1='doc1_elements',
    col_name_2='doc2_elements',
    metrics_list=['all']
)

# Perform similarity scoring on the DataFrame
results_df = text_scorer.main()

# Save results to CSV
results_df.to_csv('similarity_scores.csv', index=False)
```

---

```python
import pandas as pd
from text_scoring import TextScoring

# Sample DataFrame
df = pd.DataFrame({
    'doc1_elements': ['apple', 'banana', 'cherry'],
    'doc2_elements': ['apple', 'banana', 'date']
})

# Create TextScoring instance
text_scorer = TextScoring(
    dataframe_object=df,
    output_folder='Example1',
    col_name_1='doc1_elements',
    col_name_2='doc2_elements',
    metrics_list=['get_jaccard_similarity', 'get_editdistance']
)

# Compute similarity scores
results_df = text_scorer.main()

# Save results to CSV
results_df.to_csv('similarity_scores_example1.csv', index=False)
```

---

#### Using `text_scoring.ipynb`

For a detailed demonstration and interactive usage, refer to the [Textual-Similarity-Assessment-Suite-Notebook](text_scoring.ipynb) Jupyter Notebook. It provides examples of how to use the Textual Similarity Assessment Suite with example.
