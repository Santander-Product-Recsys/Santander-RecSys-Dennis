# Santander Product Recommendation System

## Project Overview
A comprehensive machine learning project to build a product recommendation system for Santander Bank, predicting which financial products customers are most likely to add in the next month. This project combines extensive exploratory data analysis, customer segmentation insights, and multiple recommendation algorithms to understand banking product adoption patterns.

## Business Context
Santander Bank seeks to improve customer experience and increase product adoption by providing personalized product recommendations. The challenge involves predicting up to 7 products each customer is likely to add from a catalog of 24 financial products including accounts, loans, cards, and insurance products.

## Project Structure
```
SANTANDER-RECSYS-DENNIS/
├── cf_submission_batch.csv
├── EDA.ipynb
├── ErrorAnalysis.ipynb
├── hybrid_alpha005.csv
├── mf_hybrid_alpha05.csv
├── Modelling.ipynb
├── data/
│   ├── cleaned_santander_data.csv
│   ├── kaggle_collaborative_filtering_submission.csv
│   ├── kaggle_submission.csv
│   ├── product_diff.csv
│   ├── test_ver2.csv
│   └── train_ver2.csv
├── Plots/
├── .gitignore
├── notes.md
├── Readme.md
├── ModellingDocumentation.md
├── EDADocumentation.md
└── santander-product-recommendation.zip
```

## Documentation Structure
- **README.md** (this file): Project overview and structure
- **[ModellingDocumentation.md](./ModellingDocumentation.md)**: Detailed modeling approaches, algorithms, and performance metrics
- **[EDADocumentation.md](./EDADocumentation.md)**: Comprehensive exploratory data analysis and insights

## Key Components

### 1. Data Analysis (EDA.ipynb)
- Customer segmentation analysis
- Product ownership patterns
- Demographic insights
- Provincial distribution analysis
- Temporal patterns in product adoption

### 2. Modeling Pipeline (Modelling.ipynb)
- Baseline popularity model
- Collaborative filtering approaches
- Matrix factorization techniques
- Hybrid recommendation systems
- Performance evaluation and optimization

### 3. Error Analysis (ErrorAnalysis.ipynb)
- Model failure analysis
- False positive/negative patterns
- Improvement strategies

## Dataset Characteristics
- **Source**: [Santander Product Recommendation (Kaggle Competition)](https://www.kaggle.com/c/santander-product-recommendation)
- **Timeline**: 17 months of data (Jan 2015 - May 2016)
- **Scale**: ~945K unique customers, 24 financial products
- **Sparsity**: 6.46% density in user-product interactions
- **Evaluation Metric**: MAP@7 (Mean Average Precision at 7)

## Key Findings

### Customer Insights
- Three main customer segments: Regular individuals (majority), Students, and VIPs
- Clear age and gender patterns across segments
- Product adoption follows lifecycle patterns rather than preference-based patterns

### Modeling Results
- Simple popularity baseline outperforms complex collaborative filtering
- Banking domain characteristics make traditional recommendation approaches less effective
- External factors (life events, eligibility) drive product adoption more than user similarities

| Model | MAP@7 (Local) | Kaggle Score |
|-------|--------------|--------------|
| Popularity Baseline | 0.5937 | 0.017 |
| Hybrid (MF + Popularity) | 0.5927 | TBD |
| Hybrid (CF + Popularity) | 0.5916 | 0.016 |

## Technologies Used
- **Data Processing**: Python, Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib
- **Machine Learning**: Scikit-learn, SciPy
- **Development Environment**: Jupyter Notebook

## Key Learnings
1. Domain understanding is crucial - banking products don't follow typical e-commerce patterns
2. Simple baselines can outperform complex models in regulated, lifecycle-driven domains
3. Sparsity and external factors limit collaborative filtering effectiveness in financial services
4. Customer segmentation reveals opportunities for targeted approaches

## Future Directions
- Implement segment-specific models for different customer groups
- Incorporate temporal features and seasonality
- Explore sequence-based models for product lifecycle
- Add business rules for regulatory compliance

## Author
**Dennis Mathew Jose**  
MS Data Analytics Engineering, Northeastern University  
[LinkedIn](https://www.linkedin.com/in/dennismjose/)

## Acknowledgments
- Santander Bank for providing the dataset through Kaggle
- Kaggle community for insights and discussions
