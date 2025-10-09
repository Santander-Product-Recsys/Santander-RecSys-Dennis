# Santander Product Recommendation System

## Project Overview
Building a recommendation system for Santander Bank to predict which products customers will add in the next month. The challenge involves predicting up to 7 products for ~930K customers from a catalog of 24 financial products.

## Dataset Characteristics
- **Training Data**: 17 months of customer-product relationships (Jan 2015 - May 2016)
- **Test Data**: Predict products for June 2016
- **Users**: ~945K unique customers
- **Products**: 24 financial products (accounts, cards, loans, insurance, etc.)
- **Sparsity**: 6.46% (very sparse interaction matrix)
- **Evaluation Metric**: MAP@7 (Mean Average Precision at 7)

## Approaches Implemented & Results

### 1. Baseline Model: Popularity-Based Recommender
**Design**: Recommend the same top 7 most popular products to all users

**Implementation**:
```python
# Count product frequencies across all users
# Sort by popularity and select top 7
# Recommend these 7 products to everyone
```

**Performance**:
- Local Validation MAP@7: **0.5937**
- Kaggle Public Score: **0.017**
- Status: Strong baseline established

**Key Insights**:
- Simple approach performs surprisingly well
- Most popular products: ind_cco_fin_ult1, ind_ctop_fin_ult1, ind_recibo_ult1
- High false positives (~880K) for saturated products
- Misses specialized products (investment, credit cards)

---

### 2. Item-Based Collaborative Filtering
**Design**: Find similar products based on co-occurrence patterns using cosine similarity

**Implementation**:
```python
# Build user-product interaction matrix
# Calculate cosine similarity between products
# For each user, aggregate similarities from owned products
# Recommend top 7 products with highest scores
```

**Performance**:
- Local Validation MAP@7: **0.0898** (-84.87% vs baseline)
- Kaggle Public Score: **0.014** (-17.6% vs baseline)
- Status: Failed - performed worse than baseline

**Issues Identified**:
- Similarity matrix too sparse/noisy
- Poor cold start handling
- Banking products don't follow typical co-purchase patterns

---

### 3. Hybrid Model v1: Item-CF + Popularity
**Design**: Weighted combination of collaborative filtering and popularity scores

**Formula**: `score = α × CF_score + (1-α) × popularity_score`

**Initial Bug**: Model was excluding already-owned products, causing poor performance

**After Fix - Performance**:
| Alpha | Local MAP@7 | vs Baseline | Kaggle Score |
|-------|------------|-------------|--------------|
| 0.00 | 0.5937 | 0.00% | - |
| 0.05 | 0.5916 | -0.35% | 0.016 |
| 0.10 | 0.5908 | -0.49% | - |
| 0.15 | 0.5897 | -0.67% | - |
| 0.20 | 0.5870 | -1.13% | - |
| 0.30 | 0.5851 | -1.45% | - |

**Status**: No improvement over baseline

---

### 4. Matrix Factorization (SVD)
**Design**: Decompose user-product matrix into latent factors using Singular Value Decomposition

**Implementation**:
```python
# Decompose sparse matrix into user_factors × product_factors
# Use 15 latent factors
# Score = user_vector · product_vector
```

**Performance**:
- Pure MF MAP@7: **0.3783** (-36.27% vs baseline)
- Status: Better than item-CF but still underperforms

**Key Finding**: MF captures some patterns but not as predictive as simple popularity

---

### 5. Hybrid Model v2: Matrix Factorization + Popularity
**Design**: Combine MF scores with popularity scores

**Performance**:
| Alpha | Local MAP@7 | vs Baseline |
|-------|------------|-------------|
| 0.05 | 0.5927 | -0.17% |
| 0.10 | 0.5910 | -0.45% |
| 0.15 | 0.5884 | -0.89% |
| 0.20 | 0.5878 | -1.00% |
| 0.30 | 0.5877 | -1.01% |

**Best Configuration**: α=0.05 (5% MF, 95% popularity)
- Status: Slight degradation from baseline
- Kaggle Score: Pending submission

---

## Summary of Results

| Rank | Model | Local MAP@7 | Kaggle Score | Status |
|------|-------|-------------|--------------|--------|
| 1 | Popularity Baseline | 0.5937 | 0.017 | Best |
| 2 | MF + Popularity (α=0.05) | 0.5927 | TBD | Testing |
| 3 | Item-CF + Popularity (α=0.05) | 0.5916 | 0.016 | Tested |
| 4 | Pure Matrix Factorization | 0.3783 | - | Failed |
| 5 | Pure Item-Based CF | 0.0898 | 0.014 | Failed |

## Key Learnings

### Why Collaborative Filtering Fails in Banking
1. **Product Lifecycle**: Banking products follow sequential patterns (checking → savings → credit → investment) rather than co-purchase patterns
2. **External Drivers**: Product adoption driven by life events, eligibility, and regulations rather than preferences
3. **Low Frequency**: Users rarely add new products (unlike e-commerce or entertainment)
4. **Sparse Data**: 6.46% density insufficient for reliable similarity patterns

### What Works
- Simple popularity captures the strongest signal
- Banking has universal product needs across customers
- Temporal patterns likely more important than user similarities

## Next Steps

### Immediate Actions
1. Submit MF hybrid (α=0.05) to Kaggle for final validation
2. If no improvement, proceed with alternative approaches

### Promising Directions
1. **Segment-Based Models**
   - Build separate models for age groups, tenure segments
   - Recognize different lifecycle stages have different needs

2. **Temporal Features**
   - Incorporate seasonality and trends
   - Model product adoption sequences

3. **Customer Features**
   - Use demographics (age, income, geography)
   - Build content-based filtering approach

4. **Rule-Based Enhancement**
   - Business rules for product eligibility
   - Sequential product recommendations

