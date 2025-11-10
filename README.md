# Amazon Beauty Reviews Analysis

## **Project Overview**
This is an **Aspect-Based Sentiment Analysis (ABSA)** project on Amazon's All Beauty category reviews, combining **LDA topic modeling** with **Naive Bayes sentiment classification** to understand customer opinions across different product aspects.

---

## **1. Data Pipeline (PHASES 1-3)**

### **Dataset Loaded**
- 701,528 total reviews spanning from 2000 to 2023
- 631,986 unique users reviewing 112,565 unique products
- 90.5% verified purchases

### **Initial Rating Distribution** (Highly Skewed)
The dataset shows extreme positive bias: 60% are 5-star reviews, while only 15% are 1-star and 6% are 2-star

### **Text Preprocessing**
- After cleaning and filtering short reviews (less than 5 words), 530,546 reviews remained
- Reviews converted to sentiment labels: **Positive (69%)**, **Negative (22%)**, **Neutral (9%)**

---

## **2. Topic Discovery (PHASE 4)**

### **LDA Model Results**
- Optimal number of topics determined to be 15 with a coherence score of 0.5892
- The model identified 8 main product aspects:
  1. Hair Care & Styling
  2. Skin Care & Moisturizers
  3. Makeup & Cosmetics
  4. Nail Care & Polish
  5. Scent & Fragrance
  6. Product Quality & Packaging
  7. Price & Value
  8. Shipping & Delivery


---

## **3. Sentiment Classification (PHASE 5)**

### **The SMOTE Balancing Problem**
The original training data was severely imbalanced: 294K positive, 93K negative, and only 37K neutral reviews

**Solution Applied**: 
- Used SMOTE (Synthetic Minority Over-sampling) + undersampling to rebalance to 150K positive, 100K negative, 80K neutral

### Model Training
#### Step 1: Trained Two Naive Bayes Variants on Balanced Data

- MultinomialNB: Accuracy 81.14%, F1 0.8144
- ComplementNB: Accuracy 79.89%, F1 0.8046
- Winner: MultinomialNB selected as better performer

#### Step 2: Hyperparameter Tuning via Grid Search
Performed 5-fold cross-validation testing 14 parameter combinations:

- Tested alpha values: [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
- Tested fit_prior: [True, False]
- Best parameters found: alpha=0.01, fit_prior=False
- Best CV F1 score: 0.7279

#### Step 3: Final Model Evaluation on Test Set

**Before SMOTE** (trained on imbalanced data):
- Weighted F1: 0.7776 | Accuracy: 81.85%
- Critical problem: Neutral class had only 1.9% recall and 0.037 F1-score - effectively couldn't detect neutral reviews

**After SMOTE** (trained on balanced data):
- Weighted F1: 0.7767 | Accuracy: 73.99%
- Major improvements: Neutral recall jumped from 1.9% to 61.1%, F1 score increased from 0.037 to 0.333
- Trade-off: Positive recall decreased from 98% to 77%, but overall system became more balanced


Per-class breakdown:

|          | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 0.7436    | 0.6813 | 0.7111   | 23,246  |
| Neutral  | 0.2290    | 0.6111 | 0.3331   | 9,319   |
| Positive | 0.9506    | 0.7747 | 0.8537   | 73,545  |

#### Step 4: Generated Comparison Visualizations
Created visual comparisons showing Before/After SMOTE:
- Confusion matrices side-by-side
- F1 score bar charts comparing all three classes
- Saved improvement metrics showing neutral F1 increased by +0.2965

#### Step 5: Model Serialization
Saved to disk:
- `naive_bayes_model_balanced.pkl` (trained classifier)
- `tfidf_vectorizer.pkl` (feature transformer)
- `all_beauty_predictions_balanced.csv` (106,110 test set predictions)
- `smote_improvement_report.txt` (detailed metrics)


---

## **4. Aspect-Based Insights (PHASES 6-7)**


### Step 1: Loaded Predictions & Analyzed Aspect-Sentiment Relationships

- Merged predictions with topic labels from Phase 4
- Generated cross-tabulation of topics × sentiments
- Calculated sentiment distribution percentages per aspect

### Step 2: Identified Strengths & Weaknesses
Created aspect rankings based on "Net Sentiment" (% positive - % negative):
- Strongest: Nail Care (+84%), Product Quality (+78%)
- Weakest: Shipping & Delivery (-98%), Skin Care (+8%)

### Step 3: Deep Dive Analysis per Aspect
For each of the 8 aspects, extracted:
- Total review counts
- Sentiment breakdowns with percentages
- Average ratings
- Example reviews for positive/negative/neutral cases (with confidence scores)

### Step 4: Generated Word Clouds
Created sentiment-specific word clouds for top/bottom aspects:
- Separate clouds for positive, neutral, negative within each aspect
- Color-coded: Green (positive), Blue (neutral), Red (negative)
- Files saved: wordcloud_nail_care_and_polish.png, etc.

### Step 5: Temporal Trend Analysis
- Converted timestamps to year-month periods
- Plotted sentiment trends over time (overall + per-aspect)
- Generated line charts showing how sentiment evolved from 2000-2023


### Results
#### **Best Performing Aspects** (Customer Strengths)
1. **Nail Care & Polish**: 89.7% positive, 5.7% negative (738 reviews)
2. **Product Quality & Packaging**: 83.7% positive, 5.2% negative (32,543 reviews)
3. **Scent & Fragrance**: 61.9% positive, 27.2% negative (17,184 reviews)

#### **Worst Performing Aspects** (Areas for Improvement)
1. **Shipping & Delivery**: Only 0.6% positive, 98.5% negative (328 reviews) - catastrophic failure
2. **Skin Care & Moisturizers**: 38.9% positive, 31.0% negative
3. **Hair Care & Styling**: 36.0% positive, 26.9% negative (43,734 reviews - largest category)

### **Critical Findings**

**Shipping & Delivery Crisis**: The aspect has an average rating of only 1.35/5, with negative examples showing issues like damaged products on arrival, non-functional items, and units shedding/breaking immediately

**Hair Care Neutrality**: Hair Care & Styling (the largest category with 43,734 reviews) shows massive ambivalence - 37.1% neutral reviews with only 3.42/5 average rating

---

## **5. What These Results Mean for Business**

### **Actionable Insights**
1. **Immediate Crisis**: Address shipping/packaging issues causing 98.5% negative sentiment in delivery-related reviews
2. **Low-Hanging Fruit**: Nail polish and product packaging are already strong - amplify these in marketing
3. **Strategic Focus**: Hair care (the largest category) needs product improvements - customers are lukewarm
4. **Quality Perception Gap**: Despite 83% positive sentiment on packaging quality, shipping failures undermine this strength

### **Model Limitations**
- The neutral class (9% of data) remains challenging with only 33% F1-score
- SMOTE created synthetic training data, which may not perfectly represent real neutral sentiments
- Some topic assignments show ambiguity (e.g., "Hair Care & Styling" catching diverse beauty products)

### **Technical Achievement**
The system successfully moved from a "dumb positive predictor" (98% recall on positive, ignoring neutral) to a **balanced multi-class classifier** that can detect nuanced opinions across 8 product aspects - enabling targeted business decisions rather than generic sentiment scores.


---
