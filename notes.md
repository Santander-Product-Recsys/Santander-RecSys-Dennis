# Customer Demographic Insights

## **Customer Distribution**

This bar chart shows how customers are distributed across the bank's three main segments:

### 02 - PARTICULARES (Regular Individuals)
- Represents 58.33% of the total customers — the largest group.
- These are everyday banking users, likely salaried or general population.

### 03 - UNIVERSITARIO (Students)
- Comprises 36.17% of the customer base.
- This group likely includes younger individuals still studying or early in their financial journey.

### 01 - TOP (Premium/High Net-Worth Clients)
- Only 4.12% of customers belong to this high-value segment.
- Despite the small number, they are likely to contribute disproportionately to revenue.

 **Insight**:  
The bank primarily serves regular and student clients. Premium customers are few in number but potentially more profitable.

---

## **Gender Breakdown by Segment**

This chart highlights how gender is distributed within each segment:

### PARTICULARES (Regular Individuals)
- Females slightly outnumber males.
- Indicates strong engagement of women in general banking services.

### UNIVERSITARIO (Students)
- Males outnumber females.
- Opportunity for the bank to understand and target why fewer female students are active.

### TOP (Premium Clients)
- Females again slightly outnumber males.
- Suggests room to develop premium offerings focused on women clients.

 **Insight**:  
There are subtle gender differences across segments. Understanding these can aid in designing more inclusive and personalized banking products.

---

## **Age Distribution (Violin Plot)**

**Purpose**: To visualize how age is distributed within each customer segment, separated by gender.

### 1. UNIVERSITARIO (Students)
- Ages are tightly packed between 20 and 25.
- Distribution is symmetric and indicates a homogeneous age group.
- Males show a slightly higher density.
- Few outliers — minimal data concerns.

**Insight**:  
Product design should cater to student needs — digital tools, savings, and education loans.

---

### 2. PARTICULARES (Regular Individuals)
- Age spans from early 20s to 90+.
- Peak is in early 40s; distribution is slightly right-skewed.
- Females show broader spread into older ages.

**Insight**:  
This diverse group benefits from varied services — from home loans to retirement savings. Age-aware marketing can be useful.

---

### 3. TOP (Premium Clients)
- Concentrated in 40–60 age range.
- Symmetric distribution, with some clients even 100+.
- Females show slightly broader spread.

**Insight**:  
This group is financially mature. Suitable offerings include wealth management, estate planning, and tailored advisory services.

---

## General Observations

- **Outliers**: Customers over 120 exist in some segments and may require data validation.
- **Gender Trends**: Females show wider age ranges, particularly at older ages.
- **Business Relevance**: Age and gender profiles should guide customer segmentation, product design, and communication strategies.

---


## Income Distribution by Segment and Gender – Detailed Analysis

### Overview:
The violin plot visualizes income (`renta`) distribution across three customer segments: 
- **Individuals**
- **College Students**
- **VIPs**, split by gender (H = Male, V = Female).

### Key Observations:

1. **Overall Distribution Shape**:
   - The plots are wider at the bottom and narrow at the top, confirming a **right-skewed** distribution.
   - Most customers earn within a lower to mid-range income bracket, with fewer high-income individuals.

2. **Segment-wise Breakdown**:
   - **College Students**:
     - Both male and female students have the narrowest income range and lowest median incomes.
     - Expected due to limited earning capacity in this group.
   
   - **Individuals**:
     - Wider income spread than students.
     - The median and interquartile ranges (dotted lines inside the violin) are broader.
     - Indicates this segment includes salaried employees, small business owners, etc.

   - **VIP Customers**:
     - This group has **the highest median income**.
     - Wider tails on the violins suggest a significant number of high-income outliers.
     - Strong target group for investment, credit, and premium services.

3. **Gender Comparison**:
   - No major differences in income distribution between males (H) and females (V) across all segments.
   - However, the distributions are almost symmetric, showing **gender equity in income spread** for each segment.

### Insights for Strategy:
- Financial products for students should focus on budgeting, education loans, and savings tools.
- VIPs represent a high-value group and require premium services (e.g., wealth management).
- Marketing and product design don’t need drastic gender customization in terms of income.

## Income Distribution by Segment, Gender, and Activity Status – Detailed Analysis

### Context:
This violin plot displays the income distribution (`renta`) segmented by:
- Customer Segment (`Individuals`, `College Students`, `VIP`)
- Gender (`H` = Male, `V` = Female)
- Activity Status (`0.0` = Inactive, `1.0` = Active)

The plot is split into two panels:
- Left: Inactive customers
- Right: Active customers

---

### Key Insights:

#### 1. **VIP Customers Show the Highest Income Spread**
- Both active and inactive VIPs show wide distributions, indicating a mix of medium- to high-income earners.
- VIPs have the highest median income lines and thicker tails, especially among active customers.

#### 2. **College Students Remain at the Lowest Income Levels**
- Narrowest violins across both genders and activity statuses.
- Consistently low median and quartile values.
- Suggests very limited income across this group, which aligns with expectations.

#### 3. **Individuals Sit in the Middle**
- Income distributions are wider than students but more concentrated than VIPs.
- Active individuals have slightly higher and more stable incomes than inactive ones.

#### 4. **Gender Disparity is Minimal**
- Within each group, males and females show similar income ranges.
- No major divergence in distribution patterns or median values.

#### 5. **Active Customers (Status = 1.0) Have More Balanced Distributions**
- Across all segments, the active customers tend to have:
  - Slightly higher medians.
  - Narrower tails, especially among college students and individuals.
  - More uniform and centralized distributions.

#### 6. **Inactive Customers (Status = 0.0) Show More Skewness**
- More outliers and spread, especially in the VIP category.
- Suggests income alone doesn’t guarantee continued activity — possibly impacted by engagement, product usage, or service satisfaction.

---

### Strategic Implications:

- **Retention efforts** should focus on high-income inactive customers, especially VIPs.
- **Targeted financial products** for active individuals can be designed around steady mid-tier incomes.
- **Custom engagement plans** for college students are needed — perhaps focused on future potential rather than current income.
- **Gender-based targeting** may not be necessary purely based on income distribution, since disparity is minimal.


## Customer Segment vs. Products Owned – Detailed Analysis

This grouped bar chart illustrates how product ownership is distributed across three customer segments:

- **01 - TOP** (Premium Clients)
- **02 - PARTICULARES** (Regular Individuals)
- **03 - UNIVERSITARIO** (College Students)

---

### Key Observations:

#### 1. **Dominance of Current Accounts**
- The highest ownership is seen in **Current Accounts**, with over 8 million combined owners.
- Regular customers (`PARTICULARES`) form the majority, followed by a significant number of students (`UNIVERSITARIO`).
- Even high-value clients (`TOP`) have a presence but are fewer in count.

#### 2. **Direct Debit, Payroll, and Pension Income**
- These are the next most common services.
- Their ownership is skewed heavily toward `PARTICULARES`, suggesting income and payment-related services are more relevant to working individuals than students or VIPs.

#### 3. **Low Engagement from 'TOP' Clients**
- The `TOP` segment has the smallest visual footprint across all products.
- Their ownership is mainly seen in premium offerings like:
  - **Long-term Deposits**
  - **e-Accounts**
  - **Credit Cards**
- This implies high-value clients might own fewer products but of higher value or complexity.

#### 4. **Student-Oriented Services**
- `UNIVERSITARIO` customers own:
  - **Junior Accounts**
  - **Current Accounts**
  - **e-Accounts**
- These accounts are likely tailored to low-income or entry-level customers.

#### 5. **Broad Engagement by Regular Individuals**
- The `PARTICULARES` segment leads across almost all product categories.
- This reflects their central role in the bank’s product ecosystem.
- Categories like:
  - **Payroll Account**
  - **Particular Plus Account**
  - **Credit Cards**
  - **Taxes**
  - **Pensions**
  - are heavily used by this group.

---

### Strategic Implications:

- The bank’s **product engagement is highly skewed** toward regular individuals, making them a critical demographic.
- **Cross-selling opportunities** exist for students and `TOP` clients who show low product diversity.
- Special attention can be given to students nearing graduation to **transition into higher-tier products**.
- Further segmentation within `PARTICULARES` may uncover **micro-targeting opportunities** based on product usage patterns.

## Monthly Customer Acquisition by Segment

This time series analysis reveals how new customers were acquired by the bank over time for each segment: 

### 01 - TOP (Premium Clients)
- Acquisition remained relatively low throughout the observed period (1995–2015), peaking below 450 customers per month.
- The trend exhibits minor fluctuations with a few spikes, but the overall customer growth is modest.
- This is expected due to the niche, selective nature of premium clients.

### 02 - PARTICULARES (Regular Individuals)
- Displays a strong and consistent growth trend from 1995, with a sharp surge around 2000–2002.
- Peak acquisition reached ~6000 new customers in **March 2001**, followed by sustained growth through the mid-2000s.
- From 2009 onwards, the trend is more volatile with occasional spikes but a generally downward pattern post-2014.
- The high volume and early growth indicate this segment formed the foundation of the bank’s customer base.

### 03 - UNIVERSITARIO (College Students)
- Virtually no activity is seen before 2011, followed by an explosive acquisition phase between **2011 and 2015**.
- Monthly new customer count frequently exceeds 10,000, peaking above 20,000 in certain months.
- This abrupt increase suggests a strategic campaign or partnership with academic institutions during this period.
- However, the trend also shows high volatility, likely indicating periodic, non-continuous recruitment drives.

### Overall Insights:
- The PARTICULARES and UNIVERSITARIO segments were the key drivers of customer growth.
- The TOP segment remains consistently low, aligning with its likely selective onboarding process.
- The student segment’s sudden rise hints at external influences or focused marketing efforts in that window.

## Monthly New Product Additions (Top 10)

This chart tracks the number of **new additions per month** for the top 10 financial products between January 2015 and mid-2016.

### Key Insights:

#### 1. **High Volume Products**
- `ind_recibo_ult1` (Bill Payment): 
  - Most consistently popular product across all months.
  - Peaks near **12,000 new additions**, showing sustained demand for automated bill payment services.
  
#### 2. **Spiky Products**
- `ind_nomina_ult1` (Salary Deposit) and `ind_nom_pens_ult1` (Pension Deposit):
  - These show **strong periodic spikes**, especially in **March/April 2016**.
  - Possibly tied to payroll or pension onboarding cycles.
  - These trends may reflect employment cycles or promotional campaigns.

#### 3. **Moderate Trend Products**
- `ind_tjcr_fin_ult1` (Credit Card) and `ind_ecue_fin_ult1` (E-Account):
  - Stable, consistent growth with additions between 4,000–5,000 monthly.
  - Reflects steady demand for digital and credit services.

#### 4. **Low Volume Products**
- Products like `ind_ctop_fin_ult1`, `ind_dela_fin_ult1`, `ind_reca_fin_ult1` show minimal movement (mostly under 2,000).
  - Might be niche services or not actively promoted.

### Anomalies or Outliers:
- A clear surge in `ind_nomina_ult1` and `ind_nom_pens_ult1` around **March–April 2016**—this could be investigated further for any policy changes or incentive programs.

### Business Takeaway:
- Products supporting recurring financial activity (bills, salary, pensions) dominate monthly new acquisitions.
- Strategic marketing around employment-related services could capitalize on these naturally high-acquisition periods.

## Top 15 Provinces by Customer Count

This horizontal bar chart visualizes the **number of unique customers** in each of the top 15 Spanish provinces.

### Key Observations:

#### 1. **Madrid’s Dominance**
- **Madrid** leads by a wide margin with **over 300,000 customers**.
- This could be attributed to its role as the financial and administrative capital of Spain.
- Suggests centralization of customer base in urban and economically active areas.

#### 2. **Barcelona and Valencia Follow**
- **Barcelona** is a distant second with fewer than 100,000 customers.
- **Valencia** ranks third, with around 45,000–50,000 customers.
- These are major cities with diverse populations and active economic sectors.

#### 3. **Steady Decline in Other Provinces**
- The drop-off in customer counts is steep after the top three.
- Provinces like **Sevilla**, **Coruña**, and **Murcia** range between 25,000 to 40,000.
- This indicates a long tail of customers spread thinly across smaller regions.

#### 4. **Implication for Strategy**
- Marketing and customer acquisition efforts should prioritize **Madrid, Barcelona, and Valencia** for maximum ROI.
- However, opportunities exist in underpenetrated provinces like **Zaragoza, Asturias, and Valladolid** if localized strategies are adopted.

#### 5. **Skewed Distribution**
- The shape of the bar plot emphasizes a **right-skewed distribution**, with a few provinces holding the majority of the bank’s customer base.

### Business Insight:
The bank’s services and campaigns may be more efficiently delivered with **regional customization**, especially focusing on metropolitan hubs where customer density is high. Meanwhile, regions with lower customer counts might benefit from digital-first approaches due to the cost of physical outreach.

## Treemap: Product Ownership by Province

This treemap visualizes the distribution of **product ownership** by **province**. Each rectangle represents a **product type**, nested inside its corresponding **province**. The **size of the rectangle** reflects the **number of accounts/products owned**.

---

### Key Observations:

#### 1. **Madrid Dominates Product Ownership**
- Madrid occupies the **largest region** in the treemap.
- The largest product here is `ind_cco_fin_ult1` (current account), indicating it is the most commonly owned product.
- Other popular products in Madrid include:
  - `ind_ctop_fin_ult1` (possibly term deposits or similar)
  - `ind_recibo_ult1` (bill payment services)
  - `ind_cno_fin_ult1` (investment fund accounts)
  
#### 2. **Universal Popularity of Current Accounts**
- `ind_cco_fin_ult1` appears prominently in **every province**, reinforcing its importance as the **primary financial product**.
- This suggests that a **current account is a prerequisite** or core product for customers across the board.

#### 3. **Provincial Distribution of Product Types**
- Larger urban provinces (Barcelona, Valencia, A Coruña) show more **diversity in product ownership**.
- Smaller or less urbanized provinces show **fewer and smaller product tiles**, suggesting lower product penetration or customer base.

#### 4. **Implications for Business Strategy**
- **Madrid** can be treated as the **testing ground for new financial products** due to its high activity.
- Regions with smaller rectangles might benefit from **product bundling strategies** or **targeted campaigns** to increase cross-sell rates.

#### 5. **Visual Encoding Benefits**
- The treemap helps in **visually understanding product concentration** by province and prioritizing **marketing or expansion efforts**.

---

### Insight:

The treemap validates the central role of current accounts in the bank's product suite and reveals regional disparities in product adoption. Provinces with a **smaller mix of products** may represent untapped potential, whereas provinces like Madrid or Barcelona are **mature markets** where **new product launches or premium services** might be more effective.



