# 📊 Power BI Dashboard Guide for Fake News Detection

Welcome to the dashboarding guide! A killer dashboard is essential for winning hackathons because it shows judges the *business value* of your AI.

## 🎯 Dashboard Goal
Your dashboard should answer this question for the judges: 
> *"If a media company uses our AI, what insights do they get about the news they are processing?"*

## 📁 Data Preparation
Before importing into Power BI, run the AI model on a test set and save a CSV containing:
1. The News Text
2. Actual Label (`Real`/`Fake`)
3. Predicted Label (`Real`/`Fake`)
4. Confidence Percentage (0-100%)

*(I can write a script to generate this CSV if you need it!)*

## 🎨 Layout & Theme
* **Colors**: Keep it professional. Use a dark mode theme with neon accents (e.g., Neon Blue for Real, Neon Red for Fake).
* **Font**: Segoe UI or Din (Clean and modern).

## 📈 Visualizations Needed

### 1. KPI Cards (Top Row)
These should instantly tell the story:
* **Total News Analyzed**: (Count of rows)
* **Fake News Detected**: (Count where Prediction = Fake)
* **Average AI Confidence**: (Average of Confidence Percentage)
* **Model Accuracy**: (Example: 69.6%)

### 2. Donut Chart: Fake vs Real Distribution
* **Values**: Count of News
* **Legend**: Predicted Label
* *Purpose*: Shows the proportion of fake vs real news the system is catching.

### 3. Histogram: Confidence Distribution
* **X-Axis**: Confidence Percentage (Binned in groups of 10%)
* **Y-Axis**: Count of News
* *Purpose*: Shows judges that your AI is mostly highly confident in its predictions.

### 4. Word Cloud: Top Frequent Words in Fake News
* Use a Word Cloud custom visual in Power BI.
* Filter only for `Prediction = Fake`.
* *Purpose*: Gives insights into what words are commonly used to spread misinformation (e.g., "shocking", "revealed", etc.)

### 5. Data Table: Recent Predictions
* **Columns**: News Text | Predicted Label | Confidence
* *Purpose*: Gives judges a way to read actual news items and see how the AI performed.

## 🎛️ Interactive Filters (Slicers)
* Add a Slicer for **Predicted Label** (Fake / Real). This allows judges to click "Fake" and see the whole dashboard update to show only fake news insights!

## 💡 Pro Tip for Hackathon
Don't just show the dashboard. Tell a story!
* *"As you can see on our live dashboard, our system processed 3,000 news articles today. 48% were flagged as fake, and notice the word cloud... these specific keywords are heavily used in misinformation campaigns. This allows moderators to be proactive."*
