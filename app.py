from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objs as go
import plotly.offline as pyo
from wordcloud import WordCloud
import io
import matplotlib.pyplot as plt
import base64
import seaborn as sns

app = Flask(__name__)

# Load the model and vectorizer
with open("models.p", 'rb') as mod:
    data = pickle.load(mod)
vect = data['vectorizer']
logreg_model = data["logreg"]
svm_model = data["svm"]

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', ' ', text.lower())
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in filtered_words]).strip()

# Generate word cloud image
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color="#f6f5f6", colormap='viridis').generate(text)
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    plt.close()  # Close the plot to avoid overlap
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Generate confusion matrix image
def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative", "Neutral"])
    img = io.BytesIO()
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Positive", "Negative", "Neutral"], yticklabels=["Positive", "Negative", "Neutral"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(img, format="png")
    plt.close()  # Close the plot to avoid overlap
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_result = plot_url = wordcloud_img = sentiment_counts = sentiment_table = None
    positive_count = negative_count = neutral_count = accuracy = confusion_matrix_img = error_message = None

    if request.method == 'POST':
        action = request.form['action']
        classifier = request.form['classifier']

        if action == 'single_review':
            review = preprocess_text(request.form['single_review'])
            inp_test = vect.transform([review])
            model = logreg_model if classifier == "Logistic Regression" else svm_model
            sentiment_result = model.predict(inp_test)[0]

        elif action == 'multiple_reviews':
            file = request.files['uploaded_file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                if df.shape[1] == 1:
                    df.columns = ["Review"]
                    df['Review'] = df['Review'].apply(preprocess_text)
                    model = logreg_model if classifier == "Logistic Regression" else svm_model

                    sentiments = [model.predict(vect.transform([review]))[0] for review in df["Review"]]
                    df["Sentiment"] = sentiments
                    y_true = df["Sentiment"]
                    y_pred = sentiments
                    accuracy = accuracy_score(y_true, y_pred)

                    # Generate images separately to avoid overlap
                    confusion_matrix_img = generate_confusion_matrix(y_true, y_pred)
                    wordcloud_img = generate_wordcloud(" ".join(df["Review"].astype(str)))

                    # Calculate sentiment counts
                    sentiment_counts = df["Sentiment"].value_counts().to_dict()
                    positive_count = sentiment_counts.get("Positive", 0)
                    negative_count = sentiment_counts.get("Negative", 0)
                    neutral_count = sentiment_counts.get("Neutral", 0)

                    # Generate bar chart for sentiment counts
                    fig = go.Figure([go.Bar(x=["Positive", "Negative", "Neutral"], y=[positive_count, negative_count, neutral_count], marker_color='#8d7995')])
                    fig.update_layout(title="Product Reviews Analysis", xaxis_title="Sentiment", yaxis_title="Count")
                    plot_url = pyo.plot(fig, include_plotlyjs=False, output_type='div')

                    # Convert DataFrame to HTML table
                    sentiment_table = df.to_html(classes="table table-striped", index=False)
                else:
                    error_message = "Please make sure the CSV file has only one column."
            else:
                error_message = "Please enter the CSV file only."

    return render_template('index.html', sentiment_result=sentiment_result, plot_url=plot_url,
                           wordcloud_img=wordcloud_img, sentiment_counts=sentiment_counts,
                           sentiment_table=sentiment_table, positive_count=positive_count,
                           negative_count=negative_count, neutral_count=neutral_count,
                           accuracy=accuracy, confusion_matrix_img=confusion_matrix_img,
                           error_message=error_message)


if __name__ == "__main__":
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    app.run(debug=True)
