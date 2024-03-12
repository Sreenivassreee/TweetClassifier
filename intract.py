from coremltools.models import MLModel
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()



coreml_model = MLModel("TweetClassifier.mlmodel")


input_text = ["@apple is very bad phone"]
input_text2 = ["@apple is love"]
input_features = tfidf_vectorizer.transform(input_text)
input_features2 = tfidf_vectorizer.transform(input_text2)

# Convert sparse matrix to dense array
input_features_dense = input_features.toarray()
input_features_dense2 = input_features2.toarray()

# Make predictions
prediction = coreml_model.predict({"input": input_features_dense})
prediction2 = coreml_model.predict({"input": input_features_dense2})
print(prediction)
print(prediction2)