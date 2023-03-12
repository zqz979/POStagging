from embed_def import *
from utils import *

model="./models/mlp.pth"
vectorizer="./models/vectorizer.sav"
data_format="dictionary"
input_path="./data/test.txt"
output_path="./POS.test.txt"

test_data=load_data(input_path,data_format)
X_test=extract_features(test_data)
vectorizer=load_model(vectorizer)
X_test_vec=vectorizer.transform([feature for sentence in X_test for feature in sentence])
model=load_model(model)
preds=model.predict(X_test_vec)
write_infer(output_path,test_data,preds)