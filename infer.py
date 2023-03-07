import utils

model="mlp.pth"
vectorizer="vectorizer.sav"
scaler="scaler.sav"
data_format="dictionary"
input_path="./infer/test.txt"
output_path="./POS.test.txt"

test_data=utils.load_data(input_path,data_format)
X_test=utils.extract_features(test_data)
vectorizer=utils.load_model(vectorizer)
X_test_vec=vectorizer.transform([feature for sentence in X_test for feature in sentence])
scaler=utils.load_model(scaler)
X_test_vec=scaler.transform(X_test_vec)
model=utils.load_model(model)
preds=model.predict(X_test_vec)
utils.write_infer(output_path,test_data,preds)