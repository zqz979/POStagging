import pickle
import utils

best_model="best_logistic.pth"
vectorizer="best_vectorizer.sav"
data_format="dictionary"
input_path="./infer/test.txt"
output_path="./tmp_POS.test.txt"

test_data=utils.load_data(input_path,data_format,True)
X_test=utils.extract_features(test_data)
vectorizer=utils.load_model(vectorizer)
X_test_vec=vectorizer.transform([feature for sentence in X_test for feature in sentence])
model=utils.load_model(best_model)
preds=model.predict(X_test_vec)
utils.write_infer(output_path,test_data,preds)