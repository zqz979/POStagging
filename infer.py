import pickle
import utils

best_model="logistic.pth"
data_format="dictionary"
input_path="./infer/test.txt"
output_path="./POS.test.txt"

model=utils.load_model(best_model)
test_data=utils.load_data(input_path,data_format,True)
X_test=utils.extract_features(test_data)
with open("./models/vectorizer.sav", 'rb') as file:
    vectorizer=pickle.load(file)
X_test_vec=vectorizer.transform([feature for sentence in X_test for feature in sentence])
preds=model.predict(X_test_vec)
utils.write_infer(output_path,test_data,preds)