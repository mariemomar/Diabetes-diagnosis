import joblib
import numpy as np

loaded_log_reg = joblib.load("logistic_regression_model.pkl")
loaded_svm = joblib.load("svm_model.pkl")
loaded_svm_rbf = joblib.load("svm_model_rbf.pkl")
scaler = joblib.load("scaler.pkl")

def diabetes_diagnosis(user_data):
    data = np.array(user_data).reshape(1, -1)
    data_scaled = scaler.transform(data)

    pred_logreg = loaded_log_reg.predict(data_scaled)[0]
    pred_svm = loaded_svm.predict(data_scaled)[0]
    pred_svm_rbf = loaded_svm_rbf.predict(data_scaled)[0]


    return pred_logreg , pred_svm , pred_svm_rbf



user_input = [6,148,72,35,0,33.6,0.627,50]    

res = diabetes_diagnosis(user_data=user_input)

# apply the concept of majority voting 
# if the sum of the three models output = 3 or 2 then diabetes as there are 3 or 2 models have predicted (1)
# else the person is not diabeted as there is one or zero models say that the person is diabetes

vote_sum = 0 
for i in range(3):
    vote_sum += res[i]

if vote_sum >= 2 :
    print("A Person with diabetes ")  
else :
    print("A Person with no diabetes")  
