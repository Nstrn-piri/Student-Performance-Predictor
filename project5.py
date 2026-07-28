########                    Student Performance Predictor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
student = {
    "name" : ["ali" , "sara" , "ahmad" , "reza" , "hossein" , "hamed" , "bahar" , "zahra" , "leila" , "maryam"],
    "hours" : [2 , 3 , 4 , 5 , 6 , 7 , 8 , 5 , 7 , 6],
    "absent" : [5 , 4 , 3 , 2 , 2 , 1 , 0 , 4 , 1 , 2],
    "homework" : [10 , 12 , 14 , 15 , 17 , 18 , 20 , 13 , 19 , 16],
    "result" : [0 , 0 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1]
    }
df = pd.DataFrame(student)
df.to_csv("student.csv" , index=False)
df = pd.read_csv("student.csv")
print(f"تعداد دانش آموزان :{len(df["name"])}")
print(f" میانگین ساعات مطالعه :{df["hours"].mean()}")
print(f" میانگین نمره تمرین :{df["homework"].mean()}")
print(f" بیشترین نمره تمرین :{df["homework"].max()}")
print(f" کمترین نمره تمرین :{df["homework"].min()}")
print("قبولی‌ها:")
print(df[df["result"] == 1]["name"])
print("مردودی‌ها:")
print(df[df["result"] == 0]["name"])


plt.bar(df["name"] , df["homework"])
plt.title("Home work")
plt.xlabel("name")
plt.ylabel("homework")
plt.show()

label = ["قبول" , "مردود"]
result = df["result"].value_counts()
plt.pie(result , labels=label , autopct= "%1.1f%%")
plt.title("درصد قبولی")
plt.show()

plt.hist(df["homework"])
plt.title("home work")
plt.show()

plt.scatter(df["hours"] , df["homework"])
plt.title("مقایسه")
plt.show()

#### machine learning
x = df[["hours" , "absent" , "homework"]]
y = df["result"]
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)
model = LogisticRegression()
model.fit(x_train , y_train)
joblib.dump(model , "student_model.pkl")
model = joblib.load("student_model.pkl")
try:
    hours = int(input("ساعت مطالعه را وارد کنید: "))
    absent = int(input("تعداد غیبت را وارد کنید: "))
    homework = int(input("نمره تمرین را وارد کنید: "))
except ValueError:
    print("لطفاً فقط عدد وارد کنید.")
    exit()
new_student  = pd.DataFrame({
    "hours": [hours],
    "absent": [absent],
    "homework": [homework]
})
prediction = model.predict(new_student)
print(prediction)
if prediction[0] == 1:
    print("دانش آموز قبول می‌شود ✅")
else:
    print("دانش آموز مردود می‌شود ❌")
