# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import bottle
from bottle import route, request, run,response
import os
import uuid




features = [
    "Total Received Late Fee",
    "Total Received Interest",
    "Loan Amount",
    "Debit to Income",
    "Interest Rate",
]


def train():
    # Load data
    data = pd.read_csv("train.csv")

    # Select features and target variable
    X = data[features]
    y = data["Loan Status"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create Random Forest Classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    f = open("model.dat", "wb")

    pickle.dump(model, f)
    f.close()


def test(path, server):
    f = open("model.dat", "rb")
    model = pickle.load(f)
    f.close()

    test_data = pd.read_csv(path)

    # Selecting the features
    X_test = test_data[features]

    # Predicting the results
    y_pred = model.predict(X_test)

    print(y_pred)
    for k in y_pred:
        if k != 0:
            print("DEFAULT")
        else:
            print("SAFE")

    # Evaluating the accuracy
    accuracy = model.score(X_test, y_pred)
    # Printing the accuracy score
    print("Accuracy:", accuracy)
    if server:
        os.remove(path)
    return y_pred


@route("/")
def index():
    return "Server up and running!"


@route("/login")
def login():
    return """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select a file: <input type="file" name="file" />
            <input type="submit" value="Start upload" />
        </form>
    """

@route('/upload', method=['OPTIONS', 'POST'])
def do_upload():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
    if request.method == 'OPTIONS':
        return
    
    upload = request.files.get("file")
    id = uuid.uuid1()
    save_path = f"./uploads/${id}.csv"
    upload.save(save_path)

    return str(test(save_path, server=True)[0])


run(host="localhost", port=8080)
