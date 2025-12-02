# Loan Approval Prediction System

This project is a web-based application that uses a machine learning model to predict whether a loan application will be approved or rejected based on the applicant's financial details.

## Features

-   A simple web form to input loan application data.
-   A logistic regression model (trained from scratch) to predict the outcome.
-   A clean results page displaying the prediction ("Approved" or "Rejected") and a confidence score.

## Project Structure

```
.
├── Data_approach/
│   ├── Model-Implimentation.ipynb  # Jupyter Notebook for model training and EDA.
│   └── loan_approval_dataset.csv   # The dataset used for training.
├── models/
│   └── model.pkl                   # The serialized (saved) trained model.
├── static/
│   └── css/style.css               # CSS for the web interface.
├── templates/
│   ├── index.html                  # The main application form.
│   ├── result.html                 # The page to display the prediction result.
│   └── error.html                  # Error page.
├── app.py                          # The main Flask application file.
└── requirements.txt                # A list of all Python dependencies.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Loan_Approval_Model
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv vnv
    .\vnv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

1.  **Train the Model (if `model.pkl` is not present):**
    -   Open and run the `Data_approach/Model-Implimentation.ipynb` notebook. This will train the model and save the `model.pkl` file in the `models/` directory.

2.  **Run the Flask Web Server:**
    -   Execute the main application file from your terminal:
        ```bash
        python app.py
        ```

3.  **Access the Application:**
    -   Open your web browser and go to: `http://127.0.0.1:5000`

## Technologies Used

-   **Backend:** Flask
-   **Machine Learning:** Scikit-learn, Pandas, NumPy
-   **Frontend:** HTML, CSS
-   **Development:** Jupyter Notebook

