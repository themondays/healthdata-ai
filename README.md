# Health Data AI
In this project, we're building an AI model that can analyze blood data and make predictions about the likelihood of certain diseases. Expected that the model will be trained on a large dataset of blood test results and other medical information, and it uses this training data to learn patterns and relationships that can be used to make predictions.

One of the key challenges in this project is dealing with the complexity and variability of the data. Blood test results can vary widely from person to person, and there are many different factors that can influence the results, including age, sex, ethnicity, medical history, and current medications. To build an accurate and reliable AI model, we need to carefully preprocess and clean the data, and we need to use advanced machine learning techniques to identify the most important features and patterns.

Another challenge is managing the large scale of the project. We're dealing with millions of records, and the data is constantly changing as new patients are added and new test results are obtained. To keep the model up to date and accurate, we need to use efficient algorithms and data structures, and we need to be able to quickly retrain and update the model as needed.

Overall, this is a challenging but exciting project that has the potential to make a significant impact in the field of healthcare. By helping doctors and patients better understand and manage diseases, we can improve the quality of life for people around the world and make the world a little bit more chaotic.

### Input
This is a simpliest health prediction script developed to analyse critical factors to predict further health risks.

Blood tests can yield a ton of data points, and which ones are important depends on the analysis you're doing. Some common blood test data points include:
1. Complete blood count (CBC): This test measures the number and types of cells in the blood, including red blood cells, white blood cells, and platelets. The CBC can help diagnose a wide range of conditions, such as anemia, infection, and bleeding disorders.
2. Blood chemistry tests: These tests measure the levels of certain substances in the blood, like electrolytes, enzymes, and hormones. Blood chemistry tests can diagnose a variety of conditions, including kidney and liver disease, diabetes, and hormonal imbalances.
3. Coagulation tests: These tests measure how well the blood clots, and can diagnose bleeding disorders or monitor the effectiveness of blood-thinning meds.
4. Infectious disease tests: These tests detect infectious agents like bacteria, viruses, or parasites in the blood. Infectious disease tests can diagnose or monitor the treatment of infections.
5. Tumor marker tests: These tests measure the levels of certain substances in the blood that may be elevated in certain types of cancer. Tumor marker tests are usually used with other diagnostic tests to help detect or monitor cancer.

### Preparation
`age`: Age of the patient (integer)
`sex`: Genetic sex of the patient (integer: 0 for female, 1 for male)
`ethnicity`: Ethnicity of the patient (integer)
`diseases`: List of diseases diagnosed in the patient (list of strings)
`vaccines`: List of vaccines received by the patient (list of strings)
`chemo`: List of chemotherapy treatments received by the patient (list of strings)
`hrt`: List of hormonal therapies received by the patient (list of strings)

The script loads each JSON document into a dictionary, preprocesses the data, and creates a list of data points that includes the relevant information for training the model. It then converts the list of data points to a NumPy array and saves the array to a file using the numpy.save function.

```
python3 utils/prepare.py
```

Some of the fields in the dataset are not integers, such as ethnicity and location. These fields will need to be transformed into numerical values in order to be used as input for a machine learning model.

One way to do this is to use one-hot encoding, which converts a categorical field with multiple possible values into multiple binary fields, each representing a single value. For example, if ethnicity has the possible values 'Caucasian', 'African American', 'Asian', and 'Other', then one-hot encoding would create four binary fields, one for each possible value, with a value of 1 indicating that the sample belongs to that category and a value of 0 indicating that it does not.

To apply one-hot encoding to a categorical field in the dataset, you can use a library such as scikit-learn or pandas.

```
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.DataFrame(data, columns=[
    'age',
    'sex',
    'ethnicity',
    'num_diseases',
    'num_vaccines',
    'num_chemo',
    'num_hormonal_therapies',
    'num_cancers',
    'hrt',
    'hiv_aids_positive',
    'remission',
    'red_blood_cell_count',
    'white_blood_cell_count',
    'platelet_count',
    'sodium_level',
    'potassium_level',
    'creatinine_level',
    'ast_level',
    'alt_level',
    'bilirubin_level',
    'thyroid_stimulating_hormone_level',
    'prolactin_level',
    'clotting_time'
])
df = pd.get_dummies(df, columns=['ethnicity'])
```

This will create four additional fields in the DataFrame, one for each possible value of ethnicity, and assign a value of 1 or 0 to each field depending on the value of ethnicity in the original data.

### Using a Pre-Trained Data
It is important to be aware of the various laws and regulations that protect the privacy of medical patients and limit the ability to disclose personal information to third parties. In the United States, the Health Insurance Portability and Accountability Act (HIPAA) sets strict rules for the protection of personal health information. In the European Union, the General Data Protection Regulation (GDPR) provides comprehensive rules for the protection of personal data. And in Australia, the Privacy Act 1988 sets rules for the collection, use, and disclosure of personal information by the government and certain private sector organizations. These laws and regulations are in place to ensure the privacy and security of sensitive medical data, and it is important to be familiar with them when working with this type of information.

### Data Samples and Datasets
When working with large datasets, JSON is often a better choice than CSV because it is more flexible and easier to work with. JSON is a widely used data interchange format that allows you to represent complex data structures, including lists, dictionaries, and nested objects. This makes it easier to represent and process data that has multiple levels of complexity, such as medical records that contain a variety of different types of information.

One of the key advantages of JSON is that it can be easily integrated with third-party APIs, document storage systems, and SQL databases. This makes it easy to exchange data between different systems and to build complex data pipelines that involve multiple stages of processing. For example, you might use a JSON API to fetch patient data from an electronic health records system, and then use another API to store the data in a cloud-based document storage system. This type of integration is much easier to achieve with JSON than with CSV, which is a simpler and less flexible format.

Another advantage of JSON is that it is easy to read and write, even for humans. This makes it easier to debug and troubleshoot issues with your data, and it also makes it easier to collaborate with other developers and data scientists who may be working on the same project.

```
{
  "patient_id": "64ee1ed3-0e13-43ef-959e-99331253ff15",
  "age": 35,
  "sex": "male",
  "ethnicity": "Caucasian",
  "location": "New York, NY",
  "country": "US",
  "diseases": ["A", "B", "C"],
  "vaccines": ["influenza", "tetanus"],
  "hrt": true,
  "hormonal_therapies": ["testosterone replacement"],
  "hiv_aids_positive": true,
  "cancers": ["lung", "colon"],
  "chemo": ["carboplatin", "paclitaxel"],
  "remission": false,
  "red_blood_cell_count": 4.7,
  "white_blood_cell_count": 8.1,
  "platelet_count": 250,
  "sodium_level": 142,
  "potassium_level": 4.3,
  "creatinine_level": 0.9,
  "ast_level": 22,
  "alt_level": 32,
  "bilirubin_level": 0.5,
  "thyroid_stimulating_hormone_level": 1.4,
  "prolactin_level": 10,
  "clotting_time": 8,
  "infectious_agent": "none"
}
```

This is just a small sampling of the data points that can be obtained from blood tests, and there are many other data points that can be relevant depending on the specific needs of the analysis.

This script loads a trained AI model from a file (in this case, a .h5 file), defines the input data for a patient, and uses the model to predict the likelihood of a change in disease status based on the input data. It then prints the prediction as a percentage.

### Usage
1. Before you can use the AI model, you will need to prepare your dataset. This involves placing your JSON dataset in the `input/users` directory.
2. It is important to ensure that your dataset is in the proper format. If it is not, you may need to adjust the scripts to ensure that they can process your data correctly.
3. To aggregate your dataset into a format that can be used for training the AI model, you will need to run the `prepare.py` script. You can do this by running the command `python3 utils/prepare.py`.
4. Once your dataset is prepared, you can train the AI model by running the train.py script. You can do this by running the command `python3 train.py`.
5. After the model has been trained, you can use it to make predictions by running the analyse.py script. You can do this by running the command `python3 analyse.py`.