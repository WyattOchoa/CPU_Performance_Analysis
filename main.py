from flask import Flask, render_template, request
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sys

app = Flask(__name__)

# Logging Functionality of system and user errors
sec_log = open('system_logging', 'a')
sys.stderr = sec_log


# Login Form
@app.route('/')
def login():
    title = "Login to Dashboard"
    return render_template('login.html', title=title)


# Login Page
@app.route('/dashboard', methods=["POST", 'GET'])
def dashboard():
    username = request.form.get("username")
    password = request.form.get("password")
    title = "Dashboard Home"
    if not username or not password or (username != "guest" and password != "password"):
        error_st_login_credentials = "Incorrect Login Credentials"
        sec_log.write('Invalid Login Attempt at: ' + str(datetime.now()))
        return render_template('login.html', title=title, error_st_login_credentials=error_st_login_credentials)
    return render_template('dashboard.html', title=title, username=username, password=password)


# Allows querying of a date for performance prediction and statics.
@app.route('/query', methods=["POST", "GET"])
def query():
    month = (request.form.get("month"))
    year = request.form.get("year")
    # Error Handling for empty inputs
    if month == '' or year == '':
        error_st_date_value = "Error: Values cannot be empty"
        sec_log.write('Empty M/Y Query: ' + str(datetime.now()))
        return render_template('dashboard.html', error_st_date_value=error_st_date_value)
    month = int(month)
    year = int(year)
    # Error Handling for improper inputs for month and year.
    if not (0 < month < 13) or not (2003 < year < 2035):
        error_st_date_value = "Error: Invalid Values"
        sec_log.write('Invalid M/Y Query: ' + str(datetime.now()))
        return render_template('dashboard.html', error_st_date_value=error_st_date_value)
    user_date_query = []
    user_date_query.append(str(year) + '-' + str(month) + '-01T00:00:00.000000000')
    # A single integer entered in instead of one with a leading zero would create a different in the performance output,
    # Handled by clearing the array, adding in a zero, and then passing on.
    if (len(user_date_query[0])) != 29:
        user_date_query.clear()
        user_date_query.append(str(year) + '-0' + str(month) + '-01T00:00:00.000000000')
    user_date_format_query = np.array(pd.to_datetime(user_date_query), dtype=float).reshape(-1, 1)
    user_prediction = model.predict(user_date_format_query)
    user_prediction = int(user_prediction.item())
    user_query_graphing(user_date_format_query, user_prediction)

    # avg variable = coefficient of determination (different in prediction vs actual)
    avg = model.score(x, y)
    # Rounded to and converted to a percentage for ease of viewing
    avg = str(round((avg * 100), 2)) + '%'
    return render_template('dashboard.html', user_prediction=user_prediction, avg=avg)


# Creates a df (data frame) object from the read in CSV
cpu_data = pd.read_csv('Date_Sorted_CPU_Benchmarks.csv')

# Setting the index to release date as type: DateTimeIndex
cpu_data['releaseDate'] = pd.to_datetime(cpu_data.releaseDate)
cpu_data.set_index('releaseDate', inplace=True)

# Dropping all unnecessary data
cpu_data = cpu_data.drop(columns=['cpuName', 'testDate', 'price', 'cpuMark', 'cpuValue', 'threadValue', 'TDP',
                                  'powerPerf', 'cores', 'socket', 'category'])

# cpu_data grouped by unique index (multiple releases on a single date average into one)
cpu_data_unique = cpu_data.groupby(cpu_data.index).mean()

# cpu_data grouped by year (annual performance average)
cpu_data_year = cpu_data.groupby(cpu_data.index.year).mean()

converted_annual_indexes = []
for i in cpu_data_year.index:
    converted_annual_indexes.append(str(i) + '-01-01T00:00:00.000000000')

# x converted to single column array of DateTimeIndex numeric values
x = np.array(cpu_data_unique.index.values, dtype=float).reshape(-1, 1)

# y converted to single column array of floats
y = np.array(cpu_data_unique['threadMark'], dtype=float).reshape(-1, 1)

# y rollings (smoothed out)
y_rolling = np.array((cpu_data_unique['threadMark'].rolling(window=10).mean()), dtype=float).reshape(-1, 1)
y_rolling2 = np.array((cpu_data_unique['threadMark'].rolling(window=20).mean()), dtype=float).reshape(-1, 1)

# annual average sets
x_annual = np.array(pd.to_datetime(cpu_data_year.index.values), dtype=float).reshape(-1, 1)
y_annual = np.array(cpu_data_year['threadMark'], dtype=float).reshape(-1, 1)

# annual average that will work with the trained model
x_annual_converted = np.array(pd.to_datetime(converted_annual_indexes), dtype=float).reshape(-1, 1)

# Splitting up dataset into training (75%) and testing data (25%) sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

# Array that gives the next five years after the end of the last year in the original csv
starting_year = x_annual.flatten().astype(int)[-1]
next_five_years = []
for i in range(1, 6):
    next_five_years.append(starting_year + i)
x_next_five = []
for i in next_five_years:
    x_next_five.append(str(i) + '-01-01T00:00:00.000000000')
next_five = np.array(pd.to_datetime(x_next_five), dtype=float).reshape(-1, 1)

# Linear Regression Model using training data
model = LinearRegression().fit(x_train, y_train)

# Testing of Prediction Data
prediction = model.predict(x_test)

# ===================================================================================================================

# Identifier 1: Top Left Graph
fig_1, ax1 = plt.subplots(1, 1, figsize=(12, 4))
width = 0.4

# Actual Annual Values
ax1.bar(x_annual.flatten() + width / 2, y_annual.flatten(), width, label='Actual Value')

# Prediction made based on annual values
y_annual_prediction = model.predict(x_annual_converted)

# Prediction Annual Values
ax1.bar(x_annual.flatten() - width / 2, y_annual_prediction.flatten(), width, color='pink', label='Predicted Value')
ax1.legend()

# Setting the X tick-frequency
start = x_annual.flatten().astype(int)[0]
plt.xticks(np.arange(start, len(x_annual) + start, 1))

ax1.set_facecolor('honeydew')
ax1.set_xlabel("Years")
ax1.set_ylabel("Average Annual Performance")
ax1.set_title("Accuracy of the Machine Learning Model")
plt.savefig('static/images/figure_1')

# ===================================================================================================================

# Identifier 2: Middle Right Graph
fig2, ax1 = plt.subplots(1, 1, figsize=(10, 5))

# Prediction made based on annual values
y_next_five_prediction = model.predict(next_five)

# Prediction Annual Values
five_bar_layout = ax1.bar(next_five_years, y_next_five_prediction.flatten(), width=0.5)
ax1.bar_label(five_bar_layout, padding=3)

# Setting the Y tick-frequency
y_last = len(y_next_five_prediction.flatten()) - 1
# The top of the graph will always be around 1000 larger than the highest performance value
y_top = y_next_five_prediction.flatten()[y_last] + 1500
plt.yticks(np.arange(0, y_top, 500))

ax1.set_facecolor('honeydew')
ax1.set_xlabel("Next Five Years")
ax1.set_ylabel("Average Annual Performance")
ax1.set_title("Estimated Annual Performance over the Next Five Years")
y_next_five_prediction.flatten().astype(int)
plt.savefig('static/images/figure_2')

# ===================================================================================================================

# Identifier 3a: Bottom Left Graph (Historical Data without Smoothing)
fig3a, ax1 = plt.subplots(1, 1, figsize=(7, 5))
ax1.set_facecolor('honeydew')
ax1.set_xlabel('releaseDate')
ax1.set_ylabel("Average Performance")
ax1.set_title('Original Data')
# Date Formatting of time array
x3a = np.array(x, dtype='<M8[ns]')
ax1.plot(x3a, y)
plt.savefig('static/images/figure_3a')

# Identifier 3a: Bottom Left Graph (Historical Data with Low Smoothing)
fig3b, ax1 = plt.subplots(1, 1, figsize=(7, 5))
ax1.set_facecolor('honeydew')
ax1.set_xlabel('releaseDate')
ax1.set_ylabel("Average Performance")
ax1.set_title('Smoothed Data')
# Date Formatting of time array
x3b = np.array(x, dtype='<M8[ns]')
ax1.plot(x3b, y_rolling)
plt.savefig('static/images/figure_3b')

# Identifier 3c: Bottom Left Graph (Historical Data with High Smoothing)
fig3c, ax1 = plt.subplots(1, 1, figsize=(7, 5))
ax1.set_facecolor('honeydew')
ax1.set_xlabel('releaseDate')
ax1.set_ylabel("Average Performance")
ax1.set_title('Smoothed Data')
# Date Formatting of time array
x3c = np.array(x, dtype='<M8[ns]')
ax1.plot(x3c, y_rolling2)
plt.savefig('static/images/figure_3c')

# ===================================================================================================================

# Identifier: Figure 4 Line of Regression (Without user prediction point) Middle Left Graph
fig4, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.set_facecolor('honeydew')
ax1.set_xlabel('releaseDate')
ax1.set_ylabel("Average Performance")
ax1.set_title('Calculated Regression Line')
x1_test = np.array(x_test, dtype='<M8[ns]')
ax1.plot(x1_test, prediction, color='pink')
plt.savefig('static/images/figure_4')


# Function to plot a single point using the date the user queried.
def user_query_graphing(x_entry, y_entry):
    fig4, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.set_facecolor('honeydew')
    ax1.set_xlabel('releaseDate')
    ax1.set_ylabel("Average Performance")
    ax1.set_title('Calculated Regression Line with Requested Performance Prediction')
    x1_test = np.array(x_test, dtype='<M8[ns]')
    ax1.plot(x1_test, prediction, color='pink', label='Line of Regression')
    x_entry_formatted = np.array(x_entry, dtype='<M8[ns]')
    ax1.scatter(x_entry_formatted, y_entry, s=100, color='green', label='Prediction Point')
    ax1.legend()
    plt.savefig('static/images/figure_4')


# ===================================================================================================================

# Runs the Flask Application
if __name__ == '__main__':
    app.run()
