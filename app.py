from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model/pipe.pkl', 'rb'))

teams = ['Chennai Super Kings', 'Mumbai Indians', 'Kolkata Knight Riders',
         'Royal Challengers Bangalore', 'Rajasthan Royals', 'Delhi Capitals',
         'Punjab Kings', 'Sunrisers Hyderabad']
venues = ['Wankhede Stadium', 'M. Chinnaswamy Stadium', 'Eden Gardens', 
          'Narendra Modi Stadium', 'Feroz Shah Kotla', 'MA Chidambaram Stadium']

@app.route('/')
def home():
    return render_template('index.html', teams=teams, venues=venues)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = pd.DataFrame([[
        request.form['batting_team'],
        request.form['bowling_team'],
        request.form['venue'],
        int(request.form['target_score']),
        int(request.form['current_score']),
        float(request.form['overs_left']),
        int(request.form['wickets_left'])
    ]], columns=['batting_team', 'bowling_team', 'venue', 'target_score',
                 'current_score', 'overs_left', 'wickets_left'])

    prediction = model.predict(input_data)[0]
    outcome = f"{request.form['batting_team']} is likely to win!" if prediction == 1 else f"{request.form['bowling_team']} is likely to win!"
    return render_template('result.html', prediction=outcome)

if __name__ == '__main__':
    app.run(debug=True)