import flask
import numpy as np
from flask import request, jsonify
from functions import *


app = flask.Flask(__name__)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!


@app.route("/print_piped", methods=["POST"])
def print_piped():
    if request.form['chat_in']:
        msg = request.form['chat_in']
        x_input = str(msg)
        x_input, pred_class, pred_proba = make_prediction(x_input)
        print(x_input,pred_class,pred_proba)
        pred_proba=pred_proba*100
        return flask.render_template('predictor.html',
                                chat_in=x_input,
                                prediction_class=pred_class,
                                prediction_prob=np.round(pred_proba,3))
    # return jsonify(pred_class)

@app.route("/predict", methods=["GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    print(request.args)
    if(request.args):
        x_input, pred_class, pred_proba = make_prediction(request.args['chat_in'])
        print(pred_class,pred_proba)
        print(x_input)
        return flask.render_template('predictor.html',chat_in=x_input,prediction_class=pred_class,prediction_prob=pred_proba)
    
    else: 
        return flask.render_template('predictor.html',chat_in=" ",prediction_class=" ",prediction_prob=" ")
    

if __name__=="__main__":
    # For local development:
    # app.run(debug=True)
    # For public web serving:
    app.run(host='0.0.0.0')
    app.run()
