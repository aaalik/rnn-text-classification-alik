from flask import Flask, Blueprint, flash, g, redirect, render_template, request, url_for
# from flask_assets import Environment, Bundle

app = Flask(__name__)
# assets = Environment(app)

'''
$ export FLASK_APP=hello.py
$ flask run
 * Running on http://127.0.0.1:5000/
deep learning klasifikasi teks metode rnn
python, tensorflow, tflearn, hadoop, hive, flask
 '''

@app.route('/')
def hello():
    return render_template('index.html')
    # return 'makan'

@app.route('/login')
def login():
    return 'login'

# @app.route('/user/<username>')
# def profile(username):
#     return '{}\'s profile'.format(username)

# with app.test_request_context():
#     print(url_for('hello'))
#     print(url_for('login'))
#     print(url_for('login', next='/'))
#     print(url_for('profile', username='John Doe'))

if __name__ == '__main__':
    app.debug = True
    app.run()