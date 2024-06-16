from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    return render_template('generate.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/restore')
def restore():
    return render_template('restore.html')

if __name__ == '__main__':
    app.run(debug=True)
