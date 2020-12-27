from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = '7efcdf5e6d2f610d15b3afda76a9e4cd'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aiDatabase.db'
db = SQLAlchemy(app)
db.create_all()
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

from application import routes