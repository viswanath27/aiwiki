from application import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), unique=True, nullable=False)
    email = db.Column(db.String(500), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}','{self.email}')"

class FirstLevel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), unique=True, nullable=False)
    secondlevel = db.relationship('SecondLevel',backref='parent', lazy=True)

    def __repr__(self):
        return f"firstLevel('{self.title}')"

class SecondLevel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), unique=True, nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('first_level.id'), nullable=False)

    def __repr__(self):
        return f"secondLevel('{self.title}')"

class GanTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gan_name = db.Column(db.String(1000), unique=True, nullable=False)
    gan_definition = db.Column(db.String(5000))

    def __repr__(self):
        return f"gantable('{self.gan_name}')"