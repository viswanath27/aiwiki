from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, RadioField, FloatField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from application.models import User

class RegistrationForm(FlaskForm):
    username = StringField('Username',validators=[DataRequired(), Length(min=2, max=200)])
    email = StringField('Email',validators=[DataRequired(), Email(), Length(max=500)])
    password = PasswordField('Password',validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken')

    def validate_email(self, email):
        email = User.query.filter_by(email=email.data).first()
        if email:
            raise ValidationError('That email is taken')

class LoginForm(FlaskForm):
    #username = StringField('Username',validators=[DataRequired(), Length(min=2, max=200)])
    email = StringField('Email',validators=[DataRequired(), Email(), Length(max=500)])
    password = PasswordField('Password',validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    #confirm_password = PasswordField('Confirm Password', validators=[DataRequired(),EqualTo('password')])
    submit = SubmitField('Login')

class ProcessDataForm(FlaskForm):
    extractippath = StringField('Input Folder Path',validators=[DataRequired(),Length(max=3000)])
    extractoppath = StringField('Output Folder Path',validators=[DataRequired(),Length(max=3000)])
    #FileFormat = RadioField('FileFormat', choices = [('PNG','PNG'),('JPEG','JPEG'),('BMP','BMP')])
    extractsubmit = SubmitField('Data Process')

class ConvertDataForm(FlaskForm):
    convertippath = StringField('Convert Input Path',validators=[DataRequired(),Length(max=3000)])
    convertoppath = StringField('Convert Output Path',validators=[DataRequired(),Length(max=3000)])
    imagelabel = StringField('Image Label',validators=[DataRequired(),Length(max=200)])
    FileFormat = RadioField('FileFormat', choices = [('png','PNG'),('jpeg','JPEG'),('bmp','BMP')])
    convertsubmit = SubmitField('Convert')

class PackageDataForm(FlaskForm):
    #packageippath = StringField('Package Input Path',validators=[DataRequired(),Length(max=3000)])
    packageoppath = StringField('Package Output Path',validators=[DataRequired(),Length(max=3000)])
    train_vs_test = RadioField('FileFormat', choices = [('1','80-20'),('2','70-30'),('3','60-40')])
    packagesubmit = SubmitField('Package')

class LRegressionForm(FlaskForm):
    lrippath = StringField('Logistic Regression Input Path',validators=[DataRequired(),Length(max=3000)])
    lrregression = FloatField('Regression Value',validators=[DataRequired()])
    lrsubmit = SubmitField('Start Regression')

class RecoEngineForm(FlaskForm):
    recopath = StringField('Input Path',validators=[DataRequired(),Length(max=3000)])
    recocorrection = FloatField('Rgression Correction')
    recosubmit = SubmitField('Recommendation Engine')

class image_classification(FlaskForm):
    image_classify = SubmitField('Classify Image')

class object_identify(FlaskForm):
    object_identify_submit = SubmitField('Object Identify')

class language_translate(FlaskForm):
    languages = SelectField('INPUT LANGUAGE SELECTION',choices=(('Telugu', 'Telugu'), ('Hindi', 'Hindi'), ('Spanish','Spanish')))
    models = RadioField('MODEL SELECTION', choices=[('BERT','BERT'),('TRANSFORMER','TRANSFORMER'),('LSTM','LSTM')])
    input_lang_text = StringField('Input Text',validators=[DataRequired(),Length(max=3000)])
    lang_submit = SubmitField('Translate')
    output_lang_text = StringField('Translated Text',validators=[Length(max=3000)])



    