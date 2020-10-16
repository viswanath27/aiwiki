from application.models import User, FirstLevel, SecondLevel
from application import app, db, bcrypt
from flask import render_template, flash, redirect, url_for, Markup, Flask, request, Response
from application.forms import RegistrationForm, LoginForm, ProcessDataForm, ConvertDataForm, PackageDataForm, LRegressionForm, RecoEngineForm, image_classification, object_identify, language_translate
from flask_login import login_user, current_user, logout_user
from application.dataprocess.dataprocess import ConvertDICOM2PNG, MNISTFormat, getListOfFilescount, getListOfFiles
from application.database.database import dbprocess
#from application.mlprocess.lr import lrrun
#from application.mlprocess.imageclassification import predict
#from application.mlprocess.object_identify import objectIdentify ,objectnaming
#from application.mlprocess.language_translation import main_translate_function
import os
from werkzeug.utils import secure_filename
import numpy, imageio, glob, sys, os, random
import time
from PIL import Image



#UPLOAD_FOLDER = '/home/vissu'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
count = 0
total_file_count = 0
image_path = []
image_count = []
image_tag = []
classified_full_filename = '/static/image_analysis/1.jpg'
boxed_full_filename = '/static/image_analysis/1.jpg'
#image_file_name = ''
#app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tableOfContents = [
    {'name':'Introduction'},
    {'name':'Classifiers'},
    {'name':'deeplearning'},
    {'name':'Model Selection'},
    {'name':'Dimensionality Reduction'},
    {'name':'Clustering'},
    {'name':'Hvac Domain'},
    {'name':'Problems'},
    {'name':'Open Questions'},
    {'name':'Book'}
]
'''
@app.route("/")
def index():
    return render_template("intro.html")
'''




  
'''
@app.route("/gan",methods=['GET','POST'])
def gan():
    return render_template('gan.html')
'''

'''
@app.route('/progress')
def progress():
    def generate():
        x = 0
        while x <= 100:
            yield "data:" + str(x) + "\n\n"
            x = x + 10
            time.sleep(0.5)
    return Response(generate(), mimetype= 'text/event-stream')  
'''

@app.route("/chatbot",methods=['GET','POST'])
def chatbot():
    return render_template('chatbot.html')



@app.route("/register",methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('intro'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Registration complete for {form.username.data}!','success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)




@app.route("/",methods=['GET','POST'])
@app.route("/login", methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('intro'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('start',heading='Introduction'))
        else:
            flash(f'Unsuccessful Login!. Check email and password','danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/recoengine",methods=['GET','POST'])
def recoengine():
    recoform = RecoEngineForm()
    #names = [0.25,0.5,0.75,1]
    recoform.recocorrection = [0.25,0.5,0.75,1]
    if recoform.validate_on_submit():
        recommendationEngine()
    return render_template('recoengine.html',recoform=recoform)



def upload():
    print("UPLOAD THE FILE FUNCTION")
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        print(destination)

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    global classified_full_filename
    #lrform = LRegressionForm()
    print("This is the upload feature")
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            image_file_name = photo.filename            
            photo.save(os.path.join('/home/vissu/ai-server/ai-server/application/static/image_analysis', photo.filename))
            print("Photo Saved!")
    #full_filename = '/home/vissu/ai-server/ai-server/application/mlprocess/images/dog.jpeg'
    #image_file_name = photo.filename
    print('image_file_name')
    print(classified_full_filename)
    classified_full_filename = '/static/image_analysis/'+photo.filename
    print('full_filename')
    print(classified_full_filename)
    #return render_template('lr.html')
    #get_img(full_filename)
    print('full_filename in upload')
    print(classified_full_filename)
    return redirect(url_for('lr'))


@app.route("/cnn")
def cnn():
    return render_template('cnn.html')

@app.route("/cnn2")
def cnn2():
    return render_template('/cnn_folder/cnn2.html')


@app.route("/cnn3")
def cnn3():
    return render_template('/cnn_folder/cnn3.html')

@app.route("/cnn4")
def cnn4():
    return render_template('/cnn_folder/cnn4.html')

@app.route("/cnn5")
def cnn5():
    return render_template('/cnn_folder/cnn5.html')

@app.route("/cnn6")
def cnn6():
    return render_template('/cnn_folder/cnn6.html')

@app.route("/cnn7")
def cnn7():
    return render_template('/cnn_folder/cnn7.html')

@app.route("/cnn8")
def cnn8():
    return render_template('/cnn_folder/cnn8.html')

@app.route("/cnn9")
def cnn9():
    return render_template('/cnn_folder/cnn9.html')

@app.route("/cnn10")
def cnn10():
    return render_template('/cnn_folder/cnn10.html')

@app.route("/cnn11")
def cnn11():
    return render_template('/cnn_folder/cnn11.html')

@app.route("/cnn12")
def cnn12():
    return render_template('/cnn_folder/cnn12.html')

@app.route("/cnn13")
def cnn13():
    return render_template('/cnn_folder/cnn13.html')

@app.route("/cnn14")
def cnn14():
    return render_template('/cnn_folder/cnn14.html')

@app.route("/cnn15")
def cnn15():
    return render_template('/cnn_folder/cnn15.html')

@app.route("/cnn16")
def cnn16():
    return render_template('/cnn_folder/cnn16.html')

@app.route("/cnn17")
def cnn17():
    return render_template('/cnn_folder/cnn17.html')

@app.route("/cnn18")
def cnn18():
    return render_template('/cnn_folder/cnn18.html')

@app.route("/simens")
def simens():
    return render_template('requirements/simens.html')

@app.route("/ifintalent")
def ifintalent():
    return render_template('requirements/ifintalent.html')

@app.route("/zycus")
def zycus():
    return render_template('requirements/zycus.html')

@app.route("/indegene")
def indegene():
    return render_template('requirements/indegene.html')

@app.route("/cm1")
def cm1():
    return render_template('requirements/cm1.html')

@app.route("/skyleaf")
def skyleaf():
    return render_template('requirements/skyleaf.html')

@app.route("/cm2")
def cm2():
    return render_template('requirements/cm2.html')

@app.route("/cm3")
def cm3():
    return render_template('requirements/cm3.html')    

@app.route("/requirement")
def requirement():
    return render_template('requirement.html')

@app.route("/axim")
def axim():
    return render_template('requirements/axim.html')

@app.route("/newhandleUpload", methods=['POST'])
def newhandleFileUpload():
    global boxed_full_filename
    #lrform = LRegressionForm()
    print("This is the upload feature")
    if 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            image_file_name = photo.filename            
            photo.save(os.path.join('/home/vissu/ai-server/ai-server/application/static/box_image', photo.filename))
            print("Photo Saved!")
    #full_filename = '/home/vissu/ai-server/ai-server/application/mlprocess/images/dog.jpeg'
    #image_file_name = photo.filename
    print('boxed_full_filename')
    print(boxed_full_filename)
    boxed_full_filename = '/static/box_image/'+photo.filename
    print('boxed_full_filename')
    print(boxed_full_filename)
    #return render_template('lr.html')
    #get_img(full_filename)
    print('full_filename in upload')
    print(boxed_full_filename)
    return redirect(url_for('lr'))

@app.route('/progress')
def progress():
    def generate():
        x = 0
        
        while x <= 100:
            yield "data:" + str(x) + "\n\n"
            x = x + 20
            time.sleep(0.5)

    return Response(generate(), mimetype= 'text/event-stream')


@app.route("/lr",methods=['GET','POST'])
def lr():
    global full_filename
    global boxed_full_filename
    lrform = LRegressionForm()
    image_classify = image_classification()
    obj_identify = object_identify()
    colours = ['Red', 'Blue', 'Black', 'Orange']
    #full_filename = '/static/1.jpg'
    print("This is the lr form")
    if lrform.validate_on_submit():
        print("Upload is hit")
        #print (lrrun(lrform.lrippath.data))
        #dbprocess()
        #predict()

        #upload()
        #handleFileUpload()
        #return render_template('lr.html',lrform=lrform)
    #if image_classify.validate_on_submit():
    #    full_filename = '/static/1.jpg'
    #if obj_identify.validate_on_submit():

    #print('full_filename')
    #print(full_filename)

    return render_template('lr.html',lrform=lrform,user_image=classified_full_filename,box_user_image=boxed_full_filename,obj_identify=obj_identify,colours=colours)


@app.route("/box_img")
def box_img():
    global boxed_full_filename
    print("boxed_full_filename")
    print(boxed_full_filename)
    print("This is the image display")
    print('Analyzing the image')
    file_name_list = boxed_full_filename.split('/')
    print('file name list')
    print(file_name_list)
    file_name = file_name_list[-1]
    print('/home/vissu/ai-server/ai-server/application'+boxed_full_filename)
    #result_image = objectnaming('/home/vissu/ai-server/ai-server/application'+boxed_full_filename,file_name)
    print("Result image after analysis")
    print(result_image)
    #result_image = "dog.jpeg"
    #image = Image.open(result_image)
    #image.show()
    #return 'classified_image.png'
    print('final image -----------------------------')
    #print('classified_'+image_file_name)
    return 'classified_'+file_name

@app.route("/get_img")
def get_img():
    #global image_file_name
    global classified_full_filename
    print("classified_full_filename")
    print(classified_full_filename)
    print("This is the image display")
    print('Analyzing the image')
    file_name_list = classified_full_filename.split('/')
    print('file name list')
    print(file_name_list)
    file_name = file_name_list[-1]
    print('/home/vissu/ai-server/ai-server/application'+classified_full_filename)
    #result_image = objectIdentify('/home/vissu/ai-server/ai-server/application'+classified_full_filename,file_name)
    print("Result image after analysis")
    print(result_image)
    #result_image = "dog.jpeg"
    #image = Image.open(result_image)
    #image.show()
    #return 'classified_image.png'
    print('final image -----------------------------')
    print('classified_'+file_name)
    return 'classified_'+file_name
    #return 'a.jpg'

@app.route("/Process",methods=['GET','POST'])
def Process():
    global total_file_count
    ProcessForm = ProcessDataForm()
    ConvertForm = ConvertDataForm()
    PackageForm = PackageDataForm()
    if ProcessForm.validate_on_submit():
        print ("Process Submit button pushed")
        total_file_count = getListOfFilescount(ProcessForm.extractippath.data)
        print (len(total_file_count))
        getListOfFiles(ProcessForm.extractippath.data,ProcessForm.extractoppath.data)
        print ("copy function executed")
        #return redirect(url_for('Process'))
        count = 0
    if ConvertForm.validate_on_submit():
        print("Convert button pushed")
        ConvertDICOM2PNG(ConvertForm.convertippath.data,ConvertForm.convertoppath.data,ConvertForm.FileFormat.data,ConvertForm.imagelabel.data)
    if PackageForm.validate_on_submit():
        #PackageData(PackageForm.packageoppath.data,PackageForm.train_vs_test.data)
        MNISTFormat()
    return render_template('Process.html',ProcessForm=ProcessForm,ConvertForm=ConvertForm,PackageForm=PackageForm)



@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/dataanalysis")
def dataanalysis():
    return redirect(url_for('dataanalysis.html'))

@app.route("/crypto")
def crypto():
    return render_template('crypto.html')

@app.route("/intro")
def intro():
    first_level = FirstLevel.query.all()
    return render_template("intro.html",first_level=first_level)

@app.route("/gen_lin_models")
def gen_lin_models():
    return render_template("gen_lin_models.html")


@app.route("/lin_disc_ana")
def lin_disc_ana():
    return render_template("lin_disc_ana.html")

@app.route("/kernel_ridge")
def kernel_ridge():
    return render_template("kernel_ridge.html")

@app.route("/svm")
def svm():
    return render_template("svm.html")


@app.route("/sto_grad_descent")
def sto_grad_descent():
    return render_template("sto_grad_descent.html")


@app.route("/near_neig")
def near_neig():
    return render_template("near_neig.html")


@app.route("/gauss_process")
def gauss_process():
    return render_template("gauss_process.html")


@app.route("/cross_validation")
def cross_validation():
    return render_template("cross_validation.html")


@app.route("/naive_bayes")
def naive_bayes():
    return render_template("naive_bayes.html")


@app.route("/decision_tree")
def decision_tree():
    return render_template("decision_tree.html")

@app.route("/ensemble_methods")
def ensemble_methods():
    return render_template("ensemble_methods.html")

@app.route("/multiclass_multilabel_algos")
def multiclass_multilabel_algos():
    return render_template("multiclass_multilabel_algos.html")


@app.route("/feature_selection")
def feature_selection():
    return render_template("feature_selection.html")

@app.route("/semi_super")
def semi_super():
    return render_template("semi_super.html")

@app.route("/isotonic_regression")
def isotonic_regression():
    return render_template("isotonic_regression.html")

@app.route("/probability_calibration")
def probability_calibration():
    return render_template("probability_calibration.html")

@app.route("/neural_network")
def neural_network():
    return render_template("neural_network.html")

@app.route("/nlp",methods=['GET','POST'])
def nlp():
    #languages = ['Telugu', 'Hindi', 'German', 'Kannada','Spanish']
    #l_translate = language_translate()
    #if l_translate.validate_on_submit():
        #print("Entered the nlp section")
    #    l_translate.output_lang_text.data = main_translate_function(l_translate.input_lang_text.data)
    #    l_translate.output_lang_text.data = l_translate.output_lang_text.data.split('<end>')[0]
    #    print(l_translate.output_lang_text.data)
        #text = request.form['text']
    #processed_text = text.upper()
    #processed_text = 'This is test'
    #return render_template('nlp.html',processed_text=processed_text,l_translate=l_translate)
    return render_template('nlp.html')

@app.route("/basics")
def basics():
    return render_template("basics.html")

@app.route("/complex")
def complex():
    return render_template("complex.html")

@app.route("/pca")
def pca():
    return render_template("pca.html")

@app.route("/knn")
def knn():
    legend1 = 'Monthly Data'
    #labels1 = ["January","February","March","April","May","June","July","August"]
    labels1 = "January"
    #values1 = [10,9,8,7,6,4,7,8]
    values1 = '10'
    return render_template("knn.html",values1=values1, labels1=labels1, legend1=legend1)

@app.route("/logistic_regression")
def logistic_regression():
    return render_template("logistic_regression.html")


@app.route("/Book", defaults={'subheading':'Book'})
@app.route("/Book/<string:subheading>")
def Book(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'LoadingData':
        return render_template("LoadingData.html",first_level=first_level)
    elif subheading== 'UnderstandData':
        return render_template("UnderstandData.html",first_level=first_level)
    elif subheading== 'VizualizeData':
        return render_template("VizualizeData.html",first_level=first_level)
    elif subheading== 'PrepareData':
        return render_template("PrepareData.html",first_level=first_level)
    elif subheading== 'FeatureSelection':
        return render_template("FeatureSelection.html",first_level=first_level)        
    elif subheading== 'Resampling':
        return render_template("Resampling.html",first_level=first_level)
    elif subheading== 'PerformanceMetrics':
        return render_template("PerformanceMetrics.html",first_level=first_level)
    elif subheading== 'SpotCheckClassification':
        return render_template("SpotCheckClassification.html",first_level=first_level)
    elif subheading== 'RegressionAlgorithm':
        return render_template("RegressionAlgorithm.html",first_level=first_level)
    elif subheading== 'CompareAlgos':
        return render_template("CompareAlgos.html",first_level=first_level)
    elif subheading== 'AutomateML':
        return render_template("AutomateML.html",first_level=first_level)
    elif subheading== 'Ensembles':
        return render_template("Ensembles.html",first_level=first_level)
    elif subheading== 'AlgorithmTuning':
        return render_template("AlgorithmTuning.html",first_level=first_level)                                                        
    elif subheading== 'SaveAndLoad':
        return render_template("SaveAndLoad.html",first_level=first_level)                                                        
    elif subheading== 'TypesOfMLProblems':
        return render_template("TypesOfMLProblems.html",first_level=first_level)                                                        
    else:
        return render_template("lstm.html",first_level=first_level)


@app.route("/Problems", defaults={'subheading':'Problems'})
@app.route("/Problems/<string:subheading>")
def Problems(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'WOP':
        return render_template("WOP.html",first_level=first_level)
    elif subheading== 'COG':
        return render_template("COG.html",first_level=first_level)
    elif subheading== 'DAH':
        return render_template("DAH.html",first_level=first_level)
    elif subheading== 'GWC':
        return render_template("GWC.html",first_level=first_level)
    elif subheading== 'CAR':
        return render_template("CAR.html",first_level=first_level)        
    elif subheading== 'TUG':
        return render_template("TUG.html",first_level=first_level)
    else:
        return render_template("lstm.html",first_level=first_level)


@app.route("/Clustering", defaults={'subheading':'Clustering'})
@app.route("/Clustering/<string:subheading>")
def Clustering(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'KMeans':
        return render_template("KMeans.html",first_level=first_level)
    elif subheading== 'AnityPropogation':
        return render_template("AnityPropogation.html",first_level=first_level)
    elif subheading== 'MeanShift':
        return render_template("MeanShift.html",first_level=first_level)
    elif subheading== 'SpectralClustering':
        return render_template("SpectralClustering.html",first_level=first_level)
    elif subheading== 'WardHierarchialClustering':
        return render_template("WardHierarchialClustering.html",first_level=first_level)        
    elif subheading== 'AgglomerativeClustering':
        return render_template("AgglomerativeClustering.html",first_level=first_level)
    elif subheading== 'DBScan':
        return render_template("DBScan.html",first_level=first_level)
    elif subheading== 'GaussianMixtures':
        return render_template("GaussianMixtures.html",first_level=first_level)
    elif subheading== 'Brich':
        return render_template("Brich.html",first_level=first_level)                        
    else:
        return render_template("lstm.html",first_level=first_level)


@app.route("/DimensionalReduction", defaults={'subheading':'DimensionalReduction'})
@app.route("/DimensionalReduction/<string:subheading>")
def DimensionalReduction(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'PrincipalComponentAnalysis':
        return render_template("PrincipalComponentAnalysis.html",first_level=first_level)
    elif subheading== 'SingularValueDecomposition':
        return render_template("SingularValueDecomposition.html",first_level=first_level)
    elif subheading== 'FactorAnalysis':
        return render_template("FactorAnalysis.html",first_level=first_level)
    elif subheading== 'IndependentComponentAnalysis':
        return render_template("IndependentComponentAnalysis.html",first_level=first_level)
    elif subheading== 'DictionaryLearning':
        return render_template("DictionaryLearning.html",first_level=first_level)        
    elif subheading== 'LatentDirichletAllocation':
        return render_template("LatentDirichletAllocation.html",first_level=first_level)
    else:
        return render_template("lstm.html",first_level=first_level)


@app.route("/ModelSelection", defaults={'subheading':'ModelSelection'})
@app.route("/ModelSelection/<string:subheading>")
def ModelSelection(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'TuningParameters':
        return render_template("TuningParameters.html",first_level=first_level)
    elif subheading== 'ModelEvaluation':
        return render_template("ModelEvaluation.html",first_level=first_level)
    else:
        return render_template("lstm.html",first_level=first_level)

@app.route("/DeepLearning", defaults={'subheading':'DeepLearning'})
@app.route("/DeepLearning/<string:subheading>")
def DeepLearning(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'RNN':
        return render_template("rnn.html",first_level=first_level)
    elif subheading== 'CNN':
        return render_template("cnn.html",first_level=first_level)
    else:
        return render_template("lstm.html",first_level=first_level)


@app.route("/Regression", defaults={'subheading':'regression'})
@app.route("/Regression/<string:subheading>")
def regression(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'LogisticRegression':
        return render_template("logistic_regression.html",first_level=first_level)
    elif subheading== 'KNN':
        legend1 = 'Monthly Data'
        labels1 = ["January","February","March","April","May","June","July","August"]
        #labels1 = "January"
        values1 = [10,9,8,7,6,4,7,8]
        #values1 = '10'
        return render_template("knn.html",first_level=first_level,values1=values1, labels1=labels1, legend1=legend1)
    else:
        return render_template("complex.html",first_level=first_level)


@app.route("/classifiers", defaults={'subheading':'classifier'})
@app.route("/classifiers/<string:subheading>")
def classifiers(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'LinearDiscriminant':
        return render_template("lin_disc_ana.html",first_level=first_level)
    elif subheading== 'KernelRidge':
        return render_template("kernel_ridge.html",first_level=first_level)
    elif subheading== 'NaiveBayes':
        return render_template("naive_bayes.html",first_level=first_level)        
    elif subheading== 'SVM':
        return render_template("svm.html",first_level=first_level)        
    elif subheading== 'DecisionTrees':
        return render_template("decision_tree.html",first_level=first_level)        
    elif subheading== 'BoostedTrees':
        return render_template("boosted_trees.html",first_level=first_level)  
    elif subheading== 'RandomForest':
        return render_template("random_forest.html",first_level=first_level)               
    else:
        return render_template("complex.html",first_level=first_level)


@app.route("/introduction", defaults={'subheading':'intro'})
@app.route("/introduction/<string:subheading>")
def introduction(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'Basic':
        return render_template("basics.html",first_level=first_level)
    else:
        return render_template("complex.html",first_level=first_level)



@app.route("/gan", defaults={'subheading':'gan'})
@app.route("/gan/<string:subheading>")
def gan(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'DCGAN':
        return render_template("gan_process/dcgan.html", first_level=first_level)
    elif subheading == 'CYCLEGAN':
        return render_template("gan_process/cyclegan.html", first_level=first_level)
    elif subheading == 'ADVERSARIAL_FGSM':
        return render_template("gan_process/adversarial_fgsm.html", first_level=first_level)
    elif subheading == 'DEEPDREAM':
        return render_template("gan_process/deepdream.html", first_level=first_level)
    elif subheading == 'PIX2PIX':
        return render_template("gan_process/pix2pix.html", first_level=first_level)
    elif subheading == 'VARIATIONAL_AUTOENCODER':
        return render_template("gan_process/variational_autoencoder.html", first_level=first_level)
    elif subheading == 'NUERAL_STYLE_TRANSFERR':
        return render_template("gan_process/neural_style_transfer.html", first_level=first_level)
    else:
        return render_template("gan.html", first_level=first_level)

@app.route("/NLP", defaults={'subheading':'NLP'})
@app.route("/NLP/<string:subheading>")
def NLP(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'WordEmbeddings':
        return render_template("WordEmbeddings.html",first_level=first_level)
    elif subheading== 'SentimentAnalysis':
        return render_template("SentimentAnalysis.html",first_level=first_level)
    elif subheading== 'TextGenRNN':
        return render_template("TextGenRNN.html",first_level=first_level)
    elif subheading== 'MachineTranslation':
        return render_template("MachineTranslation.html",first_level=first_level)
    elif subheading== 'ImageCaptioning':
        return render_template("ImageCaptioning.html",first_level=first_level)        
    elif subheading== 'LanguageUnderstanding':
        return render_template("LanguageUnderstanding.html",first_level=first_level)
    elif subheading== 'BookChatbot':
        return render_template("BookChatbot.html",first_level=first_level)        
    else:
        return render_template("lstm.html",first_level=first_level)

@app.route("/Stats", defaults={'subheading':'Stats'})
@app.route("/Stats/<string:subheading>")
def Stats(subheading):
    first_level = FirstLevel.query.all()
    if subheading == 'BasicStats':
        return render_template("basicstats.html",first_level=first_level)
    else:
        return render_template("SentimentAnalysis.html",first_level=first_level)

@app.route("/start/<string:heading>")
def start(heading):
    first_level = FirstLevel.query.all()
    if heading == 'Introduction':
        return render_template("intro.html",first_level=first_level)
    elif heading == 'Classifiers':
        return render_template("classifier.html",first_level=first_level)
    elif heading == 'Regression':
        return render_template("regression.html",first_level=first_level)
    elif heading == 'DeepLearning':
        return render_template("deeplearning.html",first_level=first_level)
    elif heading == 'ModelSelection':
        return render_template("modelselection.html",first_level=first_level)
    elif heading == 'DimensionalReduction':
        return render_template("dimensionalreduction.html",first_level=first_level)
    elif heading == 'Clustering':
        return render_template("clustering.html",first_level=first_level)
    elif heading == 'Problems':
        return render_template("problems.html",first_level=first_level)
    elif heading == 'OpenQuestions':
        return render_template("openquestions.html",first_level=first_level)
    elif heading == 'NLP':
        return render_template("book_nlp.html",first_level=first_level)
    elif heading == 'Book':
        return render_template("book.html",first_level=first_level)
    elif heading == 'Stats':
        return render_template("stats.html",first_level=first_level)        
    else:
        return render_template('intro.html',first_level=first_level)







@app.route("/ica")
def ica():
    return render_template("ica.html")
