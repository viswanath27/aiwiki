import pyodbc
import sys
import numpy
from PIL import Image, ImageFilter  
import base64
import PIL.Image
import io
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
from PIL import ImageEnhance  

img = 0

def resultingImageRead(cnxn):
	print("Reading the resultant file")
	cursor = cnxn.cursor()
	cursor.execute('SELECT pResultImage FROM patientInfo WHERE pId=3')
	data=cursor.fetchall()
	#print (data[0][0])
	#data1=base64.b64decode(data[0][0])
	file_like=io.BytesIO(data[0][0])
	img=PIL.Image.open(file_like)
	img.show()
	#img = img.rotate(180)
	#im1 = img.save("geeks.jpg")
	#return img 


def imagewrite(cnxn,img,patientid):
	print ("Image write")
	cursor = cnxn.cursor()
	# Prepare SQL query to INSERT a record into the database.
	#sql = "UPDATE patientInfo SET pSummary=(?) WHERE pId = 20",('this is sample text')
	try:
	   # Execute the SQL command
	   print("Executing the command")
	   cursor.execute("UPDATE patientInfo SET pResultImage=(?) WHERE pId = (?)",(img,patientid))
	   # Commit your changes in the database
	   print("Commiting the database")
	   cnxn.commit()
	except:
	   # Rollback in case there is any error
	   print("Rolling back due to error")
	   cnxn.rollback()

	# disconnect from server
	print("Closing the db connection")

def imageread(cnxn):
	cursor = cnxn.cursor()
	cursor.execute('SELECT pUploadImage FROM patientInfo WHERE pId=11')
	data=cursor.fetchall()
	#print (data[0][0])
	#data1=base64.b64decode(data[0][0])
	file_like=io.BytesIO(data[0][0])
	#img=PIL.Image.open(file_like)
	#img.show()
	#img = img.rotate(180)
	#im1 = img.save("geeks.jpg")
	#img.show()
	return file_like 

def connection2db():
	#server = 'cancerdetectiondb.database.windows.net'
	server = 'deepdivedb.database.windows.net'
	database = 'deepdivedb'
	#database = 'cancerdetectiondb'
	username = 'deepdive'
	#username = 'cancer'
	password = 'admin@123'
	#password = 'Admin@123'
	driver= '{ODBC Driver 17 for SQL Server}'
	cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
	return cnxn


def dbprocess():
	cnxn = connection2db()
	print("First Read from the Table")
	#readfromdb(cnxn)
	print("First Write to the Table")
	#writetodb(cnxn,"this is new text",19)
	print("First Read from the Table")
	#readfromdb(cnxn)
	print("this is test image read ")
	img = imageread(cnxn)
	#imagewrite(cnxn,img,3)
	#print ("reading the resulting image")
	#resultingImageRead(cnxn)
	cnxn.close()
