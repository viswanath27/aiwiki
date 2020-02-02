import os

count = 0
total_file_count = 0
image_path = []
image_count = []
image_tag = []

def PackageData(packageoppath,train_vs_test):
    global image_path
    global image_count
    global image_tag
    min_data = min(image_count)
    dir_create_cmd = "mkdir "+packageoppath+"/data"
    print(dir_create_cmd)
    os.system(dir_create_cmd)
    dir_create_cmd = "mkdir "+packageoppath+"/data/train"
    print(dir_create_cmd)
    os.system(dir_create_cmd)    
    dir_create_cmd = "mkdir "+packageoppath+"/data/validation"
    print(dir_create_cmd)    
    os.system(dir_create_cmd)    
    if (train_vs_test== 1):
        train_data_count = min_data*.8
    elif (train_vs_test== 2):
        train_data_count = min_data*.7
    #elif (train_vs_test== 3):
    else:
        train_data_count = min_data*.6
    train_data_count = round(train_data_count)
    validation_data_count = min_data - train_data_count
    #count = 0
    for x in range(len(image_count)):
    #for x in image_path:
        floder_create = "mkdir "+packageoppath+"/data/train/"+image_tag[x]
        print(floder_create)
        os.system(floder_create)
        floder_create = "mkdir "+packageoppath+"/data/validation/"+image_tag[x]
        print(floder_create)
        os.system(floder_create)        
        count = 0;
        files_int = []
        files_str = []
        for root, dirs, files in os.walk(image_path[x]):
            print (files)
            for file in files:
                file = file.rstrip('.png')
                files_int.append(int(file))
            files_int.sort()
            print 
            for file in files_int:
                files_str.append(str(file)+'.png')
            print (files_str)
            for filename in files_str:
                if ( count <= train_data_count):
                    cmd = "cp "+image_path[x]+"/"+filename+" "+packageoppath+"/data/train/"+image_tag[x]+"/"+image_tag[x]+filename
                    print(cmd)
                    os.system(cmd)
                    if ( image_tag[x] == 'lung'):
                        resize_cmd = "convert -size 1024x1024 "+packageoppath+"/data/train/"+image_tag[x]+"/"+image_tag[x]+filename+" -resize 600x600 "+packageoppath+"/data/train/"+image_tag[x]+"/"+image_tag[x]+filename
                        os.system(resize_cmd)
                    else:
                        resize_cmd = "convert -size 700x605 "+packageoppath+"/data/train/"+image_tag[x]+"/"+image_tag[x]+filename+" -resize 600x600 "+packageoppath+"/data/train/"+image_tag[x]+"/"+image_tag[x]+filename
                        os.system(resize_cmd)
                else:
                    cmd = "cp "+image_path[x]+"/"+filename+" "+packageoppath+"/data/validation/"+image_tag[x]+"/"+image_tag[x]+filename
                    print(cmd)
                    os.system(cmd)
                    if ( image_tag[x] == 'lung'):
                        resize_cmd = "convert -size 1024x1024 "+packageoppath+"/data/validation/"+image_tag[x]+"/"+image_tag[x]+filename+" -resize 600x600 "+packageoppath+"/data/validation/"+image_tag[x]+"/"+image_tag[x]+filename
                        os.system(resize_cmd)
                    else:
                        resize_cmd = "convert -size 700x605 "+packageoppath+"/data/validation/"+image_tag[x]+"/"+image_tag[x]+filename+" -resize 600x600 "+packageoppath+"/data/validation/"+image_tag[x]+"/"+image_tag[x]+filename
                        os.system(resize_cmd)                    
                count = count +1
                if (count >= min_data):
                    break

    #find the count of files 

    print ("Package function invoked")

def MNISTFormat():
    global image_path
    global image_count
    global image_tag    
    min_data = min(image_count)
    #labelsAndFiles = get_labels_and_files(argv[1], int(argv[2]))
    labelsAndFiles = get_labels_and_files(image_path, min_data)
    random.shuffle(labelsAndFiles)
    imagedata, labeldata = make_arrays(labelsAndFiles)
    #write_labeldata(labeldata, argv[3])
    write_labeldata(labeldata, "labels")
    #write_imagedata(imagedata, argv[4])
    write_imagedata(imagedata, "images")
    cmd = "gzip"+" "+image_path+"labels"
    os.system(cmd)
    cmd = "gzip"+" "+image_path+"images"
    os.system(cmd)

def getListOfFilescount(ipdir):
    global count
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(ipdir)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(ipdir, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFilescount(fullPath)
        else:
            #print (fullPath)
            #fullPath = fullPath.replace(" ", '\ ')
            #print(fullPath)
            #print (count)
            #count = count + 1           
            #cmd = 'cp '+fullPath+' '+opdir+str(count)+'.dcm'
            #print (cmd)
            #os.system(cmd)
            allFiles.append(fullPath)
    return allFiles


'''
For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(ipdir,opdir):
    global count
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(ipdir)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(ipdir, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath,opdir)
        else:
            #print (fullPath)
            fullPath = fullPath.replace(" ", '\ ')
            #print(fullPath)
            #print (count)
            count = count + 1           
            cmd = 'cp '+fullPath+' '+opdir+str(count)+'.dcm'
            print (cmd)
            os.system(cmd)
            #yield "data:" + str(count) + "\n\n"
            allFiles.append(fullPath)
    return allFiles 

def ConvertDICOM2PNG(ConvertInputPath,ConvertOutputPath,fileformat,imagelabel):
    global image_path
    global image_count
    global image_tag
    print(ConvertInputPath)
    print(ConvertOutputPath)
    print(fileformat)
    #cmd = 'dcmj2pnm --write-'+fileformat #000000.dcm 1.png'
    i=0
    for root, dirs, files in os.walk(ConvertInputPath):
        for filename in files:
            #print(filename)
            #cmd = 'dcmj2pnm --write-'+fileformat
            #cmd = cmd+' '+ConvertInputPath+'/'+filename+' '+ConvertOutputPath+'/'+str(i)+'.'+fileformat
            cmd = 'convert'
            cmd = cmd+' '+ConvertInputPath+'/'+filename+' '+ConvertOutputPath+'/'+str(i)+'.'+fileformat
            #cmd = cmd+' '+ConvertInputPath+'/'+filename+' '+ConvertOutputPath+'/'+str(i)
            #cmd = 'mv'+' '+ConvertInputPath+'/'+filename+' '+ConvertOutputPath+'/eye'+str(i)+'.png'
            print (cmd)
            os.system(cmd)

            resize_cmd = "convert "+ConvertOutputPath+'/'+str(i)+'.'+fileformat+" -resize 512x512 "+ConvertOutputPath+'/'+imagelabel+str(i)+'.'+fileformat
            #os.system(resize_cmd)
            i = i+1

    image_path.append(ConvertOutputPath)
    image_count.append(i)
    image_tag.append(imagelabel)
    print(image_path)
    print(image_count)
    print(image_tag)
