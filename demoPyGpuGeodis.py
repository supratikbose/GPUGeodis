
import os
import sys
import pathlib
import nibabel as nib
import numpy as np
import time
import matplotlib.pyplot as plt
from copy import copy

rootPath = pathlib.Path('.')

def readAndScaleImageData(fileName, folderName, clipFlag, clipLow, clipHigh, scaleFlag, scaleFactor,\
                          meanSDNormalizeFlag, finalDataType, \
                          isLabelData, labels_to_train_list, verbose=False,transpose=False): 
    returnNow = False
    #check file existence
    filePath = os.path.join(folderName, fileName)            
    if os.path.exists(filePath):
        pass
    else:
        print(filePath, ' does not exist')  
        returnNow = True  
    if returnNow:
        sys.exit() 
    #We are here => returnNow = False
    if transpose:
        fileData = np.transpose(nib.load(filePath).get_fdata(), axes=(2,1,0))  
    else:
        fileData = nib.load(filePath).get_fdata()
    #Debug code
    if verbose:
        dataMin = fileData.min()
        dataMax = fileData.max()
        print('fileName - shape - type -min -max: ', fileName, ' ', fileData.shape, ' ', fileData.dtype, ' ', dataMin, ' ', dataMax)
    #Clamp                          
    if True == clipFlag:
        np.clip(fileData, clipLow, clipHigh, out= fileData)
    #Scale   
    if True == scaleFlag:
        fileData = fileData / scaleFactor
    #mean SD Normalization
    if True == meanSDNormalizeFlag:
        fileData = (fileData - np.mean(fileData))/np.std(fileData)
    #Type conversion
    fileData = fileData.astype(finalDataType)
    if True == isLabelData:
        # pick specific labels to train (if training labels other than 1s and 0s)
        if labels_to_train_list != [1]:
            temp = np.zeros(shape=fileData.shape, dtype=fileData.dtype)
            new_label_value = 1
            for lbl in labels_to_train_list: 
                ti = (fileData == lbl)
                temp[ti] = new_label_value
                new_label_value += 1
            fileData = temp
    return fileData

#GPU version
currentDirectory = os.path.dirname(os.path.abspath(__file__))
srcDirectory = os.path.join(currentDirectory, "src")
sys.path.append(srcDirectory)
import gpuGeodis
def generateHintMapFromClicksGPU(imageData, clickSegmentation, spacing, lamb, iter):
    t0 = time.time()
    I = np.asarray(imageData, np.float32)
    if 0!= np.sum(clickSegmentation):
        hintMap = gpuGeodis.gpuGeodesic3d_raster_scan(I, clickSegmentation, spacing, lamb, iter)
    else:
        print('Empty click segmentation => random hintMap.')
        hintMap = np.random.rand(I.shape[0], I.shape[1], I.shape[2] )
    t1 = time.time()
    dt = t1-t0
    print("runtime(s) GPU-Geodis raster scan   {0:}".format(dt))
    return hintMap

# #CPU version using GeodisTK from https://github.com/taigw/GeodisTK
# #Uncomment only if above GeodisTk (CPU)  is installed
# import GeodisTK      
# def generateHintMapFromClicksCPU(imageData, clickSegmentation, spacing, lamb, iter):
#     t0 = time.time()
#     I = np.asarray(imageData, np.float32)
#     if 0!= np.sum(clickSegmentation):
#         hintMap = GeodisTK.geodesic3d_raster_scan(I, clickSegmentation, spacing, lamb, iter)
#     else:
#         print('Empty click segmentation => random hintMap.')
#         hintMap = np.random.rand(I.shape[0], I.shape[1], I.shape[2] )
#     t1 = time.time()
#     dt = t1-t0
#     print("runtime(s) raster scan   {0:}".format(dt))
#     return hintMap

def  demo_GpuGeodesicPyCuda(dataFolder, numChannel, imgFile_ch1, imgFile_ch2, seedFile, outputFile, numIterations, lamb, displayFlag, displaySliceIndex):
    img_ch1_path = dataFolder /  imgFile_ch1   
    print(f'img_ch1_path: {img_ch1_path}')
    #Read np ndarray for CT data
    img_ch1 = readAndScaleImageData(fileName=imgFile_ch1,folderName=str(dataFolder), 
        clipFlag = False, clipLow=-1000, clipHigh =3095,
        scaleFlag=False, scaleFactor=1000,
        meanSDNormalizeFlag = False, finalDataType = np.float32,
        isLabelData=False, labels_to_train_list=None, verbose=False, transpose=False)
    img_ch1_normalized = np.clip(img_ch1, -1024, 1024) / 1024

    img_ch2_path = dataFolder /  imgFile_ch2   
    print(f'img_ch2_path: {img_ch2_path}')
    #Read np ndarray for PET data
    img_ch2 = readAndScaleImageData(fileName=imgFile_ch2,folderName=str(dataFolder), 
        clipFlag = False, clipLow=-10, clipHigh =100,
        scaleFlag=False, scaleFactor=1,
        meanSDNormalizeFlag = False, finalDataType = np.float32,
        isLabelData=False, labels_to_train_list=None, verbose=False, transpose=False)
    img_ch2_normalized =  (img_ch2 - np.mean(img_ch2)) / ( np.std(img_ch2) + 1e-3)

    seed_path = dataFolder / seedFile
    print(f'seed_path: {seed_path}')
    img_seed = readAndScaleImageData(fileName=seedFile,folderName=str(dataFolder), 
        clipFlag = False,clipLow=0, clipHigh = 0,
        scaleFlag=False, scaleFactor=1, meanSDNormalizeFlag = False, finalDataType = np.uint8,
        isLabelData=True, labels_to_train_list=[1], verbose=False, transpose=False)

    img = [img_ch1_normalized, img_ch2_normalized] #, softmaxData <-- Not using as valuse may be close to zero out side prediction
    img = np.stack(img, axis=-1) #(144, 144, 144, 2)
    print(f'img,shape {img.shape}')

    #Obtain nii header from the 1st channel image itself. By assumption both channel images have same pixel spacing and
    #they are registered.
    modelImage_nii = nib.load(os.path.join(str(dataFolder), imgFile_ch1))
    modelImage_nii_aff = modelImage_nii.affine
    modelImage_nii_header = modelImage_nii.header
    spacing_a, spacing_b, spacing_c = modelImage_nii_header.get_zooms()
    modelImage_spacing = [spacing_a, spacing_b, spacing_c] 

    img_ch1_seed = np.ma.masked_where(img_ch1 * img_seed != 0, img_ch1)
    img_ch2_seed = np.ma.masked_where(img_ch2 * img_seed != 0, img_ch2)
    
    plt.rcdefaults()
    palette_r = copy(plt.cm.Greys_r)
    palette_r.set_bad(color='r', alpha=1)  # set color for the  mask  
    palette_b = copy(plt.cm.Greys_r)
    palette_b.set_bad(color='b', alpha=1)  # set color for the mask
    
    geodesicDis_gpu =  generateHintMapFromClicksGPU(imageData=img, clickSegmentation=img_seed, spacing=modelImage_spacing, lamb=lamb, iter=numIterations)
    geodesicDisFilePath_gpu = dataFolder / ('gpu_' + outputFile)
    print(f'geodesicDisFilePath_gpu : {geodesicDisFilePath_gpu}')
    nib.save(nib.Nifti1Image(geodesicDis_gpu, affine=modelImage_nii_aff), geodesicDisFilePath_gpu)

    # #Uncomment only if  GeodisTk (CPU)  is installed
    # geodesicDis_cpu =  generateHintMapFromClicksCPU(imageData=img, clickSegmentation=img_seed, spacing=modelImage_spacing, lamb=lamb, iter=numIterations)
    # geodesicDisFilePath_cpu = dataFolder / ('cpu_' + outputFile)
    # print(f'geodesicDisFilePath_cpu : {geodesicDisFilePath_cpu}')
    # nib.save(nib.Nifti1Image(geodesicDis_cpu, affine=modelImage_nii_aff), geodesicDisFilePath_cpu)


    if displayFlag:
        #Plot
        dist_gpu = np.ma.masked_where(geodesicDis_gpu * img_seed != 0, geodesicDis_gpu)

        # #Uncomment only if above GeodisTk (CPU)  is installed
        # dist_cpu = np.ma.masked_where(geodesicDis_cpu * img_seed != 0, geodesicDis_cpu)

        # Comment out this if uncommenting the next part
        fig, axes = plt.subplots(1, 3, figsize=(14, 10))
        axes[0].imshow(img_ch1_seed[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        axes[0].set_title("img_ch1_seed")
        axes[1].imshow(img_ch2_seed[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        axes[1].set_title("img_ch2_seed")
        axes[2].imshow(dist_gpu[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        axes[2].set_title("dist_GPU")        

        # # #Uncomment only if above GeodisTk (CPU)  is installed
        # fig, axes = plt.subplots(2, 3, figsize=(14, 10)) 
        # axes[0,0].imshow(img_ch1_seed[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        # axes[0,0].set_title("img_ch1_seed")
        # axes[0,1].imshow(img_ch2_seed[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        # axes[0,1].set_title("img_ch2_seed")
        # axes[0,2].imshow(dist_gpu[:, :, displaySliceIndex], cmap=palette_r, interpolation='none')
        # axes[0,2].set_title("dist_GPU")
        # axes[1,0].imshow(img_ch1_seed[:, :, displaySliceIndex], cmap=palette_b, interpolation='none')
        # axes[1,0].set_title("img_ch1_seed")
        # axes[1,1].imshow(img_ch2_seed[:, :, displaySliceIndex], cmap=palette_b, interpolation='none')
        # axes[1,1].set_title("img_ch2_seed")
        # axes[1,2].imshow(dist_cpu[:, :, displaySliceIndex], cmap=palette_b, interpolation='none')
        # axes[1,2].set_title("dist_CPU")

        plt.tight_layout()
        #lt.savefig('./data/example.png')
        plt.show()


if __name__ == '__main__':
    dataFolder = rootPath / 'data'
    #Assumption: multichannel images have same pixel spacing and they are registered
    numChannel = 2
    imgFile_ch1 = 'img_ct.nii.gz'
    imgFile_ch2 = 'img_pt.nii.gz'    
    seedFile = 'img_seed.nii.gz'
    outputFile = 'geodesic_dist.nii.gz'
    numIterations = 4
    # 0.0 <= lamb <= 1.0 where 0.0 is complete Euclidean and 1.0 is complete Geodeis distance
    lamb = 0.1 
    displayFlag = True
    displaySliceIndex = 80
    
    demo_GpuGeodesicPyCuda(dataFolder, numChannel, imgFile_ch1, imgFile_ch2, seedFile, outputFile, numIterations, lamb, displayFlag, displaySliceIndex, )









