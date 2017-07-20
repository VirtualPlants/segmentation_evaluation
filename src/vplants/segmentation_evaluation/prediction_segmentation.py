from sklearn import datasets

import pandas as pd
import numpy as np
from openalea.core.world import World
#import datadotworld-py-master as world
import pickle
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn import svm

from openalea.core.world import World

from sklearn import metrics,linear_model,neighbors,datasets



#import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sm
try:
    from timagetk.components import SpatialImage,imsave,imread
except ImportError:
  
    raise ImportError('Import Error')
np_float=np.float
np_reshape=np.reshape
class Prediction(object):
    """
    classe possedant 2 attributs
    - image 1
    - image 2
    """
    def __init__(self,img_seg,img_gray,fichier_expert,fichier_image):
        self.world = World()
        self.fichier_expert,self.fichier_image=fichier_expert,fichier_image
        self.img_seg, self.img_gray =img_seg, img_gray
        # self.img_1_maillage,self.img_1_er_maillage =PropertySpatialImage(img_1),PropertySpatialImage(img_1_er) 
        # self.img_1_maillage.compute_image_property('volume')
        
        conds_img = (isinstance(img_seg,SpatialImage) and isinstance(img_gray,SpatialImage))
                

        if conds_img:
            self.shape_seg, self.shape_gray = img_seg.get_shape(),img_gray.get_shape()

            if self.shape_seg==self.shape_gray :
                self.arr_seg, self.arr_gray = img_seg.get_array().astype(np_float), img_gray.get_array().astype(np_float)
                self.vox_seg,self.vox_gray = img_seg.get_voxelsize(),img_gray.get_voxelsize()
                
                if self.img_seg.get_dim()==2:
                    self.len1D = self.shape_seg[0]*self.shape_seg[1]
                   
                elif self.img_seg.get_dim()==3:
                    self.len1D = self.shape_seg[0]*self.shape_seg[1]*self.shape_seg[2]
                    
                self.sig_seg_1D,self.sig_gray_1D = np_reshape(self.arr_seg,(self.len1D,1)),np_reshape(self.arr_gray,(self.len1D,1))              
                self.iter1D = xrange(self.len1D)    
                self.nbre_cellules=len(np.unique(self.sig_seg_1D))
                
                self.etiq_max=int(np.max(self.sig_seg_1D))
                
                
                self.labels=np.unique(self.sig_seg_1D)
                
                print self.nbre_cellules

               
            else: 
                print('TODO')               
        
        print ('tt')
    
    def read_file(self,fichier):
        
        with open(fichier,"r") as fich:
            p=pickle.Unpickler(fich)
      
            titre_image=p.load()
            labels=p.load()
            indication=p.load()
            volume=p.load()
            #moyenne=p.load()
            variance=p.load()
            #entropie=p.load()
       

        #titre_image,labels,indication,volume,moyenne,variance,entropie=np.array(titre_image),np.array(labels,np.int16),np.array(indication,np.int8),np.array(volume),np.array(moyenne),np.array(variance),np.array(entropie)
        titre_image,labels,indication,volume,variance=np.array(titre_image),np.array(labels,np.int16),np.array(indication,np.int8),np.array(volume),np.array(variance)
        print np.shape(titre_image), np.shape(labels), np.shape(indication),np.shape(volume),np.shape(variance)
        #print ('4',np.max(volume),np.min(volume),volume[np.where(labels==1)])
        volume=volume[np.where(labels!=1)]
        #moyenne=moyenne[np.where(labels!=1)]
        variance=variance[np.where(labels!=1)]
        #entropie=entropie[np.where(labels!=1)]
        labels=labels[np.where(labels!=1)]
        indication=indication[np.where(labels!=1)]
        titre_image=titre_image[np.where(labels!=1)]
        #print ('41',np.max(volume),np.min(volume),volume[np.where(labels==1)])
        return titre_image,labels,indication,volume,variance
    
    def write_file(self,fichier,titre_image,labels,indication,volume,variance):
        
        with open(fichier,"w") as fich:
            p=pickle.Pickler(fich)
            p.dump(titre_image)
            p.dump(labels)
            p.dump(indication)
            p.dump(volume)
            #p.dump(moyenne)
            p.dump(variance)
           # p.dump(entropie)
        return fichier
    
    def normalise_value(self,val, input_range,output_range):
        """
        """

  
        if isinstance(input_range,list) and len(input_range)==2:
            min_res, max_res =output_range[0], output_range[1]
            val_min, val_max = input_range[0], input_range[1]
         
            val_res=(max_res-min_res)*float((val-val_min))/(val_max-val_min)+min_res
         
            return val_res
        else:
            print('fr')
        return val_res
    
    def compute_base_apprentissage(self,fichier):
        titre_image,labels,indication,volume,moyenne,variance,entropie=self.read_file(fichier)
        base=np.concatenate((volume,variance,indication))

        base=np.reshape(base,(3,len(indication)))
        base=np.transpose(base)

        base_X=base[0::,0:2]
        base_y=base[:,2]
        np.random.seed(0)
        indices=np.random.permutation(len(base_X))
        nb_indice_test= 100
     
        self.base_X_train=base_X[indices[:-nb_indice_test]]
        self.base_y_train=base_y[indices[:-nb_indice_test]]
        self.image_train=titre_image[indices[:-nb_indice_test]]
        self.labels_train=labels[indices[:-nb_indice_test]]
        self.base_X_test=base_X[indices[-nb_indice_test:]]
        self.base_y_test=base_y[indices[-nb_indice_test:]]
        self.image_test=titre_image[indices[-nb_indice_test:]]
        self.labels_test=labels[indices[-nb_indice_test:]]
        return titre_image,labels
        #return titre_image,labels,base_X, base_y
    def compute_marge_min_max(self,fichier_info):
        print('1')
        titre_image,labels,indication,volume,moyenne,variance,entropie=self.read_file(fichier_info)
        
        volume_max,volume_min=np.max(volume),np.min(volume)
        #print ('vv',volume_max,volume_min)
        variance_max,variance_min=np.max(variance),np.min(variance)
        moyenne_max,moyenne_min=np.max(moyenne),np.min(moyenne)
        entropie_max,entropie_min=np.max(entropie),np.min(entropie)
       
        input_range_vol,input_range_var,input_range_moy,input_range_ent=[volume_min,volume_max],[variance_min,variance_max],[ moyenne_min,moyenne_max],[entropie_min,entropie_max]
        
        return input_range_vol,input_range_var,input_range_moy,input_range_ent
    
    def compute_normalise_variables(self,fichier_info,fichier_norm):
        print('2')
        titre_image,labels,indication,volume,moyenne,variance,entropie=self.read_file(fichier_info)
        input_range_vol,input_range_var,input_range_moy,input_range_ent=self.compute_marge_min_max(fichier_info)
        output_range_vol,output_range_var,output_range_moy,output_range_ent=self.compute_marge_min_max(self.fichier_expert)
        volume_norm,moyenne_norm,variance_norm,entropie_norm=volume,moyenne,variance,entropie
        print volume[1:10]
       
        
        ind=0
        for moy,var,ent,vol in zip(moyenne,variance,entropie,volume):
           
           # print volume_norm[ind]
            variance_norm[ind]=self.normalise_value(var,input_range_var,output_range_var)
            moyenne_norm[ind]=self.normalise_value(moy,input_range_moy,output_range_moy)
            entropie_norm[ind]=self.normalise_value(ent,input_range_ent,output_range_ent)
            volume_norm[ind]=self.normalise_value(vol,input_range_vol,output_range_vol)
            ind+=1
        volume_max,volume_min=np.max(volume_norm),np.min(volume_norm)
       # print ('vv',volume_max,volume_min)
        print volume_norm[1:10]
        self.write_file(fichier_norm,titre_image,labels,indication,volume,moyenne,variance,entropie)
        return fichier_norm

    def compute_base_apprentissage1(self,fichier_info):
        print('3')
        #titre_image,labels,indication,volume,moyenne,variance,entropie=self.read_file(fichier_info)
        
        #fichier_info_norm=self.compute_normalise_variables(fichier_info,fichier_info_norm)
        titre_image,labels,indication,volume_norm,variance_norm=self.read_file(fichier_info)
    
        base=np.concatenate((volume_norm,variance_norm,indication))
        #base=np.concatenate((volume_norm,moyenne_norm,variance_norm,entropie_norm,indication))

        base=np.reshape(base,(3,len(indication)))
        base=np.transpose(base)

        base_X=base[0::,0:2]
        base_y=base[:,2]


        return titre_image,labels,base_X, base_y  
    
    def compute_SVM(self,ind_image,kernel='linear',img_expert=None):
        self.titre_image_train,self.labels_train,self.base_X_train,self.base_y_train=self.compute_base_apprentissage1(self.fichier_expert)
        self.titre_image_test,self.labels_test,self.base_X_test,self.base_y_predict=self.compute_base_apprentissage1(self.fichier_image)
        self.base_y_predict=[-1]*len(self.labels_test)

        svc=svm.SVC(kernel=kernel,C=10000000000,gamma=0.00000000000000068,max_iter=7500)
        svc.fit(self.base_X_train,self.base_y_train)

        self.base_y_predict=svc.predict(self.base_X_test)
      
        print ('SVM kernel: ',kernel,svc)
        print ('nb BS',len(self.base_y_predict[np.where(self.base_y_predict==1)]))
        print self.labels_test[np.where(self.base_y_predict==1)]
        print ('nb GS',len(self.base_y_predict[np.where(self.base_y_predict==2)]))

        #titre_image,labels,indication,volume,moyenne,variance,entropie=self.read_file(self.fichier_image)
        titre_image,labels,indication,volume,variance=self.read_file(self.fichier_image)
        self.write_file(self.fichier_image,titre_image,labels,self.base_y_predict,volume,variance)

        labels_bad_cell_sure,labels_bad_cell_doute=self.compute_cell_interet()
        
        out_gray,out_seg=self.plot_cell_interet(labels_bad_cell_sure,labels_bad_cell_doute)
        x_seg='j1_test_fusion_SVM_'+kernel+'_time_'+str(ind_image)+'_seg.inr'
        x_gray='j1_test_fusion_SVM_'+kernel+'_time_'+str(ind_image)+'.inr'
        imsave('/home/sophie/.openalea/projects/Segmentation_Evaluation/data/informations_images/'+x_seg,out_seg) 
        imsave('/home/sophie/.openalea/projects/Segmentation_Evaluation/data/informations_images/'+x_gray,out_gray)  
 
        return 

    
    def compute_statistique(self,ind_image):
        #Evaluation:
        #print np.shape(self.titre_image_train),np.shape(self.base_y_train)
        labels_train_eval=self.labels_train[np.where(self.titre_image_train=='time_'+str(ind_image)+'.inr')]

        indication_train_eval=self.base_y_train[np.where(self.titre_image_train=='time_'+str(ind_image)+'.inr')]
        ind=0
        indication_test_eval=np.ones(np.shape(indication_train_eval))*(-1)

        for ind,label in enumerate(labels_train_eval):
            indication_test_eval[ind]=self.base_y_predict[np.where(self.labels_test==label)]
            #ind+=1
            
        #print ('label',labels_train_eval)
        #print('predict',indication_test_eval)
        #print('expert',indication_train_eval)

        data_TP_TN=indication_train_eval[np.where(indication_train_eval==indication_test_eval)]
        labels_TP_TN=labels_train_eval[np.where(indication_train_eval==indication_test_eval)]
        TN=len(data_TP_TN[np.where((data_TP_TN==1) | (data_TP_TN==0))])
        labels_TN=labels_TP_TN[np.where((data_TP_TN==1) | (data_TP_TN==0))]
        TP=len(data_TP_TN[np.where(data_TP_TN==2)])
        labels_TP=labels_TP_TN[np.where(data_TP_TN==2)] 
        TP_TN=len(data_TP_TN)

        #print ('labels_TP: ', labels_TP)
       # print ('labels_TN: ',labels_TN)
        print ('TN,TP,TP_TN',TN,TP,TP_TN)
        
        data_FP_FN=indication_train_eval[np.where(indication_train_eval!=indication_test_eval)]
        labels_FP_FN=labels_train_eval[np.where(indication_train_eval!=indication_test_eval)]
        FP=len(data_FP_FN[np.where((data_FP_FN==1) | (data_FP_FN==0))])
        labels_FP=labels_FP_FN[np.where((data_FP_FN==1) | (data_FP_FN==0))]
        FN=len(data_FP_FN[np.where(data_FP_FN==2)])
        labels_FN=labels_FP_FN[np.where(data_FP_FN==2)] 
        FP_FN=len(data_FP_FN)
        #print ('labels_FP: ', labels_FP)
        print ('labels_FN: ',np.where(labels_train_eval==192))
        print ('FN,FP,FP_FN',FN,FP,FP_FN)
        TP_rate=float(TP)/(TP+FN) 
        TN_rate=float(TN)/(TN+FP)
        accuracy=float(TP+TN)/(TP+TN+FP+FN)
        print ('TP_rate: ',TP_rate)
        print ('TN_rate: ',TN_rate) 
        print ('Accuracy: ',accuracy)
        
        
        print pd.crosstab(indication_train_eval,indication_test_eval)
        #print metrics.confusion_matrix(y_true=indication_train_eval,y_pred=indication_test_eval)
        #print metrics.classification_report(y_true=indication_train_eval,y_pred=indication_test_eval)
        return
    
####Compute bad segmented cell 
    def compute_cell_interet(self):
        print np.shape(self.base_y_predict),np.shape(self.labels_test)
        labels_bad_cell_sure=self.labels_test[np.where(self.base_y_predict==1)]
        
        labels_bad_cell_doute=self.labels_test[np.where(self.base_y_predict==0)]
        print labels_bad_cell_sure,labels_bad_cell_doute
        return labels_bad_cell_sure,labels_bad_cell_doute

#### Image des cellules bad segmented
    def plot_cell_interet(self,labels_bad_sure,labels_bad_doute):

        sig_seg=255*np.ones((self.len1D,1),dtype=np.int16)
        #print len(labels_bad_sure),len(labels_bad_doute)
        
        sig_seg[np.where(self.sig_seg_1D==1)]=1
        sig_gray=self.sig_gray_1D
        sig_gray[np.where(self.sig_seg_1D==1)]=1
        for val_sure in labels_bad_sure:
        #for val_sure in labels_bad_sure:
       
            ind_sure=np.where(self.sig_seg_1D==val_sure)
            
            val_sure=int(val_sure)
            # ind_sure=np.where(self.sig_seg_1D==val_sure)
            # 
            # val_sure=int(val_sure)
           
            #print ('label: ', val,' variance: ',var_intra_par_cell[val-2] ,'volume: ',volume_cell[val-2] )
            sig_seg[ind_sure]=val_sure
       
            sig_gray[ind_sure]=200
            # sig_seg[ind_sure]=val_sure
            # sig_gray[ind_sure]=200

        out_seg = np_reshape(sig_seg.astype(np.uint16),self.shape_seg)
        out_gray = np_reshape(sig_gray.astype(np.uint16),self.shape_seg)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_seg)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_gray)
        
        return out_gray,out_seg     
    
    
if __name__ == '__main__':
    
    def data_path_seg_er (img_name):
        pwd='/home/sophie/dev/data_1/segmentation/affine_registration/segmentation_erosion_'+str(rad)+'/'
        out=''.join([pwd,img_name])
        return out
    def data_path_seg (img_name):
        pwd='/home/sophie/dev/data_1/segmentation/affine_registration/'
        out=''.join([pwd,img_name])
        return out
    def data_path_recal (img_name):
        pwd='/home/sophie/dev/data_1/grayscale/affine_registration/'
        out=''.join([pwd,img_name])
        return out
    rad=1
       
    i=6
    fich_inf='affine_time_'+str(i)+'_er'+str(rad)+'_seg_norm'
    fich_inf='j1_test_fusion1_normalise_time_6_er1_seg'
    fich_inf_norm='time'+str(i)+'_er'+str(rad)+'_seg_norm'
    x_seg='time_'+str(i)+'_er'+str(rad)+'_seg.inr'
    x_seg='test_fusion_time_6.inr'
    x_seg_orig='time_'+str(i)+'_seg.inr'
    img_seg = imread(data_path_seg_er(x_seg))
    img_seg_orig = imread(data_path_seg(x_seg_orig))
    print len(np.unique(img_seg_orig))
    y_gray='time_'+str(i)+'.inr'    
    img_gray = imread(data_path_recal(y_gray))
    S=1000
    fichier_expert='/home/sophie/.openalea/projects/Segmentation_Evaluation/data/test_echantillon/base_initiale_er1_05_norm.txt'
    fichier_expert_norm='/home/sophie/.openalea/projects/Segmentation_Evaluation/data/resultats_expertises/resultats_expertises_er'+str(rad)+'_'+str(S)+'.txt'
    #fichier='/home/sophie/.openalea/projects/Segmentation_Evaluation/data/resultats_expertises/fusion_er'+str(rad)+'_'+str(S)+'.txt'
    fichier_image='/home/sophie/.openalea/projects/Segmentation_Evaluation/data/informations_images/'+fich_inf+'.txt'
    fichier_image_norm='/home/sophie/.openalea/projects/Segmentation_Evaluation/data/informations_images/'+fich_inf_norm+'.txt'
    obj_appr=Prediction(img_seg,img_gray,fichier_expert,fichier_image)
    obj_appr.read_file(fichier_expert)
    #obj_appr.compute_base_apprentissage1(fichier_expert,fichier_expert_norm)
    kernel='rbf'
    #obj_appr.compute_SVM(i,kernel,1)
    #kernel='linear'
    #kernel='poly'
    obj_appr.compute_SVM(i,kernel,1)
    
    #obj_appr.compute_SVM(i,kernel,1)

    # n_neighbors=7
   # obj_appr.compute_KNN(i,1,1)
    # obj_appr.compute_KNN(i,2,1)
    # obj_appr.compute_KNN(i,3,1)

   # obj_appr.compute_Regression_Logistic1(i,1)
    #obj_appr.compute_States_Models(i,1)
    #obj_appr.compute_KNN_regressor(i,1)

    
    
