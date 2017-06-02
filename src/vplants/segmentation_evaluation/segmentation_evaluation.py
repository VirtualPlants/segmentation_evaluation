# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:42:03 2017

@author: ilhem
"""

# TODO entete

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import datadotworld-py-master as world

try:
    from timagetk.components import SpatialImage,imsave
except ImportError:
  
    raise ImportError('Import Error')
np_float = np.float
np_zeros = np.zeros
np_power=np.power
np_reshape = np.reshape
np_sum=np.sum
np_max = np.max
np_min=np.min
np_arange=np.arange
np_histogram=np.histogram
np_uint8=np.uint8
np_uint16=np.uint16
    
class SegmentationEvaluation(object):
    """
    classe possedant 2 attributs
    - image 1
    - image 2
    """
    def __init__(self, img_1,img_1_L1, img_2):
        """
        Similarity constructor        
        
        Parameters
        ----------
        :param *SpatialImage* img_1: input ``SpatialImage Segmented``

        :param *SpatialImage* img_2: input ``SpatialImage gray level``        
        """

        self.img_1,self.img_1_L1, self.img_2 =img_1,img_1_L1, img_2 
        conds_img = (isinstance(img_1,SpatialImage) and isinstance(img_1_L1,SpatialImage) and
                     isinstance(img_2,SpatialImage))
        if conds_img:
            self.shape_1,self.shape_1_L1, self.shape_2 = img_1.get_shape(),img_1_L1.get_shape(), img_2.get_shape()
            print self.shape_1,self.shape_1_L1, self.shape_2
            if self.shape_1==self.shape_2 and self.shape_1==self.shape_1_L1:
                self.arr_1,self.arr_1_L1, self.arr_2 = img_1.get_array().astype(np_float), img_1_L1.get_array().astype(np_float),img_2.get_array().astype(np_float)
                self.vox_1,self.vox_2 = img_1.get_voxelsize(),img_2.get_voxelsize()
                
                if self.img_1.get_dim()==2:
                    self.len1D = self.shape_1[0]*self.shape_1[1]
                   
                elif self.img_1.get_dim()==3:
                    self.len1D = self.shape_1[0]*self.shape_1[1]*self.shape_1[2]
                    
                self.sig_1_1D,self.sig_1_L1_1D, self.sig_2_1D = np_reshape(self.arr_1,(self.len1D,1)),np_reshape(self.arr_1_L1,(self.len1D,1)),np_reshape(self.arr_2,(self.len1D,1))                 
                self.iter1D = xrange(self.len1D)    
                self.nbre_cellules=len(np.unique(self.sig_1_1D))
                self.nbre_cellules_L1=len(np.unique(self.sig_1_L1_1D))
                self.etiq_max=int(np.max(self.sig_1_1D))
                self.etiq_max_L1=int(self.sig_1_L1_1D.max())
#                for i in self.sig_1_L1_1D :
#                    if 
                print ('voxel',self.vox_2)
                volume_cell=self.vox_2[0]*self.vox_2[1]*self.vox_2[2]
                print ('volume voxel',volume_cell)
                print ('val max de l etiquette',self.etiq_max)
                print ('nombre de cellules',self.nbre_cellules)
                print ('val max de l etiquette L1',self.etiq_max_L1)
                print ('nombre de cellules L1',self.nbre_cellules_L1)
#                
            else: 
                print('TODO')               
        
# calcul de la valeur moyenne de l'image en niveau de gris              


    def compute_val_moy(self):
        """0
        Compute mean value of difference image
        
        Returns
        ----------
        :return: float -- mean value of difference image
        """
        return self.img_2.get_metadata()['mean']
        
#### Calcul de volume intra cellule ####
        
    def compute_intra_volume(self,label=None):
       
        nbre_pixel_par_cell=np.ones((self.etiq_max+1,1),dtype=np.float)
        volume_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
  
        if label=='L1':
            sig_1=self.sig_1_L1_1D
            print ('l1')
        elif label=='L2':
            print ('l2')
        elif label==None :
            sig_1=self.sig_1_1D 
        ## calcul de la moyenne de chaque cellule        
        for i in self.iter1D:
            ind_cell=int(sig_1[i])            
            nbre_pixel_par_cell[ind_cell]+= 1
        volume_cell=nbre_pixel_par_cell*self.vox_2[0]*self.vox_2[1]*self.vox_2[2]
        print ('door')
        return volume_cell
 
 #### Calcul la variance intra cellule ####
        
    def compute_intra_variance(self,label=None):
        if label=='L1':
            sig_1=self.sig_1_L1_1D
            print ('l1')
        elif label=='L2':
            print ('l2')
        elif label==None :
            print('l1l2l3')
            sig_1=self.sig_1_1D    
        ## Calcul de la moyenne de chaque classe
        somme_par_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
        nbre_pixel_par_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
        volume_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
        moyenne_par_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
        var_intra_par_cell=np_zeros((self.etiq_max+1,1),dtype=np.float)
        print ('rrr')
        ## calcul de la moyenne de chaque cellule        
        for i in self.iter1D:
            ind_cell=int(sig_1[i])            
            somme_par_cell[ind_cell]=somme_par_cell[ind_cell]+self.sig_2_1D[i]
            print ('rrr')
            nbre_pixel_par_cell[ind_cell]+= 1
       
        for i,nb in enumerate(nbre_pixel_par_cell):
            if (nb!=0):
                moyenne_par_cell[i]=somme_par_cell[i]/float(nb)           
              
         # Calcul de la variance intra cellule

        for i in self.iter1D:
            ind_cell=int(sig_1[i])
            var_intra_par_cell[ind_cell]+=np_power(self.sig_2_1D[i]-moyenne_par_cell[ind_cell],2)/float(nbre_pixel_par_cell[ind_cell])
        
        #Calcul de volume de chaque cellule
        volume_cell=nbre_pixel_par_cell*self.vox_2[0]*self.vox_2[1]*self.vox_2[2]
        #Elimination des cellulules d'etiquette 0 et 1 (le fond)
        print self.etiq_max+1
        print ('vaaar',var_intra_par_cell)
        for i in np.arange(self.etiq_max+1):
            print ('label: ',i,'var: ',var_intra_par_cell[i])
        var_intra_par_cell=var_intra_par_cell[2::]

        nbre_pixel_par_cell=nbre_pixel_par_cell[2::]

        volume_cell=volume_cell[2::]
        return var_intra_par_cell,volume_cell
    
#### Calcul la variance intra cellule normalisee ####
        
    def compute_normalise_intra_variance(self):
        var_intra_par_cell_norm=np_zeros((self.etiq_max+1,1),dtype=np.float)
        var_intra_par_cell,volume_cell=self.compute_intra_variance()
        # normalisation de la variance intra-cellule
        val_min,val_max=np_min(var_intra_par_cell),np_max(var_intra_par_cell)
        input_range=[val_min,val_max]
        for i,var in enumerate(var_intra_par_cell) :
            var_intra_par_cell_norm[i]= self.normalise_value(var,input_range)
        return var_intra_par_cell_norm ,volume_cell    
        
#### Calcul de nombre de cellules bad variance
        
    def compute_number_etiq_cell_bad_variance(self,seuil):

        var_intra_par_cell,volume_cell=self.compute_intra_variance()
#        #seuil de mal segmentation
#        seuil =500
        etiq_cell_bad_seg=np.where((var_intra_par_cell>seuil-5)&(var_intra_par_cell<seuil+5))
        etiq_cell_bad_seg=etiq_cell_bad_seg[0]+2
        print ('etiquettes cellules mal segmentees',etiq_cell_bad_seg)
        cell_bad_seg=np.shape(etiq_cell_bad_seg)
        print cell_bad_seg
        nb_cell_bad_seg=cell_bad_seg[0]
        print nb_cell_bad_seg
        print ('nombre cellules mal segmentees',nb_cell_bad_seg)
    
        return nb_cell_bad_seg,etiq_cell_bad_seg 
        
#### Calcul de nombre de cellules mal segmentees
        
    def compute_taux_bad_segmentation(self,seuil):

        nb_cell_bad_seg,etiq_cell_bad_seg=self.compute_number_etiq_cell_bad_segmented(seuil)
        taux_bad_seg=float(nb_cell_bad_seg)/self.nbre_cellules
        print('taux de cellules mal segmentation',taux_bad_seg)
        return taux_bad_seg  

#### Calcul de nombre de cellules bad volume
        
    def compute_number_etiq_cell_bad_volume(self,seuil):

        volume_cell=self.compute_intra_volume()
#        #seuil de mal segmentation
#        seuil =10
        etiq_cell_bad_seg=np.where(volume_cell>seuil)
        etiq_cell_bad_seg=etiq_cell_bad_seg[0]
        print ('etiquettes cellules mal segmentees',etiq_cell_bad_seg)
        cell_bad_seg=np.shape(etiq_cell_bad_seg)
        print cell_bad_seg
        nb_cell_bad_seg=cell_bad_seg[0]
        print nb_cell_bad_seg
        print ('nombre cellules mal segmentees',nb_cell_bad_seg)
    
        return nb_cell_bad_seg,etiq_cell_bad_seg 
        
#### Calcul de nombre de cellules bad variance et volume
        
    def compute_number_etiq_cell_bad_variance_volume(self,seuil_var,seuil_vol):

        var_intra_par_cell,volume_cell=self.compute_intra_variance()
#        #seuil de mal segmentation
#        seuil =10
        etiq_cell_bad_seg_var,etiq_cell_bad_seg_vol=np.where(var_intra_par_cell>seuil_var),np.where(volume_cell<seuil_vol)
     
        etiq_cell_bad_seg_var,etiq_cell_bad_seg_vol=etiq_cell_bad_seg_var[0]+2,etiq_cell_bad_seg_vol[0]+2
        print ('etiquettes cellules mal segmentees',etiq_cell_bad_seg_var,etiq_cell_bad_seg_vol)
        etiq_cell_bad_seg=[i for i in etiq_cell_bad_seg_var if i in etiq_cell_bad_seg_vol]
        cell_bad_seg=np.shape(etiq_cell_bad_seg)
        print cell_bad_seg
        nb_cell_bad_seg=cell_bad_seg[0]
        print nb_cell_bad_seg
        print ('etiquettes cellules mal segmentees',etiq_cell_bad_seg)
        print ('nombre cellules mal segmentees',nb_cell_bad_seg)
    
        return nb_cell_bad_seg,etiq_cell_bad_seg  

### plot intra variance en fct des tailles des cellules          
    def plot_hist_volume(self,x_seg):
       
        var_intra_par_cell,volume_cell =self.compute_intra_variance()
        #volume_cell=np.around(volume_cell,1)
        nbins = len(np.unique(volume_cell) )        
        hist=np_zeros((nbins,1),dtype=np.float)
        
        X = np.unique(volume_cell)
        #histogramme 
        print ('nbins',nbins)
        for i,val in enumerate(X):
            l=np.shape(np.where(volume_cell==val))
  
            hist[i]=l[1]

        len_h=len(hist)
       
        #X=X[1:len_h-2]
        
        #hist=hist[1:len_h-2]
        moyenne=np.mean(hist)
        print np.shape(X),np.shape(hist)
        #bar(X, hist, facecolor='green', edgecolor='white', label=x_seg)
        #plt.errorbar(X, hist,yerr=moyenne, facecolor='green', edgecolor='white', label=x_seg) 
        
        #histogramme cumule
        hist_cumul=np.cumsum(hist)
        volume_median=np.median(volume_cell)
        volume_mean=np.mean(volume_cell)
        
        quartile_3=np.around(np_max(hist_cumul)*0.75)
        ind_3quartile=(np.abs(hist_cumul-quartile_3)).argmin()
        volume_3quartile=X[ind_3quartile]
        max_y= np_max(hist_cumul)        
        volume_max=np_max(X)
        volume_min=np_min(X)
        plt.figure(figsize=(200,100))
        bar(X, hist_cumul, facecolor='green', edgecolor='white', label=x_seg)
        plt.plot([volume_3quartile,volume_3quartile],[0,max_y],'b--',label=u"volume 3eme quartile")
     
        plt.plot([volume_median,volume_median],[0,max_y],'r--',label="volume median")
        plt.plot([volume_mean,volume_mean],[0,max_y],'c--',label="volume moyen")
        
        print ('volume median',volume_median,'volume moyen',volume_mean,'volume 3 Q',volume_3quartile,'volume max', volume_max,'volume min',volume_min)
        plt.ylim([0,max_y+10])
        plt.xlim([0,volume_max+10])
        plt.legend()
        plt.show()
        #x_title=u"Histogramme des volumes des cellules de "+x_seg
        x_title=u"Histogramme cumule des volumes des cellules de "+x_seg
        plt.xlabel(u"Volumes des cellules (micrometres)")
        plt.ylabel(u"Frequence")
        plt.title(x_title)
        #output path
        out_path = '/home/sophie/dev/data_1/segmentation/rigid_registration/figures/' # to save results
        if not os.path.isdir(out_path):
            new_fold = os.path.join('/home/sophie/dev/data_1/segmentation/rigid_registration/figures/')
            os.mkdir(new_fold)
        fig_name = ''.join([out_path,x_title,'.png'])
        savefig(fig_name)
       
        return 
    
### plot intra variance en fct des tailles des cellules          
    def plot_taille_variance(self,x_seg):
       
        var_intra_par_cell,volume_cell =self.compute_intra_variance()
        var_intra_par_cell_L1,volume_cell_L1 =self.compute_intra_variance('L1')
        var_median=np.median(var_intra_par_cell)
        var_mean=np.mean(var_intra_par_cell)
        taille_median=np.median(volume_cell)
        taille_mean=np.mean(volume_cell)
       
        var_max=np_max(var_intra_par_cell)
        taille_max=np_max(volume_cell)
        plt.figure(figsize=(200,100))
        plt.plot(volume_cell,var_intra_par_cell,'bo',label=u"partie interieure")
        plt.plot(volume_cell_L1,var_intra_par_cell_L1,'ro',label="partie L1") 
        plt.plot([0,taille_max],[var_median,var_median],'c',label="variance mediane")
        plt.plot([0,taille_max],[var_mean,var_mean],'c--',label="variance moyenne")
        plt.plot([taille_median,taille_median],[0,var_max],'g',label="volume median")
        plt.plot([taille_mean,taille_mean],[0,var_max],'g--',label="volume moyen")
        max_y= np_max(var_intra_par_cell)
        print max_y
        
        max_x=np_max(volume_cell)
        #max_x=25000*self.vox_2[0]*self.vox_2[1]*self.vox_2[2]
        plt.ylim([0,max_y+5])
        plt.xlim([0,max_x+5])
        
        plt.legend()
        plt.show()
        x_title=u"Variance intra-cellule en fonction des volumes des cellules de "+x_seg
        plt.xlabel(u"Volumes des  cellules (micrometres")
        plt.ylabel(u'Variance')
        plt.title(x_title)
        #output path
        out_path = '/home/sophie/dev/data_1/segmentation/rigid_registration/figures/' # to save results
        if not os.path.isdir(out_path):
            new_fold = os.path.join('/home/sophie/dev/data_1/segmentation/rigid_registration/figures/')
            os.mkdir(new_fold)
        fig_name = ''.join([out_path,x_title,'.png'])
        savefig(fig_name)
       
        return 

### plot intra variance en fct des etiquettes des cellules         
    def plot_volume_variance(self,x_seg):
        
        var_intra_par_cell,volume_cell =self.compute_intra_variance()
        x=np.linspace(0,self.etiq_max+1,self.etiq_max+1,endpoint=True)
        figure
        plt.plot(x,var_intra_par_cell,'o')      
        max_y= np_max(var_intra_par_cell)
        plt.ylim([0,max_y+1])
        plt.legend()        
        plt.show()
        x_title=u"Variance intra-cellule de "+x_seg
        plt.xlabel(u"Numero de l'etiquette")
        plt.ylabel('Variance')
        plt.title(x_title)
        #output path
        out_path = '/home/sophie/dev/data_1/segmentation/rigid_registration/figures/' # to save results
        if not os.path.isdir(out_path):
            new_fold = os.path.join('/home/sophie/dev/data_1/segmentation/rigid_registration/figures/')
            os.mkdir(new_fold)
        fig_name = ''.join([out_path,x_title,'.png'])
        savefig(fig_name)
       
        return
        
### plot intra variance normalisee en fct des etiquettes des cellules          
    def plot_normalise_cellule_variance(self,x_seg):
       
        var_intra_par_cell_norm,volume_cell =self.compute_normalise_intra_variance()
        x=np.linspace(0,self.etiq_max+1,self.etiq_max+1,endpoint=True)
        plt.plot(x,var_intra_par_cell_norm,'o')    
        max_y= np_max(var_intra_par_cell_norm)
        plt.ylim([0,max_y+0.1])
        plt.legend()       
        plt.show()
        x_title=u"Variance normalisee intra-cellule de "+x_seg
        plt.xlabel(u"Numero de l'etiquette")
        plt.ylabel(u'Variance normalisee')
        plt.title(x_title)
       
        return
    
### plot histogramme de l'intra variance       
    def plot_histogram_variance(self,x_seg):
        
        var_intra_par_cell,volume_cell =self.compute_intra_variance()
        print ('taux', np.sum(var_intra_par_cell)/float(np.sum(volume_cell)))
        #tri de tableau de la variance
        #var_intra_par_cell_sort=np.sort(var_intra_par_cell,0)
        #recuperation de l'ordre des etiquettes apres le tri des variances
#        etiq_sort=np.argsort(var_intra_par_cell,0)
        #var_intra_par_cell=np.around(var_intra_par_cell)
        
        
       

        ##Calcul de l'histogramme
        #around en entier
        
        #around par 10 ()
        var_intra_par_cell=np.around(var_intra_par_cell,decimals=-1)
        X = np.unique(var_intra_par_cell)
        print np.shape(X)
        nbins = len(np.unique(var_intra_par_cell) )
        print ('nbins',nbins)
        hist=np_zeros((nbins,1),dtype=np.float)
        for i,val in enumerate(X):
            l=np.shape(np.where(var_intra_par_cell==val))
            hist[i]=l[1]
        
        plt.figure(figsize=(200,100))
        max_y= np_max(hist)
        var_max=np_max(X)
        bar(X, hist,facecolor='green', edgecolor='white', label=x_seg)
        plt.ylim([0,max_y+10])
        plt.xlim([0,var_max+10])
        plt.legend()       
        show()
        x_title=u"Histogramme  de l'intra-variance de "+x_seg
        plt.xlabel(u"Variance ")
        plt.ylabel(u"Frequence")
        plt.title(x_title)
        len_h=len(hist) 
        print len_h
        #X=X[1:len_h-1]
        #X=X/10.0
        print hist[0:10]
        #hist=hist[1:len_h-1]
        #print hist[0:10]
        
#        #histogramme cumule
#        hist_cumul=np.cumsum(hist)
#        var_median=np.median(var_intra_par_cell)
#        #/10.0
#        var_mean=np.mean(var_intra_par_cell)
#        #/10.0
#        quartile_3=np.around(np_max(hist_cumul)*0.75)
#        ind_3quartile=(np.abs(hist_cumul-quartile_3)).argmin()
#        var_3quartile=X[ind_3quartile]
#        plt.figure(figsize=(200,100))
#
#        max_y= np_max(hist_cumul)
#        plt.figure(figsize=(200,100))
#        bar(X, hist_cumul,facecolor='green', edgecolor='white', label=x_seg)
#        plt.plot([var_3quartile,var_3quartile],[0,max_y],'b--',label=u"variance 3eme quartile")
#        plt.plot([var_median,var_median],[0,max_y],'r--',label="variance mediane")
#        plt.plot([var_mean,var_mean],[0,max_y],'c--',label="variance moyenne")
#        
#        
#        #plt.plot(var_intra_par_cell_sort,hist_cuml,'b.') 
#        #bar(X, hist, facecolor='green', edgecolor='white', label=x_seg)
#        var_min=np_min(X)
#        var_max=np_max(X)
#        print ('var median',var_median,'var moyen',var_mean,'var 3 Q',var_3quartile,'var max', var_max,'var_min',var_min)
#        plt.ylim([0,max_y+10])
#        plt.xlim([0,var_max+10])
#        plt.legend()       
#        show()
#        x_title=u"Histogramme cumule de l'intra-variance de "+x_seg
#        plt.xlabel(u"Variance")
#        plt.ylabel(u"Frequence")
#        plt.title(x_title)
        #output path
        out_path = '/home/sophie/dev/data_1/segmentation/rigid_registration/figures/' # to save results
        if not os.path.isdir(out_path):
            new_fold = os.path.join('/home/sophie/dev/data_1/segmentation/rigid_registration/figures/')
            os.mkdir(new_fold)
        fig_name = ''.join([out_path,x_title,'.png'])
        savefig(fig_name)
       
        return 
#        # Calcul de la variance inter cellule
#        var_inter=0 
#        val_moy=self.compute_dist_moy()
#        for i in arange(1,self.etiq_max+1):
#            ind_cell=int(self.sig_1_1D[i])
#            var_inter[ind_cell]+=np_power(moyenne_par_cell[ind_cell]-val_moy,2)
#        var_inter=var_inter/float(self.len1D)   
#        print ('inter',var_inter)
        
#        #Calcul de facteur d'analyse de la variance
#        fact_var=var_inter/var_intra
#        print ('facteur' , fact_var)
        
#### Image des cellules bad varaiance
    def plot_cell_bad_variance(self,x,seuil):
        
        nb_cell_bad_seg,etiq_cell_bad_seg =self.compute_number_etiq_cell_bad_variance(seuil)
        print etiq_cell_bad_seg
        print np.shape(etiq_cell_bad_seg)
        sig_1=self.sig_1_1D
        sig_2=self.sig_2_1D

        for val in etiq_cell_bad_seg:
            ind=np.where(sig_1==val)
            sig_1[ind]=255
       
            sig_2[ind]=255
        
        out_seg = np_reshape(sig_1.astype(np_uint16),self.shape_1)
        out_gray = np_reshape(sig_2.astype(np_uint8),self.shape_1)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_1)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_2)
        x='var'+str(seuil)+'_'+x
        imsave('/home/sophie/dev/data_2/seg_bad_variance/'+x,out_seg) 
        imsave('/home/sophie/dev/data_2/gray_bad_variance/'+x,out_gray)
        
#### Image des cellules bad varaiance
    def plot_cell_bad_variance(self,x,seuil):
        
        nb_cell_bad_seg,etiq_cell_bad_seg =self.compute_number_etiq_cell_bad_variance(seuil)
        print etiq_cell_bad_seg
        print np.shape(etiq_cell_bad_seg)
        sig_1=self.sig_1_1D
        sig_2=self.sig_2_1D

        for val in etiq_cell_bad_seg:
            ind=np.where(sig_1==val)
            sig_1[ind]=255
       
            sig_2[ind]=255
        
        out_seg = np_reshape(sig_1.astype(np_uint16),self.shape_1)
        out_gray = np_reshape(sig_2.astype(np_uint8),self.shape_1)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_1)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_2)
        x='var'+str(seuil)+'_'+x
        imsave('/home/sophie/dev/data_2/seg_bad_variance/'+x,out_seg) 
        imsave('/home/sophie/dev/data_2/gray_bad_variance/'+x,out_gray)
        
#### Image des cellules bad volume
    def plot_cell_bad_volume(self,x,seuil):
        
        nb_cell_bad_seg,etiq_cell_bad_seg =self.compute_number_etiq_cell_bad_volume(seuil)
        print etiq_cell_bad_seg
        print np.shape(etiq_cell_bad_seg)
        sig_1=self.sig_1_1D
        sig_2=self.sig_2_1D

        for val in etiq_cell_bad_seg:
            ind=np.where(sig_1==val)
            sig_1[ind]=255
       
            sig_2[ind]=255
        
        out_seg = np_reshape(sig_1.astype(np_uint16),self.shape_1)
        out_gray = np_reshape(sig_2.astype(np_uint8),self.shape_1)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_1)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_2)
        x='vol'+str(seuil)+'_'+x
        imsave('/home/sophie/dev/data_2/seg_bad_volume/'+x,out_seg) 
        imsave('/home/sophie/dev/data_2/gray_bad_volume/'+x,out_gray)        
       
        return
        
#### Image des cellules bad varaiance
    def plot_cell_bad_variance_volume(self,x,seuil_var,seuil_vol):
        
        nb_cell_bad_seg,etiq_cell_bad_seg =self.compute_number_etiq_cell_bad_variance_volume(seuil_var,seuil_vol)
        
        print np.shape(etiq_cell_bad_seg)
        sig_1=self.sig_1_1D
        sig_2=self.sig_2_1D

        for val in etiq_cell_bad_seg:
            ind=np.where(sig_1==val)
            sig_1[ind]=255
       
            sig_2[ind]=255
        
        out_seg = np_reshape(sig_1.astype(np_uint16),self.shape_1)
        out_gray = np_reshape(sig_2.astype(np_uint8),self.shape_1)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_1)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_2)
        x='var'+str(seuil_var)+'_vol'+str(seuil_vol)+'_'+x
        imsave('/home/sophie/dev/data_2/seg_bad_variance_volume/'+x,out_seg) 
        imsave('/home/sophie/dev/data_2/gray_bad_variance_volume/'+x,out_gray)

        
    def normalise_value(self,val, input_range,output_range=None):
        """
        """
         
        if output_range is None:
            output_range=[0.0,1.0]
        
        if isinstance(input_range,list) and len(input_range)==2:
            min_res, max_res =output_range[0], output_range[1]
            val_min, val_max = input_range[0], input_range[1]
           
            val_res=(max_res-min_res)*float((val-val_min))/(val_max-val_min)+min_res
            return val_res
        else:
            print('gh')
        return          
#### Image des cellules bad volume
    def plot_defaut_image(self,x):
        
#        labels = np.unique(self.arr_1).tolist() # labels
#        background_id = np.min(labels)
        background_id=1
        print background_id
        output_arr=self.arr_1
        print np.where(self.arr_1==255)
#        output_arr=self.arr_2
#        output_arr[np.where(self.arr_1==background_id)]=1
#
#        #output_arr[np.where(output_arr!=1)]=255
        output_arr[np.where(self.arr_1!=background_id)]=self.arr_2[np.where(self.arr_1!=background_id)]
       
      
        #output_arr[np.where(output_arr==background_id)]=255
        #output_arr[np.where(output_arr==1)]=128
        print np.where(output_arr==background_id)
        output_arr= np.reshape(output_arr.astype(np.uint8),np.shape(output_arr))
        output_arr = SpatialImage(output_arr,voxelsize=self.vox_1)

       
        x='defaut_'+x
        #imsave('/home/sophie/dev/data_1/defaut_image/'+x,output_arr) 
        
       
        return  x 

        #### Image des cellules bad volume
    def compute_taux_variance_erosion(self,obj,obj_er,x):
        
        var_intra_par_cell,volume_cell=obj.compute_intra_variance()
        var_intra_par_cell_er,volume_cell_er=obj_er.compute_intra_variance()
        taux_erosion= np_zeros(np.shape(var_intra_par_cell_er),dtype=np.float)      
        taux_erosion[np.where(var_intra_par_cell_er!=0)]=var_intra_par_cell_er[np.where(var_intra_par_cell_er!=0)]/var_intra_par_cell[np.where(var_intra_par_cell_er!=0)] 
               
        print taux_erosion
        seuil=np.max(taux_erosion)
        seuil=0.8
        etiq=np.where(taux_erosion>seuil)[0]+2
        print taux_erosion[etiq-2]
        print ('nombre etiquette mal',len(etiq))
        print etiq
        output_arr=obj.arr_2
        output_arr[np.where(output_arr==255)]=1
        for val in etiq:
            output_arr[np.where(obj_er.arr_1==val)]=255
        
        output_arr= np.reshape(output_arr.astype(np.uint8),np.shape(output_arr))
        output_arr = SpatialImage(output_arr,voxelsize=obj.vox_1)
       
        x='taux_grad_hess_'+str(seuil)+'_'+x
        imsave('/home/sophie/dev/data_1/taux_erosion/'+x,output_arr) 
       # world.add(image,output_arr)
             
        return 

#### Image des cellules d interet
    def plot_cell_interet(self,x,labels_interet):
        #var_intra_par_cell,nbre_pixel_par_cell,volume_cell=obj.compute_intra_variance()
        # nb_cell_bad_seg,etiq_cell_bad_seg =self.compute_number_etiq_cell_bad_volume(seuil)
        # print etiq_cell_bad_seg
        # print np.shape(etiq_cell_bad_seg)
        
        #labels_interet=np.unique(self.sig_1_1D)
       
        sig_1=self.sig_1_1D
        sig_2=self.sig_2_1D
        for val in labels_interet:

            ind=np.where(sig_1==val)
            val=int(val)
           
        #print ('label: ', val,' variance: ',var_intra_par_cell[val-2] ,'volume: ',volume_cell[val-2] )
            sig_1[ind]=val
       
            sig_2[ind]=val
        ########ou
      #  sig_1=np_zeros((self.len1D,1),dtype=np.int64)
      #  sig_2=self.sig_2_1D
      #  sig_1[np.where(self.sig_1_1D)==1]=self.sig_1_1D[np.where(self.sig_1_1D)==1]
      #  sig_1[np.where(self.sig_1_1D)!=1]=255
      #  for val in labels_interet:
      #       ind=np.where(self.sig_1_1D==val)
      #       val=int(val)
      #       #print ('label: ', val,' variance: ',var_intra_par_cell[val-2] ,'volume: ',volume_cell[val-2] )
      #       sig_1[ind]=val
      # 
      #       sig_2[ind]=val
        out_seg = np_reshape(sig_1.astype(np_uint16),self.shape_1)
        out_gray = np_reshape(sig_2.astype(np_uint16),self.shape_1)
        out_seg = SpatialImage(out_seg,voxelsize=self.vox_1)
        out_gray = SpatialImage(out_gray,voxelsize=self.vox_2)
        x='cell_expert_'+x
        imsave('/home/sophie/dev/data_1/seg_cellule_interet/'+x,out_seg) 
        imsave('/home/sophie/dev/data_1/gray_cellule_interet/'+x,out_gray)  
        world.add(out_gray,"image")
        return     