"""

"""

import time
import os
import pickle
import yaml
import pylab
import numpy as np
import healpy as hp
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import ugali.utils.projector
import ugali.utils.healpix
import ugali.candidate.associate as associate
import ugali.utils.bayesian_efficiency
import itertools

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

import seaborn as sns
sns.set_style("ticks")
green = sns.light_palette("green")[5]
cyan = sns.light_palette("turquoise")[5]
blue = sns.light_palette("blue")[5]
red = sns.light_palette("red")[5]
pink = sns.light_palette("magenta")[5]

pylab.ion()

############################################################

class surveySelectionFunction:

    def __init__(self, config_file):

        self.config = yaml.load(open(config_file))
        self.algorithm = self.config['operation']['algorithm']
        self.survey = self.config['operation']['survey']

        self.data_real = None
        self.m_fracdet = None
        self.classifier = None

        self.loadMask()
        #self.loadRealResults()
        #self.loadClassifier()

    def loadMask(self):
        path_to_mask = self.config['infile']['mask'] + 'healpix_mask_{}.fits'.format(self.survey)
        print('Loading survey mask from %s ...'%(path_to_mask))
        self.mask = ugali.utils.healpix.read_map(path_to_mask, nest=True)

    def loadPopulationMetadata(self):
        reader = pyfits.open(self.config['infile']['population_metadata'])
        self.data_population = reader[1].data
        
    def loadSimResults(self):
            reader = pyfits.open(self.config[self.algorithm]['sim_results'])
            self.data_sim = reader[1].data
            reader.close()
    
    def loadRealResults(self):
        if self.data_real is None:
            print('Loading real data search results from %s ...'%(self.config[self.algorithm]['real_results']))
            reader = pyfits.open(self.config[self.algorithm]['real_results'])
            self.data_real = reader[1].data
            reader.close()

    def trainClassifier(self):
        """
        Self-consistently train the classifier
        """

        #self.loadPopulationMetadata()
        self.loadSimResults()
        
        #Clean up results
        nnanidx = np.logical_and(~np.isnan(self.data_sim['TS']),~np.isnan(self.data_sim['SIG']))
        ninfidx = np.logical_and(~np.isinf(self.data_sim['TS']),~np.isinf(self.data_sim['SIG']))
        self.data_sim = self.data_sim[np.logical_and(nnanidx,ninfidx)]
        processed_ok = np.logical_or(self.data_sim['FLAG']==0, self.data_sim['FLAG']==8)
        self.data_sim = self.data_sim[processed_ok]
        self.data_sim = self.data_sim[self.data_sim['DIFFICULTY']==0]
        
        pix = ugali.utils.healpix.angToPix(4096, self.data_sim['ra'], self.data_sim['dec'], nest=True)
        
        # Survey masks
        cut_ebv = np.where(self.mask[pix] & 0b00001, False, True)
        cut_associate = np.where(self.mask[pix] & 0b00010, False, True)
        cut_dwarfs = np.where(self.mask[pix] & 0b00100, False, True)
        cut_bsc = np.where(self.mask[pix] & 0b01000, False, True)
        cut_footprint = np.where(self.mask[pix] & 0b10000, False, True)
        cut_bulk = cut_ebv & cut_footprint & cut_associate & cut_bsc
        
        # Other cuts (modulus, size, shape)
        if self.survey == 'ps1':
            cut_modulus = (self.data_sim['DISTANCE_MODULUS'] < 21.75)
        elif self.survey == 'des':
            cut_modulus = (self.data_sim['DISTANCE_MODULUS'] < 23.5)  
            
        cut_final = cut_bulk & cut_dwarfs & cut_modulus # & cut_sig
        self.data_sim = self.data_sim[cut_final]  
        
        # Detection thresholds
        cut_detect_sim_results_sig = self.data_sim['SIG'] >= self.config[self.algorithm]['sig_threshold']
        cut_detect_sim_results_ts = self.data_sim['TS'] >= self.config[self.algorithm]['ts_threshold']
        
        #Construct dataset
        x = []
        for key, operation in self.config['operation']['params_intrinsic']:
            assert operation.lower() in ['linear', 'log'], 'ERROR'
            if operation.lower() == 'linear':
                x.append(self.data_sim[key])
            else:
                x.append(np.log10(self.data_sim[key]))
        X = np.vstack(x).T
        
        #Construct dataset
        mc_source_id_detect = self.data_sim['MC_SOURCE_ID'][cut_detect_sim_results_sig & cut_detect_sim_results_ts]
        cut_detect = np.in1d(self.data_sim['MC_SOURCE_ID'], mc_source_id_detect)
        
        Y = cut_detect
        indices = np.arange(len(X))
        X_train, X_test, Y_train, Y_test, cut_train, cut_test = train_test_split(X,Y,indices,test_size=0.1)
        
        #Train MLP classifier
        if True:
            t_start = time.time()

            mlp = MLPClassifier()
            #mlp = MLPRegressor()
            
            parameter_space = {'alpha': [0.001, 0.005, 0.01, 0.05], 
                               'hidden_layer_sizes': [x for x in itertools.product((50,50),repeat=3)]}
            
            clf = GridSearchCV(mlp, parameter_space, cv=3, verbose=1)#,fit_params={'sample_weight': 1./(0.5 + np.abs(self.data_sim['SIG'][cut_train]-7))})
            self.classifier = clf.fit(X_train, Y_train)
            
            # Print the best hyperparameters:
            print(self.classifier.best_params_)
            t_end = time.time()
            print('  ... training took %.2f seconds'%(t_end - t_start))

            #Save trained classifier
            classifier_data = pickle.dumps(self.classifier)
            writer = open(self.config[self.algorithm]['classifier'], 'w')
            writer.write(classifier_data)
            writer.close()
            
        #Else load classifier
        else:
            self.loadClassifier()
        
        #Evaluate on test set
        y_pred = self.classifier.predict_proba(X_test)[:,1]
        y_pred_label = self.classifier.predict(X_test)
        
        #Confusion matrix
        cm = confusion_matrix(Y_test, y_pred_label)
        nondet_frac = cm[0][0]/(1.0*cm[0][0]+1.0*cm[0][1])
        det_frac = cm[1][1]/(1.0*cm[1][0]+1.0*cm[1][1])

        print('Fraction of non-detections test set labeled correctly: %0.2f' % nondet_frac)
        print('Fraction of detections in test set labeled correctly: %0.2f' % det_frac)

        plt.figure(figsize=(8,6))
        plt.matshow(cm)
        plt.title('Confusion Matrix', fontsize=18, position = (0.5,1.1))
        plt.colorbar()
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16, position = (0.5, -10.5))
        plt.tick_params(labelsize=12)
        plt.show()
        
        #ROC curve and AUC for each class
        BestRFselector = self.classifier.best_estimator_
        y_pred_best = BestRFselector.predict_proba(X_test)
        labels = BestRFselector.classes_
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i,label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(Y_test, y_pred_best[:, i], pos_label=label)
            roc_auc[label] = auc(fpr[label], tpr[label])
            
        plt.figure(figsize=(8,6))
        plt.plot([0, 1], [1, 1], color='red', linestyle='-', linewidth=3, label='Perfect Classifier (AUC = %0.2f)' % (1.0))
        plt.plot(fpr[1], tpr[1], lw=3, label='Random Forest (AUC = %0.2f)' % (roc_auc[1]), color='blue')
        plt.plot([0, 1], [0, 1], color='black', linestyle=':', linewidth=2.5, label='Random Classifier (AUC = %0.2f)' % (0.5))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.025])
        plt.tick_params(labelsize=16)
        plt.xlabel('False Positive Rate', fontsize=20, labelpad=8)
        plt.ylabel('True Positive Rate', fontsize=20, labelpad=8)
        plt.legend(loc="lower right", fontsize=16)
        plt.savefig('ROC_PS1.pdf')

        self.validateClassifier(cut_detect, cut_train, cut_test, y_pred)

    def validateClassifier(self, cut_detect, cut_train, cut_test, y_pred):
        """
        Make some diagnostic plots
        """

        color = {'detect': 'Red',
                 'nondetect': 'Gold',
                 'why_not': 'none',
                 'actual': 'DodgerBlue',
                 'hsc': 'lime'}
        size = {'detect': 5,
                'nondetect': 5,
                'why_not': 35,
                'actual': None,
                'hsc': None}
        marker = {'detect': 'o',
                  'nondetect': 'o',
                  'why_not': 'o',
                  'actual': 's',}
        alpha  = {'detect': None,
                 'nondetect': None,
                 'why_not': None,
                 'actual': None,
                 'hsc': None}
        edgecolor = {'detect': None,
                     'nondetect': None,
                     'why_not': 'magenta',
                     'actual': 'black',
                     'hsc': 'black'}

        import matplotlib
        cmap = matplotlib.colors.ListedColormap(['Gold', 'Orange', 'DarkOrange', 'OrangeRed', 'Red'])
        title = r'$N_{\rm{train}} =$ %i ; $N_{\rm{test}} =$ %i'%(len(cut_train),len(cut_test))

        pylab.figure()
        pylab.xscale('log')
        
        rphys = self.data_sim['r_physical']
        mag = self.data_sim['abs_mag']
        detect = cut_detect
        
        pylab.scatter(1.e3 * rphys[cut_train],
                      mag[cut_train], 
                      c=detect[cut_train].astype(int), vmin=0., vmax=1., s=size['detect'], cmap=cmap, label=None)
        pylab.scatter(1.e3 * rphys[cut_test],
                      mag[cut_test],
                      c=y_pred, edgecolor='black', vmin=0., vmax=1., s=(3 * size['detect']), cmap=cmap, label=None)
        
        colorbar = pylab.colorbar()
        colorbar.set_label('ML Predicted Detection Probability')
        pylab.scatter(0., 0., s=(3 * size['detect']), c='none', edgecolor='black', label='Test')
        pylab.xlim(1., 3.e3)
        pylab.ylim(6., -12.)
        pylab.xlabel('Half-light Radius (pc)')
        pylab.ylabel('M_V (mag)')
        pylab.legend(loc='upper left', markerscale=2)
        pylab.title(title)

        bins = np.linspace(0., 1., 10 + 1)
        centers = np.empty(len(bins) - 1)
        bin_prob = np.empty(len(bins) - 1)
        bin_prob_err_hi = np.empty(len(bins) - 1)
        bin_prob_err_lo = np.empty(len(bins) - 1)
        bin_counts = np.empty(len(bins) - 1)
        for ii in range(0, len(centers)):
            cut_bin = (y_pred > bins[ii]) & (y_pred < bins[ii + 1])
            centers[ii] = np.mean(y_pred[cut_bin])
            n_trials = np.sum(cut_bin)
            n_successes = np.sum(detect[cut_test] & cut_bin)
            efficiency, errorbar = ugali.utils.bayesian_efficiency.bayesianInterval(n_trials, n_successes, errorbar=True)
            bin_prob[ii] = efficiency
            bin_prob_err_hi[ii] = errorbar[1]
            bin_prob_err_lo[ii] = errorbar[0]
            bin_counts[ii] = np.sum(cut_bin)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        sc = ax.scatter(centers, bin_prob, c=bin_counts, edgecolor='red', s=50, cmap='Reds', zorder=999)
        ax.set_xlim(0., 1.)
        ax.set_ylim(0., 1.)
        ax.plot([0., 1.], [0., 1.], c='black', ls='--')
        ax.errorbar(centers, bin_prob, yerr=[bin_prob_err_lo, bin_prob_err_hi], c='red')
        ax.plot(centers, bin_prob, c='red')
        ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_xticklabels([r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
        ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_yticklabels([r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
        ax.set_xlabel('Predicted Detection Probability',fontsize=22)
        ax.set_ylabel('Fraction Detected',fontsize=22)
        ax.set_title(title,fontsize=22)
        cbar = plt.colorbar(sc)
        cbar.set_label(r'Counts',size=20,labelpad=4)
        cbar.ax.tick_params(labelsize=16) 
        plt.tight_layout()
        plt.savefig('training_PS1.pdf')
        plt.show()

    def loadClassifier(self):
        print('Loading machine learning classifier from %s ...'%(self.config[self.algorithm]['classifier']))
        if os.path.exists(self.config[self.algorithm]['classifier'] + '.gz') and not os.path.exists(self.config[self.algorithm]['classifier']):
            os.system('gunzip -k %s.gz'%(self.config[self.algorithm]['classifier']))
        reader = open(self.config[self.algorithm]['classifier'])
        classifier_data = ''.join(reader.readlines())
        reader.close()
        self.classifier = pickle.loads(classifier_data)

    def predict(self, lon, lat, **kwargs):
        """
        distance, abs_mag, r_physical
        """
        assert self.classifier is not None, 'ERROR'
        
        pred = np.zeros(len(long))
        
        pix = ugali.utils.healpix.angToPix(4096, lon, lat, nest=True)
        mask = ugali.utils.healpix.read_map('/nfs/slac/g/ki/ki21/cosmo/ollienad/Data/DES/healpix_mask_{}.fits'.format(self.survey), nest=True)
        
        cut_ebv = np.where(self.mask[pix] & 0b00001, False, True)
        cut_associate = np.where(self.mask[pix] & 0b00010, False, True)
        cut_dwarfs = np.where(self.mask[pix] & 0b00100, False, True)
        cut_bsc = np.where(self.mask[pix] & 0b01000, False, True)
        cut_footprint = np.where(self.mask[pix] & 0b10000, False, True)
        cut_bulk = cut_ebv & cut_footprint & cut_associate & cut_bsc  
            
        cut_final = cut_bulk & cut_dwarfs # & cut_sig
        self.data_sim = self.data_sim[cut_final]  
    
        x_test = []
        for key, operation in self.config['operation']['params_intrinsic']:
            assert operation.lower() in ['linear', 'log'], 'ERROR'
            if operation.lower() == 'linear':
                x_test.append(kwargs[key])
            else:
                x_test.append(np.log10(kwargs[key]))

        x_test = np.vstack(x_test).T
        pred[cut_final] = self.classifier.predict_proba(x_test[cut_final])[:,1]

        self.validatePredict(pred, cut_final, lon, lat, kwargs['r_physical'], kwargs['abs_mag'], kwargs['distance'])

        return pred, flags_geometry

    def validatePredict(self, pred, cut_final, lon, lat, r_physical, abs_mag, distance):
        import matplotlib
        cmap = matplotlib.colors.ListedColormap(['Gold', 'Orange', 'DarkOrange', 'OrangeRed', 'Red'])

        pylab.figure()
        pylab.scatter(lon, lat, c=flags_geometry, s=10)
        pylab.colorbar()

        pylab.figure()
        pylab.xscale('log')
        pylab.scatter(1.e3 * r_physical[cut_final], 
                      abs_mag[cut_final], 
                      c=pred[cut_final], vmin=0., vmax=1., s=10, cmap=cmap)
        pylab.plot([3,300],[0.0,-10.0],c='black', ls='--')
        pylab.plot([30,1000],[0.0,-7.75],c='black', ls='--')
        pylab.colorbar().set_label('ML Predicted Detection Probability')
        pylab.xlim(1., 3.e3)
        pylab.ylim(6., -12.)
        pylab.xlabel('Half-light Radius (pc)')
        pylab.ylabel('M_V (mag)')

        pylab.figure()
        pylab.xscale('log')
        pylab.scatter(distance[cut_final], 
                      abs_mag[cut_final], 
                      c=pred[cut_final], vmin=0., vmax=1., s=10, cmap=cmap) 
        pylab.colorbar().set_label('ML Predicted Detection Probability')
        pylab.xlim(3., 600.)
        pylab.ylim(6., -12.)
        pylab.xlabel('Distance (kpc)')
        pylab.ylabel('M_V (mag)')

############################################################

if __name__ == "__main__":
    config_file = 'des_y3a2_survey_selection_function.yaml'
    my_ssf = surveySelectionFunction(config_file)

    my_ssf.trainClassifier()
    #my_ssf.loadClassifier()

    #Test with the simulated population, just as an illustration
    config = yaml.load(open(config_file))
    reader_sim = pyfits.open(config['infile']['population_metadata'])
    data_sim = reader_sim[1].data
    reader_sim.close()
    
    #Alternatively, make your own new population
    #distance = 10**np.random.uniform(np.log10(10.), np.log10(400.), n) # kpc
    #abs_mag = np.linspace()
    #r_physical = 10**np.random.uniform(np.log10(0.01), np.log10(1.), n) # kpc

    pred, flags_geometry = my_ssf.predict(lon=data_sim['ra'], lat=data_sim['dec'],
                                          distance=data_sim['distance'],
                                          abs_mag=data_sim['abs_mag'],
                                          r_physical=data_sim['r_physical'])

    #pylab.figure()
    #pylab.scatter(lon, lat, c=pred, s=10)

    """
    # Test
    n = 10000
    lon = np.random.uniform(0., 360., n)
    lat = np.degrees(np.arcsin(np.random.uniform(-1., 1., n)))
    cut_geometry, flags_geometry = my_ssf.applyGeometry(lon, lat)
    pylab.figure()
    pylab.scatter(lon, lat, c=flags_geometry, s=10)
    pylab.colorbar()
    """

    #pylab.figure()
    #pylab.hist(lat, bins=np.linspace(-90., 90., 51))

############################################################