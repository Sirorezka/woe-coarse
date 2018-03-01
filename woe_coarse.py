from scipy.stats import chi2, chi2_contingency, pearsonr
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection  import cross_val_score
from math import log
import pandas as pd
import numpy as np

## Calculates iv values
##
class iv_splitter():
    
    """
    q_num  - number of quantiles for binning
    bin_type - 'quantile' or 'tree'
    merge_types - 'chimerge', actually can be array of different merging types
    """
    def __init__(self,q_num, bin_type = 'tree', merge_types = None,verbose = False):
        self.x_dt = None
        self.target = None
        self.x_woe = []    ## x produced from woe bins
        self.bin_type = bin_type
        if merge_types is None:
            self.merge_types = []
        else:
            self.merge_types = merge_types
        self.verbose = verbose

        
        self.global_bd = None
        self.global_gd = None
        self.bins_bd = []  ## count good cases
        self.bins_gd = []  ## count bad cases
        self.bins_bd_perc = []  ## count bad percent
        self.bins_gd_perc = []  ## count good percent
        self.bins_woe = []  ## woe by bins
        self.bins_vi = []   ## information value by bin
        self.bins = [] 
        self.coarse_bins = [] 

        ## initial split
        self.q_num = q_num


        
    ## split by 10 quantiles
    ##
    def make_quant_split(self, q_num = 10):
        self.bins = pd.qcut(self.x_dt,q=q_num,retbins=True,duplicates ='drop')[1]
        self.bins[0] = self.bins[0] - 0.00001 ## for a[i] < x <= a[i+1]
        self.bins = list(self.bins)

    ## get iv for the interval (a,b]
    ##
    def get_iv_int (self, a, b):  
        is_in_bin = (self.x_dt <=b) & (self.x_dt > a)
        bin_bd = sum(self.target[is_in_bin])
        bin_gd = sum(is_in_bin) - bin_bd
        bin_bd_perc = bin_bd/self.global_bd
        bin_gd_perc = bin_gd/self.global_gd
        bin_woe = math.log(bin_gd_perc/bin_bd_perc)
        bin_vi = bin_woe * (bin_gd_perc - bin_bd_perc)
        return bin_woe, bin_vi
    
    ##  get chi square for two intervals [a,b],[b,c]
    ##
    def get_chi_int (self,a,b,c):
        int1 = (self.x_dt>a) & (self.x_dt<=b)
        int2 = (self.x_dt>b) & (self.x_dt<=c)
        aa = []
        for i_int in [int1,int2]: 
            a_row = []
            for i_targ in [0,1]: 
                a_row.append(sum(self.target[i_int]==i_targ))
            aa.append(a_row)
        
        ##print (aa)
        try:
            p_val = chi2_contingency(aa)[1] 
        except:
            p_val = -1

        return p_val
      
    ##  Merge intervals based on chi stat
    ##
    def ChiMerge (self, verbose = False):
        bins = self.bins
        if (len(bins)<=2):
            self.coarse_bins = bins
            return 0
        i = 0
        
        if verbose:
                print ("{:>4}".format("id"),
                    "{:>10}".format("bin_a"),"{:>10}".format("bin_b"),"{:>10}".format("bin_c"),
                    "{:>10}".format("p_value"))
                
        while i<len(bins)-2:
            i = i+1
            a = bins[i-1]
            b = bins[i]
            c = bins[i+1]
            p_val = self.get_chi_int(a,b,c)
            if p_val>0.01:
                bins = bins[:i] + bins[i+1:]
                i=i-1
            if (verbose): print ("{:>4}".format(i),
                                 "{:>10}".format(round(a,2)),"{:>10}".format(round(b,2)),"{:>10}".format(round(c,2)),
                                 "{:>10}".format(round(p_val,5)))
                
        self.coarse_bins = bins
        if (verbose): print(self.coarse_bins)

        return 0
    
    
    ##  Merging based on monotonic relationship
    ##
    def MonoMerge (self, verbose = False):

        x = spl_iv.x_dt
        y = spl_iv.target
        bins = spl_iv.bins

        cnt_sum = []
        x_mean = []
        y_mean = []

        ## calculate initial correlation
        for i in range(len(bins)-1):
            a = bins[i]
            b = bins[i+1]
            filt = (x>a) & (x<=b)

            cnt_sum.append(sum(filt))
            x_mean.append(np.mean(x[filt]))
            y_mean.append(np.mean(y[filt]))

            if verbose:
                print (a,b,sum(filt),np.mean(x[filt]),np.mean(y[filt]))


        key_finish = False
        eps = 0.001
        while not key_finish and len(bins)>3:
            max_corr_old = abs(pearsonr(x_mean,y_mean)[0])
            max_corr = max_corr_old
            for i in range(1,len(bins)-1):

                cnt_new = cnt_sum
                x_mean_new = (x_mean[i-1]*cnt_new[i-1] + x_mean[i]*cnt_new[i])/(cnt_new[i-1]+cnt_new[i])
                x_mean_new = x_mean[:i-1] + [x_mean_new] + x_mean[i+1:]

                y_mean_new = (y_mean[i-1]*cnt_new[i-1] + y_mean[i]*cnt_new[i])/(cnt_new[i-1]+cnt_new[i])
                y_mean_new = y_mean[:i-1] + [y_mean_new] + y_mean[i+1:]

                cnt_new = cnt_new[:i-1] + [cnt_new[i-1]+cnt_new[i]] + cnt_new[i+1:]

                new_corr =  abs(pearsonr(x_mean_new,y_mean_new)[0])
                if new_corr >=max_corr+eps:
                    max_corr = new_corr
                    max_x_mean = x_mean_new
                    max_y_mean = y_mean_new
                    max_cnt_sum = cnt_new
                    new_bins = bins[:i] + bins[i+1:]

                    i = i - 1

            if (max_corr>max_corr_old):
                x_mean = max_x_mean
                y_mean = max_y_mean
                cnt_sum = max_cnt_sum
                bins = new_bins
                ##print ("Updated")
                ##print (x_mean)
                ##print (y_mean)
                ##print (bins)
            else:
                key_finish = True
        self.coarse_bins = bins
        return 0

    
    ##  Get bining devision based on tree model
    ##
    def tree_optimize(self, criterion=None, fix_depth=None, qnt_num=10, 
                      max_depth=None, cv=3, min_samples_leaf=50, verbose = False):
            """
            WoE bucketing optimization (continuous variables only)
            :param criterion: binary tree split criteria
            :param fix_depth: use tree of a fixed depth (2^fix_depth buckets)
            :param qnt_num: number of quantiles
            :param max_depth: maximum tree depth for a optimum cross-validation search
            :param cv: number of cv buckets
            :param scoring: scorer for cross_val_score
            :param min_samples_leaf: minimum number of observations in each of optimized buckets
            :return: WoE class with optimized continuous variable split

            __author__ = 'Denis Surzhko'
            """

            tree_type = DecisionTreeClassifier
            m_depth = int(np.log2(qnt_num)) + 1 if max_depth is None else max_depth

            x_train = pd.DataFrame(self.x_dt)
            y_train = self.target

            if (verbose): print ("Tree max depth:", m_depth)
            
            start = 1
            cv_scores = []
            if fix_depth is None:
                for i in range(start, m_depth):
                    if criterion is None:
                        d_tree = tree_type(max_depth=i, min_samples_leaf=min_samples_leaf)
                    else:
                        d_tree = tree_type(criterion=criterion, max_depth=i, min_samples_leaf=min_samples_leaf)

                    scores = cross_val_score(d_tree, x_train, y_train, cv=cv, scoring = 'roc_auc')
                    cv_scores.append(scores.mean())
                    if (verbose): print ("tree: ",i, scores)
                best = np.argmax(cv_scores) + start
            else:
                best = fix_depth
            final_tree = tree_type(max_depth=best, min_samples_leaf=min_samples_leaf)
            final_tree.fit(x_train, y_train)
            opt_bins = final_tree.tree_.threshold[final_tree.tree_.feature >= 0]
            
            opt_bins = np.append(opt_bins,[max(x_train.values),min(x_train.values)-0.00001])
            opt_bins = np.sort(opt_bins)
            
            if (verbose): 
                print("Tree final bins: ")
                print (opt_bins)
                
            self.bins = list(opt_bins)
            self.bins[0] = self.bins[0] - 0.00001 ## for a[i] < x <= a[i+1]


    def fit(self, x_dt, target):
        
        self.x_dt = x_dt
        self.target = target      
        verbose = self.verbose
    
        if self.bin_type == 'tree':
            if (verbose): print ("Doing tree split")
            self.tree_optimize(qnt_num = self.q_num, verbose = verbose)
        else:
            if (verbose): print ("Doing uniform split")
            self.make_quant_split(q_num = self.q_num, verbose = verbose)    
        
        if 'chimerge' in self.merge_types:
            if (verbose): print ("Doing ChiMerge")
            self.ChiMerge(verbose = verbose)
            self.bins = self.coarse_bins
        
        self.global_bd = sum(self.target)
        self.global_gd = len(self.x_dt)- self.global_bd
        if (verbose): print ("Global bad %f; Global good %f" % (self.global_bd, self.global_gd))
        
        ## get bins stat
        self.get_iv_bins(verbose = verbose)
        
        
    ## transpose variable vector
    ## using self.bins
    def transpose(self,x_tr):
        
        x_woe =  np.zeros(len(x_tr))
        for i in range(len(self.bins)-1):
            is_in_bin = (x_tr <=self.bins[i+1]) & (x_tr > self.bins[i])
            x_woe[is_in_bin] = self.bins_woe[i]

        return x_woe
    
    def get_iv_bins(self, bins = None, verbose = False):
        
        if not bins:
            bins = self.bins
            
        assert len(bins)>1
        
        self.bins_bd = []
        self.bins_gd = []
        self.bins_bd_perc = []
        self.bins_gd_perc = []
        self.bins_woe = []
        self.bins_vi = []

        if verbose:
                print (
                    "{:>25}".format("bin"),"{:>12}".format("bad count"),"{:>12}".format("good count"),
                    "{:>10}".format("bad %"),"{:>10}".format("good %"),
                    "{:>6}".format("WOE"),"{:>6}".format("IV"))
                
        ##print (bins)
        for i in range(len(bins)-1):
            is_in_bin = (self.x_dt <=bins[i+1]) & (self.x_dt > bins[i])
            bin_bd = sum(self.target[is_in_bin])
            bin_gd = sum(is_in_bin) - bin_bd
            bin_bd_perc = bin_bd/self.global_bd
            bin_gd_perc = bin_gd/self.global_gd
            if (bin_bd_perc>0):
                bin_woe = log(bin_gd_perc/bin_bd_perc)
            else:
                print ("encountered bin with zero bad rate, 'log' not defined")
                bin_woe = 5
            bin_vi = bin_woe * (bin_gd_perc - bin_bd_perc)
            
            self.bins_bd.append(bin_bd)
            self.bins_gd.append(bin_gd)
            self.bins_bd_perc.append(bin_bd_perc)
            self.bins_gd_perc.append(bin_gd_perc)
            self.bins_woe.append(bin_woe)
            self.bins_vi.append(bin_vi)

            if verbose:
                print ("{:>25}".format(str(round(bins[i],1))+"<x<="+str(round(bins[i+1],1))), 
                       "{:>12}".format(bin_bd),"{:>12}".format(bin_gd),
                       "{:>10}".format(round(bin_bd_perc,2)), "{:>10}".format(round(bin_gd_perc,2)),
                       "{:>6}".format(round(bin_woe,2)), "{:>6}".format(round(bin_vi,2)))
                
            