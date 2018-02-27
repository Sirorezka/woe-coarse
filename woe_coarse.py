from scipy.stats import chi2, chi2_contingency, pearsonr

## Calculates iv values
##
class iv_splitter():
    def __init__(self,col_dt,target):
        self.col_dt = col_dt
        self.target = target
        
        self.bins_bd = []  ## count good cases
        self.bins_gd = []  ## count bad cases
        self.bins_bd_perc = []  ## count bad percent
        self.bins_gd_perc = []  ## count good percent
        self.bins_woe = []  ## woe by bins
        self.bins_vi = []   ## information value by bin
        self.bins = [] 
        self.coarse_bins = [] 

        ## initial split
        self.make_init_split(q_num = 10)    
        self.global_bd = sum(self.target)
        self.global_gd = len(col_dt)- self.global_bd
        print ("Global bad %f; Global good %f" % (self.global_bd, self.global_gd))

        
    ## split by 10 quantiles
    ##
    def make_init_split(self, q_num = 10):
        self.bins = pd.qcut(self.col_dt,q=q_num,retbins=True,duplicates ='drop')[1]
        self.bins[0] = self.bins[0] - 0.00001 ## for a[i] < x <= a[i+1]
        self.bins = list(self.bins)

    ## get iv for the interval (a,b]
    ##
    def get_iv_int (self, a, b):  
        is_in_bin = (self.col_dt <=b) & (self.col_dt > a)
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
        int1 = (self.col_dt>a) & (self.col_dt<=b)
        int2 = (self.col_dt>b) & (self.col_dt<=c)
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
    def ChiMerge (self):
        bins = self.bins
        if (len(bins)<=2):
            self.coarse_bins = bins
            return 0
        i = 0
        while i<len(bins)-2:
            i = i+1
            a = bins[i-1]
            b = bins[i]
            c = bins[i+1]
            p_val = self.get_chi_int(a,b,c)
            if p_val>0.01:
                bins = bins[:i] + bins[i+1:]
                i=i-1
            ##print (i,a,b,c,p_val)
        self.coarse_bins = bins
        
        ##print (self.bins)
        return 0
    
    
    ##  Merging based on monotonic relationship
    ##
    def MonoMerge (self):

        x = spl_iv.col_dt
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

            ## print (a,b,sum(filt),np.mean(x[filt]),np.mean(y[filt]))


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

    def get_iv_bins(self, bins = None):
        
        if not bins:
            bins = self.bins
            
        assert len(bins)>1
        
        self.bins_bd = []
        self.bins_gd = []
        self.bins_bd_perc = []
        self.bins_gd_perc = []
        self.bins_woe = []
        self.bins_vi = []

        ##print (bins)
        for i in range(len(bins)-1):
            is_in_bin = (self.col_dt <=bins[i+1]) & (self.col_dt > bins[i])
            bin_bd = sum(self.target[is_in_bin])
            bin_gd = sum(is_in_bin) - bin_bd
            bin_bd_perc = bin_bd/self.global_bd
            bin_gd_perc = bin_gd/self.global_gd
            bin_woe = math.log(bin_gd_perc/bin_bd_perc)
            bin_vi = bin_woe * (bin_gd_perc - bin_bd_perc)
            
            self.bins_bd.append(bin_bd)
            self.bins_gd.append(bin_gd)
            self.bins_bd_perc.append(bin_bd_perc)
            self.bins_gd_perc.append(bin_gd_perc)
            self.bins_woe.append(bin_woe)
            self.bins_vi.append(bin_vi)
            
            ##print (i,self.bins[i],self.bins[i+1],bin_bd_perc,bin_gd_perc,bin_woe,bin_vi)
            
        ##print (self.bins_vi)
    
## target_var = 'lbt_inf_total_qty'
## target_var = 'srv_mb_1st_txn_ever_dt'
## target_var = 'lbt_payroll_min_dt'
## target_var = 'lne_tot_issued_ever_rub_amt'

