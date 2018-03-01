Package calculates WOE and IV for scoring models. Main reason to make this package was to create functions that will allow to create SAS style coarse groups during binarization. So basically package allows to get optimal bining for scoring models. Package orginized in scikit style. You should 'initialize', 'fit' model and then 'transpose'

Currently there are following methods:

make_init_split() - uniform bining based on quantiles

tree_optimize() - bining based on desicion tree

MonoMerge() - merge bins is such way that creates monotonic relationship in %bad

ChiMerge() - merge bins based on chi square statistic