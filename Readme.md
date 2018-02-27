Package calculates WOE and IV for scoring models. Main reason to make this package was to create functions that will allow to create SAS style coarse groups during binarization. So basically package allows to get optimal bining for scoring models.

Currently there are two methods:

MonoMerge() - merge bins is such a way to create monotonic relationship in %bad

ChiMerge() - merge bins based on chi square statistic