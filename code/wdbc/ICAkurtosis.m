wdbc = csvread('wdbc_without_headers.csv');
wdbc_skew =  skewness(wdbc);
wdbc_skew;

wdbc_kurtosis = kurtosis(wdbc);
wdbc_kurtosis;

mu = mean(wdbc);
