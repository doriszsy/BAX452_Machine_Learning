
------------------------------------------------------------------------

output: github\_document
------------------------

##### BAX452 HW4

``` r
library(parallel)
library(ggplot2)
library(GGally)
```

    ## Warning: package 'GGally' was built under R version 3.4.3

``` r
library(FNN)
```

    ## Warning: package 'FNN' was built under R version 3.4.3

``` r
library(mvtnorm)
```

### Exercise 1.22

##### 1 Try doing several runs of the linear and k-NN code in that section, comparing results.

``` r
# linear model
library(freqparcoord)
```

    ## Warning: package 'freqparcoord' was built under R version 3.4.3

    ## 
    ##    
    ## 
    ##    
    ## 
    ##    For a quick introduction, type ?freqparcoord, and
    ##    run the examples, making sure to read the comments.
    ##    
    ## 
    ## 

``` r
data(mlb)

xvalpart <- function(data, p) {
  n <- nrow(data)
  ntrain <- round(p*n)
  trainidxs <- sample(1:n, ntrain, replace = FALSE)
  list(train = data[trainidxs,], valid = data[-trainidxs,])
}

xvallm <- function(data, ycol, predvars, p, meanabs = TRUE){
  tmp <- xvalpart(data, p)
  train <- tmp$train
  valid <- tmp$valid
  trainy <- train[, ycol]
  trainpreds <- train[, predvars]
  trainpreds <- as.matrix(trainpreds)
  lmout <- lm(trainy ~ trainpreds)
  validpreds <- as.matrix(valid[, predvars])
  predy <- cbind(1, validpreds) %*% coef(lmout)
  realy <- valid[, ycol]
  if (meanabs) return(mean(abs(predy - realy)))
  list(predy = predy, realy = realy)
}


xvallm(mlb,5 ,c(4, 6), 2/3)
```

    ## [1] 13.89232

``` r
# run 5 times, results are different
#[1] 14.53308
#[1] 14.10442
#[1] 13.39469
#[1] 13.46165
#[1] 13.7847
```

``` r
#k-NN model

library(dummies)
```

    ## dummies-1.5.6 provided by Decision Patterns

``` r
library(regtools)
```

    ## Warning: package 'regtools' was built under R version 3.4.3

    ## Loading required package: car

    ## Warning: package 'car' was built under R version 3.4.3

``` r
library(car)
set.seed(9999)

xvalknn <- function(data, ycol, predvars,k, p, meanabs = TRUE){
  data <- data[, c(predvars, ycol)]
  ycol <- length(predvars) + 1
  tmp <- xvalpart(data, p)
  train <- tmp$train
  valid <- tmp$valid
  valid <- as.matrix(valid)
  xd <- preprocessx(train[, -ycol],k)
  kout <- knnest(train[,ycol], xd,k)
  predy <- predict(kout, valid[,-ycol], TRUE)
  realy <- valid[, ycol]
  if (meanabs) return(mean(abs(predy - realy)))
  list(predy = predy, realy = realy)
}

xvalknn(mlb,5, c(4,6), 25, 2/3)
```

    ## [1] 14.32817

``` r
# run 5 times, results are different
#[1] 14.32817
#[1] 14.13207
#[1] 13.73988
#[1] 14.16343
#[1] 14.1968
#The two methods gave similar results.
```

##### 2 . Extend (1.28) to include interaction terms for age and gender, and age2 and gender.

##### Run the new model, and find the estimated effect of being female, for a 32-year-old person with a Master's degree.

``` r
library(freqparcoord)
data(prgeng)
prgeng$age2 <- prgeng$age^2
edu <- prgeng$educ
prgeng$ms <- as.integer(edu == 14)
prgeng$phd <- as.integer(edu == 16)
prgeng$fem <- prgeng$sex - 1
tmp <- prgeng[edu >= 13,]
pe <- tmp[, c(1,12,9,13,14,15,8)]
pe <- as.matrix(pe)

#adding age and gender, age2 and gender

lm_wage = lm(wageinc ~ age + age2 + wkswrkd + ms + phd + fem + 
     ms:fem + phd:fem + age:fem + age2 : fem, data = prgeng)
summary(lm_wage)
```

    ## 
    ## Call:
    ## lm(formula = wageinc ~ age + age2 + wkswrkd + ms + phd + fem + 
    ##     ms:fem + phd:fem + age:fem + age2:fem, data = prgeng)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -100913  -20401   -4117   12684  291822 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -90276.228   3690.178 -24.464  < 2e-16 ***
    ## age           4327.358    187.142  23.123  < 2e-16 ***
    ## age2           -44.964      2.148 -20.929  < 2e-16 ***
    ## wkswrkd       1193.388     21.880  54.541  < 2e-16 ***
    ## ms           16240.124    846.929  19.175  < 2e-16 ***
    ## phd          25021.190   1759.583  14.220  < 2e-16 ***
    ## fem          33394.352   8219.271   4.063 4.86e-05 ***
    ## ms:fem       -3619.016   1731.531  -2.090  0.03662 *  
    ## phd:fem     -13164.682   4607.748  -2.857  0.00428 ** 
    ## age:fem      -2085.167    403.096  -5.173 2.33e-07 ***
    ## age2:fem        23.075      4.748   4.860 1.18e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 42610 on 20079 degrees of freedom
    ## Multiple R-squared:  0.2372, Adjusted R-squared:  0.2368 
    ## F-statistic: 624.4 on 10 and 20079 DF,  p-value: < 2.2e-16

``` r
predict(lm_wage, data.frame(age = 32,age2 = 32^2,wkswrkd = 52, ms = 1, phd = 0 ,fem = 1))
```

    ##     1 
    ## 67132

``` r
# 32 yrs old female with MS degree earns $67132 wage income
```

##### 3 Use lm() to form a prediction equation for density from the other variables (skipping the first three)

##### comment on whether use of indirect methods in this way seems feasible.

``` r
bodyfat <- read.csv('bodyfat.csv')
lm_density = lm(density ~ age + weight + height + neck + chest + abdomen + hip + thigh + knee + ankle + biceps + forearm + wrist, data = bodyfat)
summary(lm_density)
```

    ## 
    ## Call:
    ## lm(formula = density ~ age + weight + height + neck + chest + 
    ##     abdomen + hip + thigh + knee + ankle + biceps + forearm + 
    ##     wrist, data = bodyfat)
    ## 
    ## Residuals:
    ##       Min        1Q    Median        3Q       Max 
    ## -0.021527 -0.007717  0.000096  0.006491  0.034114 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  1.139e+00  4.030e-02  28.248  < 2e-16 ***
    ## age         -1.203e-04  7.515e-05  -1.601  0.11062    
    ## weight       2.395e-04  1.243e-04   1.926  0.05528 .  
    ## height       1.498e-04  2.230e-04   0.672  0.50243    
    ## neck         1.075e-03  5.401e-04   1.991  0.04765 *  
    ## chest        1.232e-04  2.303e-04   0.535  0.59339    
    ## abdomen     -2.277e-03  2.008e-04 -11.335  < 2e-16 ***
    ## hip          5.513e-04  3.390e-04   1.626  0.10521    
    ## thigh       -6.149e-04  3.354e-04  -1.833  0.06799 .  
    ## knee        -4.844e-05  5.622e-04  -0.086  0.93141    
    ## ankle       -6.314e-04  5.145e-04  -1.227  0.22094    
    ## biceps      -5.755e-04  3.976e-04  -1.448  0.14907    
    ## forearm     -1.017e-03  4.626e-04  -2.198  0.02891 *  
    ## wrist        3.959e-03  1.243e-03   3.185  0.00164 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.01 on 238 degrees of freedom
    ## Multiple R-squared:  0.7381, Adjusted R-squared:  0.7238 
    ## F-statistic:  51.6 on 13 and 238 DF,  p-value: < 2.2e-16

``` r
#from the summary we can see there are several variable are significat (p-value <0.05) predictors of density
#indirect method seems a feasible way since abdomen, wrist, forearm and neck are easier to measure.
```

##### 4

##### (a) Write English prose that relates the overall mean height of people

##### and the gender-specific mean heights.

``` r
#The overall mean height of people is a weighted average of the gender-specifiec mean heights,
#with the weight for each gender being its proportion of the overall population.
```

##### (b) Write English prose that relates the overall proportion of people

##### taller than 70 inches to the gender-specific proportions.

``` r
#The overall proportion of people taller than 70 inches is composed by 
#proportion of female taller than 70 inches and proportion of male taller than 70 inches
```

### Exercise 2.14

##### 1 Consider the census data

##### (a) Form an approximate 95% confidence interval for b6 in the model

``` r
library(freqparcoord)
data(prgeng)
prgeng$age2 <- prgeng$age^2
edu <- prgeng$educ
prgeng$ms <- as.integer(edu == 14)
prgeng$phd <- as.integer(edu == 16)
prgeng$fem <- prgeng$sex - 1
tmp <- prgeng[edu >= 13,]
pe <- tmp[, c(1,12,9,13,14,15,8)]
pe <- as.matrix(pe)

lm2_wage = lm(wageinc ~ age + age2 + wkswrkd + ms + phd + fem, data = prgeng)
summary(lm2_wage)
```

    ## 
    ## Call:
    ## lm(formula = wageinc ~ age + age2 + wkswrkd + ms + phd + fem, 
    ##     data = prgeng)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -98563 -20332  -4273  12781 290808 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -81136.70    3284.75  -24.70   <2e-16 ***
    ## age           3900.35     168.77   23.11   <2e-16 ***
    ## age2           -40.33       1.95  -20.68   <2e-16 ***
    ## wkswrkd       1196.39      21.89   54.65   <2e-16 ***
    ## ms           15431.07     738.80   20.89   <2e-16 ***
    ## phd          23183.97    1626.70   14.25   <2e-16 ***
    ## fem         -11484.49     705.30  -16.28   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 42650 on 20083 degrees of freedom
    ## Multiple R-squared:  0.2356, Adjusted R-squared:  0.2354 
    ## F-statistic:  1032 on 6 and 20083 DF,  p-value: < 2.2e-16

``` r
# CI for b6
b6_l95 = round((-11484.49 - 1.96 * 705.30), 2)
b6_r95 = round((-11484.49 + 1.96 * 705.30), 2)
sprintf("95 percent Confidence Interval for B6 female: (%s, %s)", b6_l95, b6_r95)
```

    ## [1] "95 percent Confidence Interval for B6 female: (-12866.88, -10102.1)"

##### (b) Form an approximate 95% confidence interval for the gender effect

##### for Master's degree holders, b6 + b7, in the model.

``` r
lm3_wage = lm(wageinc ~ age + age2 + wkswrkd + ms + phd + fem + 
                +                ms:fem + phd:fem , data = prgeng)
summary(lm3_wage)
```

    ## 
    ## Call:
    ## lm(formula = wageinc ~ age + age2 + wkswrkd + ms + phd + fem + 
    ##     +ms:fem + phd:fem, data = prgeng)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -100361  -20374   -4247   12824  289893 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -81216.778   3283.880 -24.732  < 2e-16 ***
    ## age           3894.320    168.735  23.080  < 2e-16 ***
    ## age2           -40.293      1.949 -20.669  < 2e-16 ***
    ## wkswrkd       1195.309     21.889  54.609  < 2e-16 ***
    ## ms           16433.668    846.655  19.410  < 2e-16 ***
    ## phd          25325.315   1759.281  14.395  < 2e-16 ***
    ## fem         -10276.797    804.498 -12.774  < 2e-16 ***
    ## ms:fem       -4157.253   1728.329  -2.405  0.01617 *  
    ## phd:fem     -14061.635   4605.664  -3.053  0.00227 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 42640 on 20081 degrees of freedom
    ## Multiple R-squared:  0.2361, Adjusted R-squared:  0.2358 
    ## F-statistic: 775.9 on 8 and 20081 DF,  p-value: < 2.2e-16

``` r
#CI for b6+b7
b6b7_l95 = round((-4157.253 - 1.96 * 1728.329), 2)
b6b7_r95 = round((-4157.253 + 1.96 * 1728.329), 2)
sprintf("95 percent Confidence Interval for B6+B7: (%s, %s)", b6b7_l95, b6b7_r95)
```

    ## [1] "95 percent Confidence Interval for B6+B7: (-7544.78, -769.73)"

##### 2 Extend the analysis in Section 2.8.5 to the full data set,

##### adding dummy variables indicating the second and third year.

##### Form an approximate 95% confidence interval for the difference between the coefficients of these two dummies.

``` r
shar <- read.csv('day.csv')
shar$temp2 <- shar$temp^2
shar$clearday <- as.integer(shar$weathersit == 1)
# yr=1 indicates 2nd yr, there's no data for 3rd yr
lm_shar <- lm(registered ~ temp + temp2 + workingday + clearday + yr, data = shar)
summary(lm_shar)
```

    ## 
    ## Call:
    ## lm(formula = registered ~ temp + temp2 + workingday + clearday + 
    ##     yr, data = shar)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -4503.3  -463.8    14.0   481.2  1902.4 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -2453.60     213.92  -11.47   <2e-16 ***
    ## temp         14449.21     906.77   15.94   <2e-16 ***
    ## temp2       -10590.58     912.67  -11.60   <2e-16 ***
    ## workingday     953.34      60.91   15.65   <2e-16 ***
    ## clearday       621.81      59.72   10.41   <2e-16 ***
    ## yr            1716.25      56.68   30.28   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 762.7 on 725 degrees of freedom
    ## Multiple R-squared:  0.7627, Adjusted R-squared:  0.761 
    ## F-statistic:   466 on 5 and 725 DF,  p-value: < 2.2e-16

``` r
yr_l95 = round((1716.25 - 1.96 *  56.68), 2)
yr_r95 = round((1716.25 + 1.96 *  56.68), 2)
sprintf("95 percent Confidence Interval for B6+B7: (%s, %s)", yr_l95, yr_r95)
```

    ## [1] "95 percent Confidence Interval for B6+B7: (1605.16, 1827.34)"

##### 3 Explain why each Di is (k-1)-variate normal,

##### and derive matrix expressions for the mean vector and covariance matrices.

Since Hi is k-variate normal and Dij = Hi,j+1 - Hij, j=1,2,...k-1 at age k-1, Di,k = Hi,k - Hi,k-1, there's no Di at age K Di is (k-1) variate normal mean vector is u = E\[Di\] = \[E\[D1\], E\[D2\],...,E\[Dk-1\]\]^T coveriance matrices is Sum = E\[(Di- u)(Di - u) ^T\] = \[Cov\[Di,Dj\]; 1&lt;= i, j&lt;= k-1\]

##### 4 In the simulation in Section 2.9.3, it is claimed that p2 = 0:50.

##### Confirm this through derivation.

set variance of xj =1 Y= x1+..+Xp and variance of e = p p^2 = 1 - Var(e) / Var(Y) = 1 - Var(e) / (Var(u(X)) + Var(e)) = 1 - p/(1\*p+p) = 0.5
