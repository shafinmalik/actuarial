# GLM Case Study 1
# Task 1: Convert numeric variables to factors

persinj <- read.csv("persinj.csv")
summary(persinj)


# convert inj to a factor
persinj$inj <- as.factor(persinj$inj)
summary(persinj)

# now inj will display counts instead of numerical summary stats

# Task 2:
# a - split data intro training and test
# b - run OLS model for claim size or a transformation of claim size on training set
# c - calculate RMSE of the model on the test set. This provides benchmark for further model dev

library(caret)
set.seed(2019)

partition <- createDataPartition(y = persinj$amt, p = 0.75, list = FALSE)
data.train <- persinj[partition, ]
data.test <- persinj[-partition, ]
summary(data.train$amt)
summary(data.test$amt)

# these are skewed data sets, so the train and test might look quite different
# fortunately the medians are very close
# note due to how skewed the data is, mean absolute error (MAE) may be more
# useful than RMSE

# our first candidate model is a linear mdel on the log of claim size
# using OLS, but via the glm() function

glm.ols <- glm(log(amt) ~ inj + legrep * op_time,
    family = gaussian(link = "identity"), data = data.train)

summary(glm.ols)

pred.ols <- exp(predict(glm.ols, newdata = data.test))
head(pred.ols)

# example - '3' refers to obs 3, which is the first obs in test set

RMSE(data.test$amt, pred.ols)

RMSE(data.test$amt, mean(data.train$amt))

# ols is an improvement in RMSE compared to null model

# Task 3:
# Evaluate potential combinations of distribution and link fnc
# a - explain why the choices are reasonable
# b - fit 3 models to the training set and calculate their test RMSE
# c - select the best combination, justify your choice

# proposed GLM 1: log link normal GLM on claim size
# now we take a linear model with amt size as target (not log amt)
# however we will use a log link
# so basically normal distribution with log link instead of identity

# this glm ensures positive prediction
# however it allows for possible negative target var obs

glm.log <- glm(amt ~ inj + legrep * op_time,
            family = gaussian(link = "log"), data = data.train)

summary(glm.log)

# by default predict() generates predictions on the scale of
# the linear predictor (log scale here) due to link
# to transform to the original scale use type = "response"

pred.log <- predict(glm.log, newdata = data.test, type = "response")
head(pred.log)
RMSE(data.test$amt, pred.log)

# note that RMSE decreases, so normal GLM with log link beats
# OLS with with log transform.

# GLM 2: log link gamma GLM
# justification for log link:
# ensures all predictions are positive (like the target variable)
# also makes coefficients easy to interpret - they are related to
# multiplicative changes to the linear predictor
# thus the log link is better than the canonical inverse link

glm.gamma <- glm(amt ~ inj + legrep * op_time,
            family = Gamma(link = "log"), data = data.train)

summary(glm.gamma)

RMSE(data.test$amt, predict(glm.gamma, newdata  = data.test, type = "response"))

# GLM 3: Inverse Gaussian GLM
# alternative to gamma glm. Has a fatter right tail. canonical link is inverse square link
# this ensures positive predictions but not easy to predict
# thus log link is more commonly used

# glm.ig <- glm(amt ~ inj + legrep * op_time, data = data.train, family = inverse.gaussian(link = "log"))

# note that the model will fail to converge
# we will stick to the gamma glm for subsequent tasks

par(bg = "#ffffff")
# Task 4: Validate the model
# a - compare the recommended model from task 3 to OLS model from task 2
# b - provide and interpret the diagnostic plots for rec. model to check
# -- model assumptions

# Gamma GLM vs Benchmark OLS
# RMSE of gamma glm is lower than that of OLS model
# 68.6k vs 72.8k
# Summary of gamma glm shows that all coeff estimates are statistically sig.
# Retain all existing features in the model.

# Diagnostics
# residuals vs fitted plot
plot(glm.gamma, which = 1)

# QQ plot
qqnorm(residuals(glm.gamma))
qqline(residuals(glm.gamma))

# Residuals vs Fitted:
# residuals mostly scatter around 0.
# However spread of positive residuals tends to be much higher than the negative residuals
# a few unusually large positive obs.

# QQ plot allows us to assess normality of standardized deviance residuals (not raw resid)
# The model looks problematic on the right end with points deviating significantly upwards.
# There are a lot more large positiuve standardized deviance residuals than under a normal distr.
# the data and deviance residuals are more skewed than the gamma distribution
# a fatter-tailed model may perform better.

# Recall that standardized deviance residuals should be approx normal in glm

# Task 5: Interpret Model
# Run selected model on full dataset
# a - run summary and provide output
# b - interpret the coeff estimates in a way that will provide useful
# -- info to the insurance company or personal injury policyholder

# Note that we use persinj data - running on full data set
glm.final <- glm(amt ~ inj + legrep * op_time,
    family = Gamma(link = "log"), data = persinj)
summary(glm.final)

# exponentiate coefficients to get multiplicative changes since we are using log link
exp(coef(glm.final))

# The model summary shows all features are stat. significant
# the interaction between legrep and op time is also significant
# this means that the effect of operational time on expected claim size
# varies for injuries with and without legal representation
# The expected claim size is muliplied by e^Bj for every unit increase
# in a numeric predictor or when a qual. predictor moves from baseline
# to a new level representend by a dummy var, holding all else constant.

# alternatively, the % change in exp. claim size is e^Bj - 1

# Every injury code except 9 is expected to have a claim size higher than code 1
# code 5 for ex. is est. to be e^1.3396 = 4.0404 times of that with code 1.
# Note we would expect code 6 to have the largest claims out of any inj code
# as this represents fatalities but this is not the case in Gamma. 

# op time is pos. associated with expected claim size, but effect is not 
# as positive for injuries with leg rep. 
# a unit increase in op time is associated with a mult increase of 1.0349 for 
# injuries with legal rep. 

# For most injuries those with legal rep have a higher expected claim size than
# those without leg rep, unless the injuries take an extraordinary long time to settle