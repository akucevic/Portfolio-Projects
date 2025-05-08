library(lme4)
data <- read.csv('glucose_trial.csv')
model <- lmer(glucose ~ time * (age + race) + (1|subject), data=data)
summary(model)
